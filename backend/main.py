from collections import Counter, OrderedDict, deque
import asyncio
import base64
import io
import json
import logging
import os
import shutil
from typing import Any

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from azure_storage import download_backbone, download_ridge_model, ensure_container, load_manifest, ridge_model_exists
from eye_detector import detect_eyes
from model import extract_features, predict_base
from pipeline.step_calibrate_user import (
    TEMP_CALIBRATION_DIR,
    calibrate_user,
    cleanup_local_calibration_data,
    load_local_calibration_crops,
)
from runtime_config import get_runtime_device_str
from training_config import LABELS

logger = logging.getLogger(__name__)

app = FastAPI()
SMOOTHING_WINDOW = 5
TARGET_CALIBRATION_IMAGES = 100
RIDGE_CACHE_MAX_SIZE = 64
RIDGE_MODEL_CACHE: OrderedDict[str, Any] = OrderedDict()



@app.on_event("startup")
def log_runtime_device() -> None:
    logger.info("EyeSist backend starting on device: %s", get_runtime_device_str())
    ensure_container()
    from model import MODEL_PATH
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        f.write(download_backbone())
    logger.info("Backbone loaded from Azure → %s", MODEL_PATH)


_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class Frame(BaseModel):
    image: str




def bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    """Decode encoded image bytes into a BGR OpenCV image."""
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image bytes")
    return img_bgr


def base64_to_bgr(b64_str: str) -> np.ndarray:
    """Decode a base64 or data-URL image string into a BGR OpenCV image."""
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        return bytes_to_bgr(base64.b64decode(b64_str))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {exc}") from exc


def normalize_calibration_label(label: str) -> str:
    """Normalize and validate a calibration label coming from the frontend."""
    normalized = label.strip().lower()
    if normalized not in LABELS:
        raise ValueError(f"Unsupported calibration label: {label}")
    return normalized


def session_temp_dir(session_id: str) -> str:
    """Return the local temp directory for one session's calibration capture."""
    return os.path.join(TEMP_CALIBRATION_DIR, session_id)


def label_temp_dir(session_id: str, label: str) -> str:
    """Return the local temp directory for one session/label calibration capture."""
    return os.path.join(session_temp_dir(session_id), label)


def reset_local_capture_state(session_id: str) -> None:
    """Clear any staged local calibration images for the given session."""
    shutil.rmtree(session_temp_dir(session_id), ignore_errors=True)


def reset_local_label_capture_state(session_id: str, label: str) -> None:
    """Clear any staged local calibration images for one session/label pair."""
    shutil.rmtree(label_temp_dir(session_id, label), ignore_errors=True)


def _write_local_calibration_crop(session_id: str, label: str, jpeg_bytes: bytes, index: int) -> str:
    """Persist one locally staged calibration crop for a session/label."""
    label_dir = label_temp_dir(session_id, label)
    os.makedirs(label_dir, exist_ok=True)
    file_path = os.path.join(label_dir, f"{label}_{index:04d}.jpg")
    with open(file_path, "wb") as file_obj:
        file_obj.write(jpeg_bytes)
    return file_path


async def save_local_calibration_crop(session_id: str, label: str, jpeg_bytes: bytes, index: int) -> str:
    """Persist one locally staged calibration crop without blocking the event loop."""
    return await asyncio.to_thread(_write_local_calibration_crop, session_id, label, jpeg_bytes, index)


def pick_eye_crops(img_bgr: np.ndarray) -> list[np.ndarray]:
    """Return all detected eye crops from a frame for calibration ingestion."""
    detections = detect_eyes(img_bgr)
    if not detections:
        return []
    return [detection["crop"] for detection in detections]


def _cache_ridge_model(session_id: str, ridge_clf: Any) -> Any:
    """Store a Ridge model in the in-memory LRU cache and return it."""
    RIDGE_MODEL_CACHE[session_id] = ridge_clf
    RIDGE_MODEL_CACHE.move_to_end(session_id)
    while len(RIDGE_MODEL_CACHE) > RIDGE_CACHE_MAX_SIZE:
        RIDGE_MODEL_CACHE.popitem(last=False)
    return ridge_clf


def _load_ridge(session_id: str):
    """Load a session Ridge model from memory first, then Azure on cache miss."""
    import joblib

    cached_model = RIDGE_MODEL_CACHE.get(session_id)
    if cached_model is not None:
        RIDGE_MODEL_CACHE.move_to_end(session_id)
        return cached_model

    if not ridge_model_exists(session_id):
        raise FileNotFoundError(f"No Ridge model found for session {session_id}")

    model_bytes = download_ridge_model(session_id)
    return _cache_ridge_model(session_id, joblib.load(io.BytesIO(model_bytes)))


def predict_from_eye_crops(eye_crops: list[np.ndarray], session_id: str | None) -> list[str]:
    """Predict gaze labels from eye crops using a session model when available."""
    if session_id:
        try:
            ridge_clf = _load_ridge(session_id)
            features = [extract_features(crop) for crop in eye_crops]
            pred_indices = ridge_clf.predict(features)
            return [LABELS[pred_idx] for pred_idx in pred_indices]
        except FileNotFoundError:
            pass
    return [predict_base(crop) for crop in eye_crops]


def infer_frame(image_b64: str, session_id: str | None = None) -> dict:
    """Decode and run gaze inference for a base64-encoded frame."""
    return infer_frame_bgr(base64_to_bgr(image_b64), session_id)


def infer_frame_bgr(img_bgr: np.ndarray, session_id: str | None = None) -> dict:
    """Run eye detection and gaze inference on a decoded BGR frame."""
    detections = detect_eyes(img_bgr)
    if not detections:
        return {
            "error": "No eye detected",
            "gaze": None,
            "boxes": [],
            "frame_size": {"width": int(img_bgr.shape[1]), "height": int(img_bgr.shape[0])},
        }

    eye_crops = [detection["crop"] for detection in detections]
    boxes = [detection["box"] for detection in detections]
    gaze = predict_from_eye_crops(eye_crops, session_id)

    return {
        "gaze": gaze,
        "boxes": boxes,
        "frame_size": {"width": int(img_bgr.shape[1]), "height": int(img_bgr.shape[0])},
    }


def smooth_gaze(history: deque[list[str]], gaze: list[str]) -> tuple[list[str], list[float]]:
    """Apply short-window majority-vote smoothing to per-eye predictions."""
    history.append(gaze)
    smoothed: list[str] = []
    confidences: list[float] = []
    for eye_idx in range(len(gaze)):
        labels = [frame_gaze[eye_idx] for frame_gaze in history if len(frame_gaze) > eye_idx]
        label, count = Counter(labels).most_common(1)[0]
        smoothed.append(label)
        confidences.append(count / len(labels))
    return smoothed, confidences


def collapse_gaze(gaze: list[str], confidences: list[float]) -> str:
    """Collapse smoothed per-eye predictions into a single final gaze label."""
    if not gaze:
        raise ValueError("Expected at least one gaze label to collapse.")
    best_idx = max(range(len(gaze)), key=lambda idx: confidences[idx] if idx < len(confidences) else 0.0)
    return gaze[best_idx]


@app.websocket("/ws/gaze")
async def websocket_predict(websocket: WebSocket):
    """Stream prediction frames over WebSocket with optional session personalization."""
    await websocket.accept()
    session_id: str | None = None
    gaze_history: deque[list[str]] = deque(maxlen=SMOOTHING_WINDOW)

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            if message.get("bytes") is not None:
                img_bgr = bytes_to_bgr(message["bytes"])
                result = infer_frame_bgr(img_bgr, session_id)
                if result.get("gaze"):
                    smoothed_gaze, confidences = smooth_gaze(gaze_history, result["gaze"])
                    result["gaze"] = collapse_gaze(smoothed_gaze, confidences)
                else:
                    gaze_history.clear()
                await websocket.send_json(result)
                continue

            if message.get("text") is not None:
                payload = json.loads(message["text"])
                session_id = payload.get("session_id", session_id)
                image = payload.get("image")

                if image:
                    result = infer_frame(image, session_id)
                    if result.get("gaze"):
                        smoothed_gaze, confidences = smooth_gaze(gaze_history, result["gaze"])
                        result["gaze"] = collapse_gaze(smoothed_gaze, confidences)
                    else:
                        gaze_history.clear()
                    await websocket.send_json(result)
                else:
                    await websocket.send_json({"status": "ready", "session_id": session_id})
                continue

            await websocket.send_json({"error": "Unsupported WebSocket payload", "gaze": None})
    except WebSocketDisconnect:
        logger.info("WebSocket /ws/gaze closed")
    except json.JSONDecodeError:
        await websocket.send_json({"error": "Invalid JSON payload", "gaze": None})
    except Exception as exc:
        logger.exception("WebSocket /ws/gaze error")


@app.websocket("/ws/calibration")
async def websocket_calibrate(websocket: WebSocket):
    """Stream calibration frames and stage accepted eye crops locally per label."""
    await websocket.accept()
    session_id: str | None = None
    label: str | None = None

    try:
        init_payload: dict[str, Any] = json.loads(await websocket.receive_text())
        session_id = init_payload["session_id"]
        label = normalize_calibration_label(init_payload["label"])
        if bool(init_payload.get("reset")):
            reset_local_capture_state(session_id)
        reset_local_label_capture_state(session_id, label)

        captured_count = 0
        await websocket.send_json(
            {
                "status": "ready",
                "session_id": session_id,
                "label": label,
                "captured_count": captured_count,
                "target_count": TARGET_CALIBRATION_IMAGES,
            }
        )

        while captured_count < TARGET_CALIBRATION_IMAGES:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            frame_bytes = message.get("bytes")
            if frame_bytes is None:
                continue

            img_bgr = bytes_to_bgr(frame_bytes)
            detections = detect_eyes(img_bgr)
            frame_size = {"width": int(img_bgr.shape[1]), "height": int(img_bgr.shape[0])}
            boxes = [detection["box"] for detection in detections]
            eye_crops = [detection["crop"] for detection in detections]

            if not eye_crops:
                await websocket.send_json(
                    {
                        "status": "retry",
                        "session_id": session_id,
                        "label": label,
                        "captured_count": captured_count,
                        "target_count": TARGET_CALIBRATION_IMAGES,
                        "reason": "No eye detected",
                        "boxes": [],
                        "frame_size": frame_size,
                    }
                )
                continue

            remaining = TARGET_CALIBRATION_IMAGES - captured_count
            saved_any = False
            for crop_bgr in eye_crops[:remaining]:
                ok, jpeg = cv2.imencode(".jpg", crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                if not ok:
                    continue
                await save_local_calibration_crop(session_id, label, jpeg.tobytes(), captured_count)
                captured_count += 1
                saved_any = True

            if not saved_any:
                await websocket.send_json(
                    {
                        "status": "retry",
                        "session_id": session_id,
                        "label": label,
                        "captured_count": captured_count,
                        "target_count": TARGET_CALIBRATION_IMAGES,
                        "reason": "Failed to encode eye crop",
                        "boxes": boxes,
                        "frame_size": frame_size,
                    }
                )
                continue

            await websocket.send_json(
                {
                    "status": "complete" if captured_count >= TARGET_CALIBRATION_IMAGES else "capturing",
                    "session_id": session_id,
                    "label": label,
                    "captured_count": captured_count,
                    "target_count": TARGET_CALIBRATION_IMAGES,
                    "boxes": boxes,
                    "frame_size": frame_size,
                }
            )
    except WebSocketDisconnect:
        logger.info("WebSocket /ws/calibration closed")
    except Exception as exc:
        await websocket.send_json({"status": "error", "detail": str(exc)})
        logger.exception("WebSocket /ws/calibration error")


@app.post("/sessions/{session_id}/calibration")
def calibrate_v2(session_id: str):
    """Finalize calibration from staged local crops and persist the new session model."""
    try:
        calibration_crops, crops_to_upload = load_local_calibration_crops(session_id)
        _, result, ridge_clf = calibrate_user(
            calibration_crops,
            crops_to_upload,
            session_id=session_id,
            manifest=load_manifest(),
        )
        _cache_ridge_model(session_id, ridge_clf)
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        logger.info("Cleaning up local calibration data for session %s", session_id)
        cleanup_local_calibration_data(session_id)


@app.get("/health")
def health():
    return {"status": "ok"}
