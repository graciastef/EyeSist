import io
import os
import shutil
from datetime import datetime, timezone

import cv2
import numpy as np
from clearml import Task
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from azure_storage import (
    delete_manifest_entry,
    delete_ridge_model,
    delete_session_images,
    delete_session_meta,
    upload_calibration_images,
    upload_ridge_model,
    upload_session_meta,
    upsert_manifest_entry,
)
from model import extract_features, predict_base
from training_config import BASE_MODEL_GOOD_ACCURACY, LABELS, SPLIT_TARGETS

ALPHA = 1.0
TEMP_CALIBRATION_DIR = "./tmp/eyesist_calibration"


def _assign_split(manifest: list[dict], num_samples: int) -> str:
    """Assign a dataset split by current session/sample deficit against split targets."""
    counts = {split: 0 for split in SPLIT_TARGETS}
    sample_counts = {split: 0 for split in SPLIT_TARGETS}

    for entry in manifest:
        split = entry.get("split")
        if split in counts:
            counts[split] += 1
            sample_counts[split] += entry.get("num_samples", 0)

    total_sessions = sum(counts.values()) + 1
    total_samples = sum(sample_counts.values()) + num_samples

    best_split, best_deficit = "train", -float("inf")
    for split, target in SPLIT_TARGETS.items():
        deficit = max(
            target - counts[split] / total_sessions,
            target - sample_counts[split] / total_samples,
        )
        if deficit > best_deficit:
            best_deficit = deficit
            best_split = split

    return best_split


def _log_calibration_to_clearml(session_id: str, metrics: dict, label_counts: dict[str, int]) -> None:
    """Log calibration metrics for one session to a standalone ClearML task."""
    try:
        task = Task.init(
            project_name="EyeSist",
            task_name=f"Calibration {session_id}",
            task_type=Task.TaskTypes.training,
            reuse_last_task_id=False,
        )
        task.connect(
            {
                "session_id": session_id,
                "num_samples": metrics["num_samples"],
                "label_counts": label_counts,
            }
        )
        logger = task.get_logger()
        logger.report_single_value("base_accuracy", metrics["base_accuracy"])
        logger.report_single_value("ridge_accuracy", metrics["ridge_accuracy"])
        logger.report_single_value("accuracy_gain", metrics["ridge_accuracy"] - metrics["base_accuracy"])
        print(f"ClearML calibration task created: id={task.id} session={session_id}")
        task.close()
    except Exception as exc:
        print(f"Warning: failed to log calibration to ClearML: {exc}")


def load_local_calibration_crops(session_id: str) -> tuple[dict[str, list[np.ndarray]], list[tuple[str, bytes]]]:
    """Load staged local calibration crops and their raw JPEG bytes for one session."""
    session_dir = os.path.join(TEMP_CALIBRATION_DIR, session_id)
    if not os.path.isdir(session_dir):
        raise FileNotFoundError(f"No local calibration data found for session {session_id}")

    calibration_crops: dict[str, list[np.ndarray]] = {}
    crops_to_upload: list[tuple[str, bytes]] = []

    for label in LABELS:
        label_dir = os.path.join(session_dir, label)
        if not os.path.isdir(label_dir):
            continue

        for filename in sorted(os.listdir(label_dir)):
            file_path = os.path.join(label_dir, filename)
            with open(file_path, "rb") as file_obj:
                jpeg_bytes = file_obj.read()
            crop_bgr = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
            if crop_bgr is None:
                continue
            calibration_crops.setdefault(label, []).append(crop_bgr)
            crops_to_upload.append((label, jpeg_bytes))

    if not calibration_crops:
        raise FileNotFoundError(f"No valid local calibration images found for session {session_id}")

    return calibration_crops, crops_to_upload


def cleanup_local_calibration_data(session_id: str) -> None:
    """Delete all staged local calibration data for a session."""
    shutil.rmtree(os.path.join(TEMP_CALIBRATION_DIR, session_id), ignore_errors=True)


def calibrate_user(
    calibration_crops: dict[str, list[np.ndarray]],
    crops_to_upload: list[tuple[str, bytes]],
    session_id: str,
    manifest: list[dict],
) -> tuple[str, dict, Pipeline]:
    """Train a personalized Ridge model and retain retraining data only when useful."""

    X, y = [], []
    label_counts: dict[str, int] = {}
    base_correct = 0
    base_total = 0

    for label_idx, label in enumerate(LABELS):
        crops = calibration_crops.get(label, [])
        if not crops:
            continue

        label_counts[label] = len(crops)
        for crop_bgr in crops:
            try:
                feat = extract_features(crop_bgr)
                X.append(feat)
                y.append(label_idx)
                base_correct += int(predict_base(crop_bgr) == label)
                base_total += 1
            except Exception as exc:
                print(f"Warning: failed on {label} crop: {exc}")

    X = np.asarray(X)
    y = np.asarray(y)

    if len(X) < 10:
        raise ValueError(f"Need at least 10 calibration samples, got {len(X)}")

    base_accuracy = base_correct / base_total if base_total else 0.0

    ridge_clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", RidgeClassifier(alpha=ALPHA, class_weight="balanced")),
        ]
    )
    ridge_clf.fit(X, y)
    ridge_accuracy = float(ridge_clf.score(X, y))

    buf = io.BytesIO()
    import joblib
    joblib.dump(ridge_clf, buf)
    model_bytes = buf.getvalue()
    delete_session_images(session_id)
    delete_ridge_model(session_id)
    delete_session_meta(session_id)
    delete_manifest_entry(session_id)

    upload_ridge_model(session_id, model_bytes)

    timestamp = datetime.now(timezone.utc).isoformat()
    split = None

    if base_accuracy < BASE_MODEL_GOOD_ACCURACY:
        upload_calibration_images(session_id, crops_to_upload)
        split = _assign_split(
            [entry for entry in manifest if entry.get("session_id") != session_id],
            num_samples=len(X),
        )
        upsert_manifest_entry(
            {
                "session_id": session_id,
                "split": split,
                "timestamp": timestamp,
                "num_samples": int(len(X)),
                "label_counts": label_counts,
            }
        )
    else:
        print(
            f"Session {session_id[:8]}...: base_acc={base_accuracy:.3f} >= "
            f"{BASE_MODEL_GOOD_ACCURACY:.2f} - skipping retraining image upload"
        )

    meta = {
        "session_id": session_id,
        "timestamp": timestamp,
        "num_samples": int(len(X)),
        "label_counts": label_counts,
        "base_accuracy": float(base_accuracy),
        "ridge_accuracy": ridge_accuracy,
        "split": split,
    }
    upload_session_meta(session_id, meta)

    result = {
        "session_id": session_id,
        "base_accuracy": float(base_accuracy),
        "ridge_accuracy": ridge_accuracy,
        "num_samples": int(len(X)),
        "split": split,
        "cached": False,
    }
    _log_calibration_to_clearml(session_id, result, label_counts)
    return session_id, result, ridge_clf
