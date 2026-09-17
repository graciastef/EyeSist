import os
import json
import io
from functools import lru_cache

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContentSettings

ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME", "anonifyme")
ACCOUNT_URL =  os.getenv("AZURE_BLOB_URL", f"https://{ACCOUNT_NAME}.blob.core.windows.net")
CONTAINER = "eyesist"

# Blob path constants
BACKBONE_BLOB = "models/backbone/ethxgaze_backbone.pth"
YOLO_BLOB = "models/yolo/yolo_eye.pt"
MANIFEST_BLOB = "dataset/manifest.json"
LAST_RETRAIN_BLOB = "dataset/last_retrain.json"
MODEL_VERSION_BLOB = "models/backbone/version.json"


@lru_cache(maxsize=1)
def _credential():
    return DefaultAzureCredential()


@lru_cache(maxsize=1)
def _client() -> BlobServiceClient:
    return BlobServiceClient(account_url=ACCOUNT_URL, credential=_credential())


def ensure_container() -> None:
    container = _client().get_container_client(CONTAINER)
    if not container.exists():
        container.create_container()


def upload_bytes(blob_path: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    blob = _client().get_blob_client(container=CONTAINER, blob=blob_path)
    blob.upload_blob(data, overwrite=True,
                     content_settings=ContentSettings(content_type=content_type))


def upload_json(blob_path: str, data: dict) -> None:
    upload_bytes(blob_path, json.dumps(data).encode(), "application/json")


def upload_file(blob_path: str, local_path: str) -> None:
    with open(local_path, "rb") as f:
        upload_bytes(blob_path, f.read())


def download_bytes(blob_path: str) -> bytes:
    blob = _client().get_blob_client(container=CONTAINER, blob=blob_path)
    return blob.download_blob().readall()


def download_json(blob_path: str) -> dict:
    return json.loads(download_bytes(blob_path))


def download_file(blob_path: str, local_path: str) -> None:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(download_bytes(blob_path))


def blob_exists(blob_path: str) -> bool:
    blob = _client().get_blob_client(container=CONTAINER, blob=blob_path)
    return blob.exists()


def list_blobs(prefix: str) -> list[str]:
    container = _client().get_container_client(CONTAINER)
    return [b.name for b in container.list_blobs(name_starts_with=prefix)]


def list_session_images(session_id: str) -> list[str]:
    return sorted(list_blobs(f"calibration-data/{session_id}/images/"))


def upload_calibration_images(session_id: str, crops: list[tuple[str, bytes]]) -> None:
    """Upload list of (label, jpeg_bytes) crops for a session."""
    label_counts: dict[str, int] = {}
    for label, jpeg_bytes in crops:
        idx = label_counts.get(label, 0)
        label_counts[label] = idx + 1
        blob_path = f"calibration-data/{session_id}/images/{label}/{idx:04d}.jpg"
        upload_bytes(blob_path, jpeg_bytes, "image/jpeg")


def delete_ridge_model(session_id: str) -> None:
    blob_path = f"calibration-data/{session_id}/ridge_model.pkl"
    if blob_exists(blob_path):
        delete_blob(blob_path)


def delete_session_meta(session_id: str) -> None:
    blob_path = f"calibration-data/{session_id}/meta.json"
    if blob_exists(blob_path):
        delete_blob(blob_path)


def upload_session_meta(session_id: str, meta: dict) -> None:
    upload_json(f"calibration-data/{session_id}/meta.json", meta)


def download_session_meta(session_id: str) -> dict:
    return download_json(f"calibration-data/{session_id}/meta.json")


def load_manifest() -> list[dict]:
    """Return the full dataset manifest, or [] if it doesn't exist yet."""
    try:
        return download_json(MANIFEST_BLOB)
    except Exception:
        return []


def append_to_manifest(entry: dict) -> None:
    """Append one session entry to the manifest. Thread-unsafe — fine for sequential ingestion."""
    manifest = load_manifest()
    manifest.append(entry)
    upload_json(MANIFEST_BLOB, manifest)


def upsert_manifest_entry(entry: dict) -> None:
    session_id = entry.get("session_id")
    manifest = [item for item in load_manifest() if item.get("session_id") != session_id]
    manifest.append(entry)
    upload_json(MANIFEST_BLOB, manifest)


def delete_manifest_entry(session_id: str) -> None:
    manifest = [item for item in load_manifest() if item.get("session_id") != session_id]
    upload_json(MANIFEST_BLOB, manifest)


def load_last_retrain_info() -> dict:
    """Return info about the last completed retrain, or defaults if none yet."""
    try:
        return download_json(LAST_RETRAIN_BLOB)
    except Exception:
        return {"timestamp": None, "num_train_sessions": 0, "num_train_samples": 0}


def save_last_retrain_info(info: dict) -> None:
    print(f"Saving last retrain info: {info}")
    upload_json(LAST_RETRAIN_BLOB, info)


def update_session_meta(session_id: str, updates: dict) -> None:
    """Patch an existing session meta.json with the given fields."""
    print(f"Updating meta for session {session_id}: {updates}")
    meta = download_session_meta(session_id)
    meta.update(updates)
    upload_session_meta(session_id, meta)


def upload_backbone(data: bytes) -> None:
    """Overwrite the production backbone checkpoint on Azure."""
    print("Uploading backbone checkpoint to Azure…") 
    upload_bytes(BACKBONE_BLOB, data)


def download_backbone() -> bytes:
    print("Downloading backbone checkpoint from Azure…")
    return download_bytes(BACKBONE_BLOB)


def delete_blob(blob_path: str) -> None:
    print(f"Deleting blob {blob_path}…")
    blob = _client().get_blob_client(container=CONTAINER, blob=blob_path)
    blob.delete_blob()


def upload_ridge_model(session_id: str, model_bytes: bytes) -> None:
    print(f"Uploading ridge model for session {session_id}…")
    upload_bytes(f"calibration-data/{session_id}/ridge_model.pkl", model_bytes)


def download_ridge_model(session_id: str) -> bytes:
    print(f"Downloading ridge model for session {session_id}…")
    return download_bytes(f"calibration-data/{session_id}/ridge_model.pkl")


def ridge_model_exists(session_id: str) -> bool:
    exists =  blob_exists(f"calibration-data/{session_id}/ridge_model.pkl")
    print(f"Ridge model exists for session {session_id}: {exists}")
    return exists

def delete_session_images(session_id: str) -> None:
    """Delete all calibration images for a session (keeps meta and ridge model)."""
    blobs = list_session_images(session_id)
    for blob_path in blobs:
        try:
            delete_blob(blob_path)
        except Exception as e:
            print(f"  Warning: could not delete {blob_path}: {e}")
    print(f"Deleted {len(blobs)} images for session {session_id[:8]}…")


def load_model_version() -> dict:
    """Return normalized model version info, with backward compatibility for old schema."""
    try:
        print("Loading current model version info from Azure…")
        info = download_json(MODEL_VERSION_BLOB)
        return {
            "latest_version": info.get("version", 0),
            "latest_promoted_blob": info.get("promoted_blob"),
            "latest_promoted_timestamp": info.get("timestamp"),
            "latest_candidate_test_acc": info.get("candidate_test_acc"),
            "latest_production_test_acc": info.get("production_test_acc"),
            "production_blob": BACKBONE_BLOB,
            "production_version": None,
        }
    except Exception:
        return {
            "latest_version": 0,
            "latest_promoted_blob": None,
            "latest_promoted_timestamp": None,
            "latest_candidate_test_acc": None,
            "latest_production_test_acc": None,
            "production_blob": BACKBONE_BLOB,
            "production_version": None,
        }


def save_model_version(info: dict) -> None:
    print(f"Saving model version info: {info}")
    upload_json(MODEL_VERSION_BLOB, info)


def list_all_session_metas() -> list[dict]:
    """Return all session meta.json contents, with session_id injected from blob path."""
    metas = []
    for blob_path in list_blobs("calibration-data/"):
        if blob_path.endswith("/meta.json"):
            try:
                meta = download_json(blob_path)
                if "session_id" not in meta:
                    meta["session_id"] = blob_path.split("/")[1]
                metas.append(meta)
            except Exception:
                pass
    return metas
