import os


def validate_file(state: dict) -> dict:
    path = state["file_path"]

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    if not path.lower().endswith(".pdf"):
        raise ValueError("Only PDF supported in v1")

    max_bytes = 50 * 1024 * 1024  # 50MB
    size = os.path.getsize(path)
    if size > max_bytes:
        raise ValueError(f"File exceeds 50MB limit ({size / 1024 / 1024:.1f}MB)")

    return {**state, "validated": True}