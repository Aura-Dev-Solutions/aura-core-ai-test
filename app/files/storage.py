import os

def _storage_root() -> str:
    """
    Input:
      - None.

    Process:
      - Ensure storage directory exists under ./data/storage.

    Output:
      - Absolute path to storage root.
    """
    root = os.path.join(os.getcwd(), "data", "storage")
    os.makedirs(root, exist_ok=True)
    return root


def _file_path(document_id: str) -> str:
    """
    Input:
      - document_id: identifier of the document.

    Process:
      - Build a deterministic path for the file.

    Output:
      - Full file path as string.
    """
    return os.path.join(_storage_root(), f"{document_id}.bin")


def save_file(document_id: str, content: bytes) -> None:
    """
    Input:
      - document_id: identifier of the document.
      - content: raw bytes to store.

    Process:
      - Write bytes to disk at the computed path.
    """
    path = _file_path(document_id)
    with open(path, "wb") as f:
        f.write(content)


def load_file(document_id: str) -> bytes:
    """
    Input:
      - document_id: identifier of the document.

    Process:
      - Read bytes from disk using the stored path.

    Output:
      - Raw file bytes.
    """
    path = _file_path(document_id)
    with open(path, "rb") as f:
        return f.read()
