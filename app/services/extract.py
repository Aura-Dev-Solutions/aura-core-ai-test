from pathlib import Path
from typing import Tuple
from pypdf import PdfReader
from docx import Document as DocxDocument
import json

def detect_mime(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return "application/pdf"
    if ext == ".docx":
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if ext == ".json":
        return "application/json"
    return "application/octet-stream"

def extract_text(path: Path) -> Tuple[str, str]:
    mime = detect_mime(path)
    if mime == "application/pdf":
        reader = PdfReader(str(path))
        text = "\n".join([p.extract_text() or "" for p in reader.pages])
        return text, mime
    if mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = DocxDocument(str(path))
        text = "\n".join([p.text for p in doc.paragraphs])
        return text, mime
    if mime == "application/json":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        text = json.dumps(data, ensure_ascii=False)
        return text, mime
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    return txt, mime
