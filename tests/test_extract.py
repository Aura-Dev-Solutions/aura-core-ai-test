from app.services.extract import extract_text, detect_mime
from pathlib import Path

def test_detect_mime_pdf():
    assert detect_mime(Path("file.pdf")) == "application/pdf"

def test_extract_text_json(tmp_path):
    p = tmp_path / "x.json"
    p.write_text('{"a":1,"b":2}', encoding="utf-8")
    text, mime = extract_text(p)
    assert mime == "application/json"
    assert '"a": 1' in text
