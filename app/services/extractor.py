import io
import fitz
import docx
import json


def extract_text(file_bytes: bytes, filename: str) -> str:
    if filename.endswith(".pdf"):
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        return "\n".join([page.get_text() for page in doc])
    elif filename.endswith(".docx"):
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join([p.text for p in doc.paragraphs])
    elif filename.endswith(".json"):
        data = json.loads(file_bytes)
        return json.dumps(data, indent=2)
    else:
        raise ValueError("Unsupported file type")
