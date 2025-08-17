import fitz
import docx
import json
import io

def extract_text_from_pdf(content: bytes) -> str:
    """Extrae texto de un PDF, manteniendo saltos de página y línea."""
    text = ""
    with fitz.open(stream=content, filetype="pdf") as doc:
        for page in doc:
            page_text = page.get_text("text")
            text += page_text + "\n--- FIN PAGINA ---\n"
    return text

def extract_text_from_docx(content: bytes) -> str:
    """Extrae texto de un DOCX, manteniendo saltos de párrafo."""
    file_stream = io.BytesIO(content)
    document = docx.Document(file_stream)
    paragraphs = [p.text for p in document.paragraphs if p.text.strip() != ""]
    return "\n".join(paragraphs)

def extract_text_from_json(content: bytes) -> str:
    """Extrae el texto de un JSON. Si no hay campo específico, devuelve el JSON plano."""
    data = json.loads(content.decode("utf-8"))
    # Prueba a devolver el campo de texto principal si existe
    if isinstance(data, dict):
        for key in ["text", "texto", "body", "contenido"]:
            if key in data and isinstance(data[key], str):
                return data[key]
    # Si es un JSON plano o lista, conviértelo a string para no perderlo
    return json.dumps(data, ensure_ascii=False, indent=2)

def extract_text(content: bytes, filename: str) -> str:
    """Detecta el tipo de archivo y usa el extractor apropiado."""
    if filename.endswith(".pdf"):
        return extract_text_from_pdf(content)
    elif filename.endswith(".docx"):
        return extract_text_from_docx(content)
    elif filename.endswith(".json"):
        return extract_text_from_json(content)
    else:
        raise ValueError("Formato de archivo no soportado: " + filename)