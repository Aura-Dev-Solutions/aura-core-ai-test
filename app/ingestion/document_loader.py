from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain_core.documents import Document
import json
from pathlib import Path

class DocumentLoader:
    @staticmethod
    def load(file_path: str) -> list[Document]:
        path = Path(file_path)
        if path.suffix.lower() == ".pdf":
            loader = PyMuPDFLoader(file_path)
        elif path.suffix.lower() in [".docx", ".doc"]:
            loader = Docx2txtLoader(file_path)
        elif path.suffix.lower() == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            text = json.dumps(data, ensure_ascii=False, indent=2)
            return [Document(page_content=text, metadata={"source": path.name, "type": "json"})]
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        return loader.load()