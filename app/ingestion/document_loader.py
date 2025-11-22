# app/ingestion/document_loader.py
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from loguru import logger
from pathlib import Path
import fitz  # PyMuPDF directamente


class DocumentLoader:
    @staticmethod
    def load(file_path: str) -> list[Document]:
        path = Path(file_path)
        logger.info(f"Loading document: {file_path}")

        try:
            if path.suffix.lower() == ".pdf":

                doc = fitz.open(file_path)
                documents = []
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text = page.get_text("text")
                    blocks = page.get_text("dict")["blocks"]
                    metadata = {
                        "source": path.name,
                        "format": "pdf",
                        "page": page_num + 1,
                        "total_pages": len(doc)
                    }
                    
                    documents.append(Document(
                        page_content=text.strip(),
                        metadata=metadata
                    ))
                doc.close()
                return documents

            elif path.suffix.lower() in [".docx", ".doc"]:
                from langchain_community.document_loaders import Docx2txtLoader
                loader = Docx2txtLoader(file_path)
                return loader.load()

            elif path.suffix.lower() == ".json":
                import json
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                text = json.dumps(data, ensure_ascii=False, indent=2)
                return [Document(
                    page_content=text,
                    metadata={"source": path.name, "format": "json"}
                )]

            else:
                logger.error(f"Unsupported file type: {path.suffix}")
                raise ValueError(f"Unsupported file type: {path.suffix}")

        except Exception as e:
            logger.error(f"Failed to load document {file_path}: {e}")
            raise