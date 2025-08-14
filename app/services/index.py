import faiss, os, json
import numpy as np
from pathlib import Path
from typing import List, Tuple
from app.config import settings
from app.services.embeddings import embed_texts

class FaissIndex:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.docs: List[Tuple[str, str]] = []

    def add(self, doc_id: str, text: str):
        self.docs.append((doc_id, text))
        print(f"DEBUG: Documento agregado. Total docs: {len(self.docs)}")


    def build(self):
        if not self.docs:
            print("DEBUG: No hay documentos para indexar.")
            return
        self.index.reset()
        embeddings = embed_texts([t for _, t in self.docs])
        self.index.add(embeddings)
        print(f"DEBUG: Índice FAISS reconstruido. Total embeddings: {self.index.ntotal}")

    def search(self, query: str, top_k: int = 5):
        q_emb = embed_texts([query])
        scores, idxs = self.index.search(q_emb, top_k)
        results = []
        for score, i in zip(scores[0], idxs[0]):
            if i == -1:
                continue
            doc_id, text = self.docs[i]
            snippet = text[:300].replace("\n", " ")
            results.append({"doc_id": doc_id, "score": float(score), "text_snippet": snippet})
        return results

    def save(self):
        os.makedirs(os.path.dirname(settings.embeddings_path), exist_ok=True)
        faiss.write_index(self.index, settings.embeddings_path + ".faiss")
        meta = [{"doc_id": d, "text": t[:2000]} for d, t in self.docs]
        Path(settings.embeddings_path + ".meta.json").write_text(json.dumps(meta), encoding="utf-8")

    def load(self):
        faiss_path = settings.embeddings_path + ".faiss"
        meta_path = settings.embeddings_path + ".meta.json"
        if Path(faiss_path).exists() and Path(meta_path).exists():
            self.index = faiss.read_index(faiss_path)
            meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
            self.docs = [(m["doc_id"], m["text"]) for m in meta]

_index = FaissIndex()

def get_index() -> FaissIndex:
    return _index
