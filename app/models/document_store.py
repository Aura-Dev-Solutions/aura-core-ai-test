import numpy as np
import faiss
import os

class DocumentStore:
    def __init__(self, dimension=384, index_path="app/data/document_index.faiss", metadata_path="app/data/metadata.json"):
        """Inicializa el almacén de documentos con un índice FAISS.
        dimension: tamaño del embedding (paraphrase-MiniLM-L6-v2 tiene 384)"""
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # Índice L2 para distancia euclidiana
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.metadata = []  # Lista para almacenar info como filename
        self.load_index()

    def load_index(self):
        """Carga un índice existente desde disco si existe."""
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        if os.path.exists(self.metadata_path):
            import json
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)

    def save_index(self):
        """Guarda el índice y metadata a disco."""
        faiss.write_index(self.index, self.index_path)
        import json
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def add_document(self, embedding: np.ndarray, filename: str, additional_info: dict = None):
        """Añade un embedding y su metadata al índice."""
        self.index.add(np.array([embedding], dtype=np.float32))
        self.metadata.append({
            "filename": filename,
            "additional_info": additional_info or {}
        })
        self.save_index()

    def search(self, query_embedding: np.ndarray, k=5):
        """Busca los k documentos más similares al embedding de consulta."""
        distances, indices = self.index.search(np.array([query_embedding], dtype=np.float32), k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                results.append({
                    "filename": self.metadata[idx]["filename"],
                    "distance": distances[0][i],
                    "metadata": self.metadata[idx]["additional_info"]
                })
        return results