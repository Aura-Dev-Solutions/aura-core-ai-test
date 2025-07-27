import sqlite3
import json
import numpy as np


class DocumentRepository:
    def __init__(self, path: str):
        self.db_path = path

    def insert_document(self, filename, text, embedding, category, entities):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO documents (filename, text, embedding, category, entities)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                filename,
                text,
                json.dumps(embedding),
                category,
                json.dumps(entities)
            ))
            conn.commit()

    def get_document_by_id(self, doc_id: int):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, filename, text, embedding, category, entities FROM documents WHERE id = ?',
                           (doc_id,))
            row = cursor.fetchone()
            if row:
                return {
                    "id": row[0],
                    "filename": row[1],
                    "text": row[2],
                    "embedding": json.loads(row[3]),
                    "category": row[4],
                    "entities": json.loads(row[5])
                }
            return {"error": "Document not found"}

    def cosine_similarity(self, vec1, vec2):
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)

    def search_similar_documents(self, query_embedding, top_k=5):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, filename, text, embedding, category, entities FROM documents')
            rows = cursor.fetchall()

            scored_results = []
            for row in rows:
                doc_embedding = json.loads(row[3])
                similarity = self.cosine_similarity(query_embedding, doc_embedding)
                scored_results.append((similarity, {
                    "id": row[0],
                    "filename": row[1],
                    "text": row[2],
                    "embedding": doc_embedding,
                    "category": row[4],
                    "entities": json.loads(row[5])
                }))

            scored_results.sort(key=lambda x: x[0], reverse=True)
            return [item[1] for item in scored_results[:top_k]]
