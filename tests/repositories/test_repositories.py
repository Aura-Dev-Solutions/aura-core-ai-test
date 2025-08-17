import unittest
from unittest.mock import MagicMock
from app.repositories.sqlite_repository import DocumentRepository


class TestDocumentRepository(unittest.TestCase):
    def setUp(self):
        self.repo = DocumentRepository(":memory:")
        self.repo.insert_document = MagicMock()
        self.repo.get_document_by_id = MagicMock(return_value={"id": 1, "text": "text"})
        self.repo.search_similar_documents = MagicMock(return_value=[{"id": 1, "score": 0.9}])

    def test_insert_document(self):
        self.repo.insert_document("test.docx", "text", [0.1, 0.2], "contract", [{"text": "Juan", "label": "PER"}])
        self.repo.insert_document.assert_called_once()

    def test_get_document_by_id(self):
        doc = self.repo.get_document_by_id(1)
        self.assertEqual(doc["id"], 1)

    def test_search_similar_documents(self):
        results = self.repo.search_similar_documents([0.1, 0.2], 1)
        self.assertEqual(len(results), 1)