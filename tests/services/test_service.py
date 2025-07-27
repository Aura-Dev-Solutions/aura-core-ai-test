import unittest
from unittest.mock import MagicMock, AsyncMock,patch
from fastapi import UploadFile
from app.services.document_service import DocumentService


class TestDocumentService(unittest.IsolatedAsyncioTestCase):
    @patch("app.services.document_service.extract_text", return_value="Texto simulado")
    @patch("app.services.document_service.get_embedding", return_value=[0.1] * 384)
    @patch("app.services.document_service.classify", return_value="legal")
    @patch("app.services.document_service.extract_entities", return_value=[{"text": "Juan", "label": "PERSON"}])
    async def test_process_document(self, mock_ner, mock_classify, mock_embed, mock_extract):
        mock_repo = MagicMock()
        service = DocumentService(mock_repo)

        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "test.docx"
        mock_file.read = AsyncMock(return_value=b"fake content")

        result = await service.process_document(mock_file)

        self.assertEqual(result["text"], "Texto simulado")
        self.assertEqual(result["category"], "legal")
        self.assertTrue(isinstance(result["embedding"], list))
        self.assertEqual(result["entities"][0]["text"], "Juan")
        mock_repo.insert_document.assert_called_once()

    def test_get_document(self):
        mock_repo = MagicMock()
        mock_repo.get_document_by_id.return_value = {"id": 1}
        service = DocumentService(mock_repo)
        doc = service.get_document(1)
        self.assertEqual(doc["id"], 1)

    def test_search(self):
        mock_repo = MagicMock()
        mock_repo.search_similar_documents.return_value = [{"id": 1}]
        service = DocumentService(mock_repo)
        results = service.search("contract", 1)
        self.assertEqual(results[0]["id"], 1)
