import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import UploadFile
from app.services.document_service import DocumentService
from app.exceptions import InvalidDocumentFormatError, EmptyDocumentError


class TestDocumentService(unittest.IsolatedAsyncioTestCase):
    @patch("app.services.document_service.run_in_threadpool", new_callable=AsyncMock)
    async def test_process_document_success(self, mock_threadpool):
        # Setup mocks para cada paso secuencial del threadpool
        mock_threadpool.side_effect = [
            "Texto simulado",                      # extract_text
            [0.1] * 384,                           # get_embedding
            "legal",                               # classify
            [{"text": "Juan", "label": "PERSON"}], # extract_entities
            None                                   # insert_document
        ]

        mock_repo = MagicMock()
        service = DocumentService(mock_repo)

        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "test.docx"
        mock_file.read = AsyncMock(return_value=b"fake content")

        result = await service.process_document(mock_file)

        self.assertEqual(result["text"], "Texto simulado")
        self.assertEqual(result["category"], "legal")
        self.assertIsInstance(result["embedding"], list)
        self.assertEqual(result["entities"][0]["text"], "Juan")
        self.assertEqual(mock_threadpool.call_count, 5)  # 4 funciones + insert_document

    @patch("app.services.document_service.run_in_threadpool", new_callable=AsyncMock)
    async def test_process_document_empty_text(self, mock_threadpool):
        mock_threadpool.side_effect = [
            "    "  # extract_text (solo espacios)
        ]
        mock_repo = MagicMock()
        service = DocumentService(mock_repo)

        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "test.docx"
        mock_file.read = AsyncMock(return_value=b" ")

        with self.assertRaises(EmptyDocumentError):
            await service.process_document(mock_file)

    @patch("app.services.document_service.run_in_threadpool", new_callable=AsyncMock)
    async def test_process_document_invalid_format(self, mock_threadpool):
        mock_threadpool.side_effect = ValueError("Invalid format")
        mock_repo = MagicMock()
        service = DocumentService(mock_repo)

        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "malformed.txt"
        mock_file.read = AsyncMock(return_value=b"invalid content")

        with self.assertRaises(InvalidDocumentFormatError):
            await service.process_document(mock_file)

    def test_get_document_success(self):
        mock_repo = MagicMock()
        mock_repo.get_document_by_id.return_value = {"id": 1}
        service = DocumentService(mock_repo)

        result = service.get_document(1)
        self.assertEqual(result["id"], 1)

    def test_get_document_not_found(self):
        from app.exceptions import DocumentNotFoundError
        mock_repo = MagicMock()
        mock_repo.get_document_by_id.return_value = {"error": "Not found"}
        service = DocumentService(mock_repo)

        with self.assertRaises(DocumentNotFoundError):
            service.get_document(99)

    @patch("app.services.document_service.get_embedding", return_value=[0.1, 0.2])
    def test_search_success(self, mock_get_embedding):
        mock_repo = MagicMock()
        mock_repo.search_similar_documents.return_value = [{"id": 1}]
        service = DocumentService(mock_repo)

        results = service.search("some query", 1)
        self.assertEqual(results[0]["id"], 1)
        mock_get_embedding.assert_called_once()
