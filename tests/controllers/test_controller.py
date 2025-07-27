import unittest
from unittest.mock import MagicMock, AsyncMock
from fastapi import UploadFile
from app.controllers.document_controller import DocumentController


class TestDocumentController(unittest.IsolatedAsyncioTestCase):
    async def test_process_document_success(self):
        mock_service = MagicMock()
        mock_service.process_document = AsyncMock(return_value={"text": "some text"})

        controller = DocumentController(mock_service)

        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "test.docx"

        result = await controller.process_document(mock_file)
        self.assertIn("text", result)

    async def test_process_document_failure(self):
        mock_service = MagicMock()
        mock_service.process_document = AsyncMock(side_effect=Exception("fail"))

        controller = DocumentController(mock_service)

        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "test.docx"  

        with self.assertRaises(Exception):
            await controller.process_document(mock_file)

    def test_get_document(self):
        mock_service = MagicMock()
        mock_service.get_document.return_value = {"id": 1}
        controller = DocumentController(mock_service)
        doc = controller.get_document(1)
        self.assertEqual(doc["id"], 1)

    def test_search(self):
        mock_service = MagicMock()
        mock_service.search.return_value = [{"id": 1}]
        controller = DocumentController(mock_service)
        results = controller.search("query", 1)
        self.assertEqual(results[0]["id"], 1)
