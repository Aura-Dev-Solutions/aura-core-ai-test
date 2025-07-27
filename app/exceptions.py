class DocumentNotFoundError(Exception):
    """Raised when a document with the given ID is not found."""
    pass


class InvalidDocumentFormatError(Exception):
    """Raised when the uploaded document has an unsupported format."""
    pass


class EmptyDocumentError(Exception):
    """Raised when the uploaded document has no extractable content."""
    pass
