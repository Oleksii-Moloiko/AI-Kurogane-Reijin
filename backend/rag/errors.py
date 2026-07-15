"""RAG-specific exceptions."""


class DocumentError(Exception):
    """Base exception for document processing errors."""


class UnsupportedDocumentError(DocumentError):
    """Raised when the document type is not supported."""


class DocumentNotFoundError(DocumentError):
    """Raised when the requested document does not exist."""


class DocumentTooLargeError(DocumentError):
    """Raised when the document exceeds the configured size limit."""


class EmptyDocumentError(DocumentError):
    """Raised when no readable text can be extracted."""

class DuplicateDocumentError(DocumentError):
    """Raised when the same document was already indexed."""


class DocumentIndexingError(DocumentError):
    """Raised when document indexing cannot be completed."""