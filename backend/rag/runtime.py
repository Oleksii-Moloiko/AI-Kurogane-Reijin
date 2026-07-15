"""Runtime retrieval orchestration for chat requests."""

from backend.rag.context import RAGContextBuilder
from backend.rag.models import RAGContext
from backend.rag.search import VectorSearchService
from backend.utils.logger import get_logger


logger = get_logger(__name__)


class RAGRuntime:
    """Retrieve document context for a user message."""

    def __init__(
        self,
        *,
        search_service: VectorSearchService,
        context_builder: RAGContextBuilder,
        enabled: bool = True,
    ) -> None:
        self.search_service = search_service
        self.context_builder = context_builder
        self.enabled = enabled

    def retrieve(
        self,
        query: str,
    ) -> RAGContext:
        if not self.enabled:
            return RAGContext(
                text="",
                sources=(),
            )

        normalized_query = query.strip()

        if not normalized_query:
            return RAGContext(
                text="",
                sources=(),
            )

        try:
            results = self.search_service.search(
                normalized_query
            )
        except Exception:
            # RAG не повинен ламати звичайний чат.
            logger.exception(
                "RAG retrieval failed"
            )

            return RAGContext(
                text="",
                sources=(),
            )

        return self.context_builder.build(
            results
        )