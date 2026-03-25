from app.core.constants import TOP_K_RESULTS
from app.core.logging_config import get_logger
from app.ingestion.policy_ingestor import ingest_policy_documents
from app.retrieval.embeddings import EmbeddingClient
from app.retrieval.vectorstore import VectorStore, RetrievedChunk

logger = get_logger(__name__)


class PolicyRetriever:
    """
    High-level retriever for prior authorization policy documents.
    Automatically ingests policies if the vector store is empty.
    """

    def __init__(self) -> None:
        self._embedder = EmbeddingClient()
        self._store = VectorStore(self._embedder)
        self._ensure_policies_indexed()

    def _ensure_policies_indexed(self) -> None:
        """
        If the vector store is empty, ingest all policy documents.
        This runs automatically on first use.
        """
        if self._store.count() == 0:
            logger.info(
                "Vector store is empty — running policy ingestion..."
            )
            chunks = ingest_policy_documents()
            self._store.add_chunks(chunks)
            logger.info(
                f"Indexed {self._store.count()} chunks into vector store"
            )
        else:
            logger.info(
                f"Vector store ready with {self._store.count()} chunks"
            )

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the most relevant policy chunks for a given query.

        Args:
            query: Search query built from diagnosis + requested service.
            top_k: Number of chunks to return.

        Returns:
            List of RetrievedChunk objects sorted by relevance.
        """
        return self._store.search(query=query, top_k=top_k)
