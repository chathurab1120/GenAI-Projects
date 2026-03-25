from pathlib import Path
from dataclasses import dataclass
import chromadb
from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.ingestion.text_chunker import DocumentChunk
from app.retrieval.embeddings import EmbeddingClient

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """A chunk returned from a vector search with its similarity score."""
    chunk_id: str
    content: str
    source_file: str
    similarity_score: float
    chunk_index: int


class VectorStore:
    """
    ChromaDB-backed vector store for policy document chunks.
    Stores embeddings locally at the path defined in settings.
    """

    def __init__(self, embedding_client: EmbeddingClient) -> None:
        settings = get_settings()
        store_path = Path(settings.vector_store_path)
        store_path.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(store_path))
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = embedding_client
        logger.info(
            f"VectorStore ready at {store_path} | "
            f"collection: {settings.chroma_collection_name} | "
            f"existing docs: {self._collection.count()}"
        )

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        """
        Embed and store a list of document chunks.
        Skips chunks that are already in the store by chunk_id.

        Args:
            chunks: List of DocumentChunk objects to store.
        """
        if not chunks:
            logger.warning("add_chunks called with empty list — skipping")
            return

        existing_ids = set(
            self._collection.get(ids=[c.chunk_id for c in chunks])["ids"]
        )
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]

        if not new_chunks:
            logger.info("All chunks already exist in store — skipping")
            return

        texts = [c.content for c in new_chunks]
        embeddings = self._embedder.embed_batch(texts)

        self._collection.add(
            ids=[c.chunk_id for c in new_chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {
                    "source_file": c.source_file,
                    "document_type": c.document_type.value,
                    "chunk_index": c.chunk_index,
                    "total_chunks": c.total_chunks,
                }
                for c in new_chunks
            ],
        )
        logger.info(f"Added {len(new_chunks)} chunks to vector store")

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """
        Search the vector store for chunks relevant to the query.

        Args:
            query: The search query text.
            top_k: Number of top results to return.

        Returns:
            List of RetrievedChunk objects sorted by relevance.
        """
        if self._collection.count() == 0:
            logger.warning("Vector store is empty — run ingestion first")
            return []

        query_embedding = self._embedder.embed_text(query)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        retrieved: list[RetrievedChunk] = []
        for i, chunk_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i]
            similarity = round(max(0.0, 1.0 - distance), 4)
            metadata = results["metadatas"][0][i]

            retrieved.append(RetrievedChunk(
                chunk_id=chunk_id,
                content=results["documents"][0][i],
                source_file=metadata.get("source_file", "unknown"),
                similarity_score=similarity,
                chunk_index=metadata.get("chunk_index", 0),
            ))

        logger.info(
            f"Retrieved {len(retrieved)} chunks for query: "
            f"{query[:60]}..."
        )
        return retrieved

    def count(self) -> int:
        """Return the number of chunks currently in the store."""
        return self._collection.count()
