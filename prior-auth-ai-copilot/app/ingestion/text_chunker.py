from dataclasses import dataclass
from app.core.constants import CHUNK_SIZE, CHUNK_OVERLAP, DocumentType
from app.core.logging_config import get_logger
from app.ingestion.document_loader import LoadedDocument

logger = get_logger(__name__)


@dataclass
class DocumentChunk:
    """A single text chunk ready to be embedded and stored."""
    chunk_id: str
    content: str
    source_file: str
    document_type: DocumentType
    chunk_index: int
    total_chunks: int


def chunk_documents(
    documents: list[LoadedDocument],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """
    Split a list of loaded documents into overlapping text chunks.

    Args:
        documents: List of LoadedDocument objects.
        chunk_size: Approximate number of characters per chunk.
        chunk_overlap: Number of characters to overlap between chunks.

    Returns:
        List of DocumentChunk objects ready for embedding.
    """
    all_chunks: list[DocumentChunk] = []

    for doc in documents:
        chunks = _split_text(doc.content, chunk_size, chunk_overlap)
        total = len(chunks)

        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{doc.source_file}::chunk_{i:04d}"
            all_chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                content=chunk_text.strip(),
                source_file=doc.source_file,
                document_type=doc.document_type,
                chunk_index=i,
                total_chunks=total,
            ))

        logger.info(
            f"Chunked {doc.source_file} into {total} chunks"
        )

    logger.info(f"Total chunks created: {len(all_chunks)}")
    return all_chunks


def _split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """
    Split text into overlapping chunks by character count.
    Tries to split at newlines to avoid cutting mid-sentence.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            # Try to find a newline near the end to split cleanly
            newline_pos = text.rfind("\n", start, end)
            if newline_pos > start + (chunk_size // 2):
                end = newline_pos

        chunks.append(text[start:end])
        start = end - chunk_overlap

    return chunks
