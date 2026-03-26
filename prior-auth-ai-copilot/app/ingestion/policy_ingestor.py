from app.core.constants import DocumentType
from app.core.logging_config import get_logger
from app.ingestion.document_loader import load_documents_from_folder
from app.ingestion.text_chunker import chunk_documents, DocumentChunk
from pathlib import Path

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent
POLICY_DOCS_PATH = str(_PROJECT_ROOT / "data" / "policy_docs")
CLINICAL_NOTES_PATH = str(_PROJECT_ROOT / "data" / "clinical_notes")


def ingest_policy_documents() -> list[DocumentChunk]:
    """
    Load and chunk all policy documents from data/policy_docs.

    Returns:
        List of DocumentChunk objects ready for embedding.
    """
    logger.info("Starting policy document ingestion...")
    documents = load_documents_from_folder(
        folder_path=POLICY_DOCS_PATH,
        document_type=DocumentType.POLICY,
    )
    chunks = chunk_documents(documents)
    logger.info(
        f"Policy ingestion complete: "
        f"{len(documents)} docs, {len(chunks)} chunks"
    )
    return chunks


def ingest_clinical_note(note_text: str, note_id: str) -> list[DocumentChunk]:
    """
    Chunk a single clinical note provided as a string.
    Used at review time — notes are not pre-indexed.

    Args:
        note_text: Raw clinical note text.
        note_id: Identifier for this note (e.g. case_id).

    Returns:
        List of DocumentChunk objects.
    """
    from app.ingestion.document_loader import LoadedDocument
    doc = LoadedDocument(
        content=note_text,
        source_file=f"clinical_note_{note_id}",
        document_type=DocumentType.CLINICAL_NOTE,
        file_extension=".txt",
    )
    chunks = chunk_documents([doc])
    logger.info(
        f"Clinical note {note_id} chunked into {len(chunks)} chunks"
    )
    return chunks
