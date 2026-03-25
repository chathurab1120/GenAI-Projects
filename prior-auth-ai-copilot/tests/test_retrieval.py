import pytest
from unittest.mock import MagicMock, patch
from app.retrieval.vectorstore import RetrievedChunk
from app.ingestion.text_chunker import chunk_documents
from app.ingestion.document_loader import LoadedDocument
from app.core.constants import DocumentType, CHUNK_SIZE


def test_chunk_short_document():
    """Short documents produce a single chunk."""
    doc = LoadedDocument(
        content="Short clinical note.",
        source_file="test.txt",
        document_type=DocumentType.CLINICAL_NOTE,
        file_extension=".txt",
    )
    chunks = chunk_documents([doc])
    assert len(chunks) == 1
    assert chunks[0].content == "Short clinical note."
    assert chunks[0].chunk_index == 0
    assert chunks[0].total_chunks == 1


def test_chunk_long_document():
    """Long documents are split into multiple chunks."""
    long_text = "This is a sentence with some content. " * 50
    doc = LoadedDocument(
        content=long_text,
        source_file="long_policy.md",
        document_type=DocumentType.POLICY,
        file_extension=".md",
    )
    chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.source_file == "long_policy.md"
        assert chunk.document_type == DocumentType.POLICY


def test_chunk_ids_are_unique():
    """Every chunk gets a unique chunk_id."""
    long_text = "Word content here for testing. " * 100
    doc = LoadedDocument(
        content=long_text,
        source_file="policy.md",
        document_type=DocumentType.POLICY,
        file_extension=".md",
    )
    chunks = chunk_documents([doc], chunk_size=100, chunk_overlap=10)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_empty_list():
    """Empty document list returns empty chunk list."""
    chunks = chunk_documents([])
    assert chunks == []


def test_retrieved_chunk_fields():
    """RetrievedChunk dataclass holds all expected fields."""
    chunk = RetrievedChunk(
        chunk_id="policy.md::chunk_0000",
        content="Policy content here",
        source_file="policy.md",
        similarity_score=0.85,
        chunk_index=0,
    )
    assert chunk.chunk_id == "policy.md::chunk_0000"
    assert chunk.similarity_score == 0.85
