from enum import Enum


class Decision(str, Enum):
    """
    The three possible prior authorization decisions.
    Every workflow run must return one of these.
    """
    APPROVE = "APPROVE"
    DENY = "DENY"
    NEED_MORE_INFO = "NEED_MORE_INFO"


class DocumentType(str, Enum):
    """Types of documents the system can ingest."""
    POLICY = "policy"
    CLINICAL_NOTE = "clinical_note"


# RAG settings
CHUNK_SIZE = 512          # tokens per chunk
CHUNK_OVERLAP = 50        # token overlap between chunks
TOP_K_RESULTS = 5         # number of chunks to retrieve

# LLM settings
MAX_RETRIES = 3           # retry attempts for API calls
RETRY_WAIT_SECONDS = 2    # base wait time between retries

# Confidence thresholds
HIGH_CONFIDENCE = 0.85
LOW_CONFIDENCE = 0.50
