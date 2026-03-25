from pathlib import Path
from dataclasses import dataclass
from app.core.constants import DocumentType
from app.core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class LoadedDocument:
    """A single loaded document with its raw text and metadata."""
    content: str
    source_file: str
    document_type: DocumentType
    file_extension: str


def load_documents_from_folder(
    folder_path: str,
    document_type: DocumentType,
) -> list[LoadedDocument]:
    """
    Load all supported documents (.txt, .md, .pdf) from a folder.

    Args:
        folder_path: Path to the folder containing documents.
        document_type: Whether these are POLICY or CLINICAL_NOTE docs.

    Returns:
        List of LoadedDocument objects with content and metadata.

    Raises:
        FileNotFoundError: If the folder does not exist.
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    supported_extensions = {".txt", ".md", ".pdf"}
    documents: list[LoadedDocument] = []

    for file_path in sorted(folder.iterdir()):
        if file_path.suffix.lower() not in supported_extensions:
            continue
        if file_path.name == ".gitkeep":
            continue

        try:
            content = _read_file(file_path)
            if not content.strip():
                logger.warning(f"Skipping empty file: {file_path.name}")
                continue

            documents.append(LoadedDocument(
                content=content,
                source_file=file_path.name,
                document_type=document_type,
                file_extension=file_path.suffix.lower(),
            ))
            logger.info(f"Loaded: {file_path.name} ({len(content)} chars)")

        except Exception as e:
            logger.error(f"Failed to load {file_path.name}: {e}")
            continue

    logger.info(
        f"Loaded {len(documents)} documents from {folder_path}"
    )
    return documents


def _read_file(file_path: Path) -> str:
    """
    Read a single file and return its text content.
    Supports .txt, .md (plain text) and .pdf (via pypdf).
    """
    if file_path.suffix.lower() == ".pdf":
        return _read_pdf(file_path)
    return file_path.read_text(encoding="utf-8")


def _read_pdf(file_path: Path) -> str:
    """Extract text from a PDF file using pypdf."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(file_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF {file_path.name}: {e}")
