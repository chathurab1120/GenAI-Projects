from app.db.models import Base
from app.db.session import engine
from app.core.logging_config import get_logger

logger = get_logger(__name__)


def init_db() -> None:
    """
    Create all database tables if they do not already exist.
    Safe to call multiple times — will not overwrite existing data.
    """
    logger.info("Initialising database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ready.")


if __name__ == "__main__":
    init_db()
