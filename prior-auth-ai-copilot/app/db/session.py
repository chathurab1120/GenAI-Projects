from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from pathlib import Path
from app.core.config import get_settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


def get_engine():
    """
    Create SQLAlchemy engine pointing to the SQLite audit database.
    Creates the data/ directory if it does not exist yet.
    """
    settings = get_settings()
    db_path = Path(settings.sqlite_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        echo=False,
    )
    logger.info(f"Database engine created at {db_path}")
    return engine


engine = get_engine()
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


@contextmanager
def get_db_session() -> Session:
    """
    Context manager for database sessions.
    Always use this pattern:
        with get_db_session() as session:
            session.add(record)
            session.commit()
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
