from datetime import datetime
from sqlalchemy import Column, String, Float, DateTime, Integer, Text
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class AuditLog(Base):
    """
    Stores every prior authorization review request and result.
    Uses synthetic identifiers only — no real PHI stored.
    """
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(String(64), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Request fields (synthetic data only)
    member_id = Column(String(64), nullable=True)
    patient_age = Column(Integer, nullable=True)
    diagnosis = Column(String(256), nullable=True)
    requested_service = Column(String(256), nullable=True)
    provider_specialty = Column(String(128), nullable=True)

    # Decision output
    decision = Column(String(32), nullable=False)
    confidence = Column(Float, nullable=True)
    rationale = Column(Text, nullable=True)
    missing_information = Column(Text, nullable=True)

    # Performance tracking
    latency_seconds = Column(Float, nullable=True)
    llm_model = Column(String(64), nullable=True)
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<AuditLog case_id={self.case_id} "
            f"decision={self.decision} "
            f"confidence={self.confidence}>"
        )
