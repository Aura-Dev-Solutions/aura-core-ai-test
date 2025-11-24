from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy.orm import declarative_base, Mapped, mapped_column
from sqlalchemy import String, Text, JSON, DateTime

Base = declarative_base()

class DocumentORM(Base):
    """
      - Maps Python attributes to DB columns using SQLAlchemy.
    """
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    content_type: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, index=True)

    text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    structure: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    classification: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    entities: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    embeddings: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
