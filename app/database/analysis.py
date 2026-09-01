from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db_connection import Base


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    image_path: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    result: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    user = relationship(
        "User",
        back_populates="analyses",
    )