import enum
import secrets
from datetime import datetime

from sqlalchemy import JSON, BigInteger, DateTime, Enum, Float, ForeignKey, Index, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Provider(str, enum.Enum):
    openai = "openai"
    fireworks = "fireworks"
    together = "together"


class JobStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


class ModelStatus(str, enum.Enum):
    training = "training"
    evaluating = "evaluating"
    active = "active"       # promoted into the routing table
    inactive = "inactive"   # demoted or replaced
    failed = "failed"


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True)
    domain: Mapped[str] = mapped_column(String(255))  # e.g. "legal", "finance", "support"
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    models: Mapped[list["FineTunedModel"]] = relationship(back_populates="customer")
    api_keys: Mapped[list["ApiKey"]] = relationship(back_populates="customer")


class FineTunedModel(Base):
    __tablename__ = "finetuned_models"

    id: Mapped[int] = mapped_column(primary_key=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"))
    name: Mapped[str] = mapped_column(String(255))
    provider: Mapped[Provider] = mapped_column(Enum(Provider))
    provider_model_id: Mapped[str | None] = mapped_column(String(512), nullable=True)
    base_model: Mapped[str] = mapped_column(String(255))  # e.g. "gpt-4o-mini", "llama-3.1-8b"
    status: Mapped[ModelStatus] = mapped_column(Enum(ModelStatus), default=ModelStatus.training)
    domain: Mapped[str] = mapped_column(String(255))  # routing domain

    # Routing metadata — embedding of what this model is good at
    domain_description: Mapped[str | None] = mapped_column(Text, nullable=True)
    domain_embedding: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # Eval scores
    eval_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Cost tracking
    cost_per_1k_input: Mapped[float | None] = mapped_column(Float, nullable=True)
    cost_per_1k_output: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    customer: Mapped["Customer"] = relationship(back_populates="models")
    jobs: Mapped[list["FineTuneJob"]] = relationship(back_populates="model")


class FineTuneJob(Base):
    __tablename__ = "finetune_jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("finetuned_models.id"))
    provider: Mapped[Provider] = mapped_column(Enum(Provider))
    provider_job_id: Mapped[str | None] = mapped_column(String(512), nullable=True)
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus), default=JobStatus.pending)
    base_model: Mapped[str] = mapped_column(String(255))
    training_file: Mapped[str | None] = mapped_column(String(512), nullable=True)
    hyperparameters: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    model: Mapped["FineTunedModel"] = relationship(back_populates="jobs")


class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(primary_key=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"))
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    key_prefix: Mapped[str] = mapped_column(String(12))  # "mod_abc123..." for display
    name: Mapped[str] = mapped_column(String(255))  # e.g. "production", "staging"
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    customer: Mapped["Customer"] = relationship(back_populates="api_keys")

    @staticmethod
    def generate() -> tuple[str, str]:
        """Generate a new API key. Returns (raw_key, key_hash)."""
        import hashlib
        raw = "mod_" + secrets.token_urlsafe(32)
        hashed = hashlib.sha256(raw.encode()).hexdigest()
        return raw, hashed


class UsageLog(Base):
    """Logs every inference request for billing and analytics."""
    __tablename__ = "usage_logs"
    __table_args__ = (
        Index("ix_usage_customer_created", "customer_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"))
    model_id: Mapped[int | None] = mapped_column(ForeignKey("finetuned_models.id"), nullable=True)
    provider: Mapped[str] = mapped_column(String(50))
    provider_model_id: Mapped[str] = mapped_column(String(512))
    is_fallback: Mapped[bool] = mapped_column(default=False)
    prompt_tokens: Mapped[int] = mapped_column(BigInteger, default=0)
    completion_tokens: Mapped[int] = mapped_column(BigInteger, default=0)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
