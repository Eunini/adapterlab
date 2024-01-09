from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool


class Base(DeclarativeBase):
    pass


def utcnow():
    return datetime.now(UTC).replace(tzinfo=None)


class Dataset(Base):
    __tablename__ = "datasets"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(80))
    description: Mapped[str] = mapped_column(Text)
    license: Mapped[str] = mapped_column(String(80))
    synthetic: Mapped[bool] = mapped_column(Boolean)
    records: Mapped[list] = mapped_column(JSON)
    report: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Run(Base):
    __tablename__ = "runs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(80))
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"))
    config: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)
    lease_until: Mapped[datetime | None] = mapped_column(DateTime)
    worker_id: Mapped[str | None] = mapped_column(String(36))
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    step: Mapped[int] = mapped_column(Integer, default=0)
    total_steps: Mapped[int] = mapped_column(Integer, default=0)
    epoch: Mapped[float] = mapped_column(Float, default=0)
    result: Mapped[dict | None] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(Text)


class Metric(Base):
    __tablename__ = "metrics"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    step: Mapped[int] = mapped_column(Integer)
    epoch: Mapped[float] = mapped_column(Float)
    values: Mapped[dict] = mapped_column(JSON)
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Worker(Base):
    __tablename__ = "workers"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    last_seen: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    status: Mapped[str] = mapped_column(String(20), default="idle")


def setup_database(url):
    options = {}
    if url.startswith("sqlite"):
        options["connect_args"] = {"check_same_thread": False, "timeout": 20}
        if ":memory:" in url:
            options["poolclass"] = StaticPool
        else:
            Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, **options)
    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def sqlite_options(connection, _):
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA journal_mode=WAL")

    Base.metadata.create_all(engine)
    return engine, sessionmaker(engine, expire_on_commit=False)


def dataset_summary(dataset):
    return {
        "id": dataset.id,
        "name": dataset.name,
        "description": dataset.description,
        "license": dataset.license,
        "synthetic": dataset.synthetic,
        "report": dataset.report,
        "created_at": dataset.created_at.isoformat() + "Z",
    }


def run_summary(run):
    result = {
        column.name: getattr(run, column.name)
        for column in Run.__table__.columns
        if column.name not in ["worker_id", "lease_until"]
    }
    for key in ["created_at", "started_at", "completed_at"]:
        if result[key] is not None:
            result[key] = result[key].isoformat() + "Z"
    return result
