import hmac
import os
import threading
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select

from .config import MODEL_ID, MODEL_REVISION, Settings
from .data import demo_dataset, prepare_dataset
from .limits import BodyLimit
from .queue import ACTIVE, TERMINAL
from .schemas import DatasetInput, PredictionInput, RunInput
from .storage import (
    Dataset,
    Metric,
    Run,
    Worker,
    dataset_summary,
    run_summary,
    setup_database,
    utcnow,
)


def register_dataset(session, request):
    records, report = prepare_dataset(request)
    existing = session.get(Dataset, report["fingerprint"])
    if existing:
        return existing
    dataset = Dataset(
        id=report["fingerprint"],
        name=request.name,
        description=request.description,
        license=request.license,
        synthetic=request.synthetic,
        records=records,
        report=report,
    )
    session.add(dataset)
    session.commit()
    return dataset


def create_app(settings=None):
    settings = settings or Settings()
    settings.validate()

    @asynccontextmanager
    async def lifespan(app):
        engine, sessions = setup_database(settings.database_url)
        app.state.engine, app.state.sessions = engine, sessions
        settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
        if settings.seed_demo:
            with sessions() as session:
                register_dataset(session, demo_dataset())
        yield
        engine.dispose()

    app = FastAPI(title="AdapterLab", version="0.1.0", lifespan=lifespan)
    app.add_middleware(BodyLimit)
    app.state.settings = settings
    cache, inference_lock = OrderedDict(), threading.Lock()

    def session_dependency():
        with app.state.sessions() as session:
            yield session

    def operator(authorization: str = Header(default="")):
        if settings.protected and not hmac.compare_digest(
            authorization.encode(), f"Bearer {settings.operator_token}".encode()
        ):
            raise HTTPException(401, "Operator access required")

    router = APIRouter(prefix="/api", dependencies=[Depends(operator)])

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request, exception):
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    {key: error[key] for key in ("loc", "msg", "type")}
                    for error in exception.errors()
                ]
            },
        )

    @app.get("/api/health")
    def health():
        return {"status": "ok", "protected": settings.protected}

    def get_run(run_id, session):
        run = session.get(Run, run_id)
        if run is None:
            raise HTTPException(404, "Run not found")
        return run

    @router.get("/system")
    def system(session=Depends(session_dependency)):
        workers = session.scalar(
            select(func.count())
            .select_from(Worker)
            .where(Worker.last_seen > utcnow() - timedelta(seconds=45), Worker.status != "stopped")
        )
        return {
            "workers_online": workers,
            "queued": session.scalar(
                select(func.count()).select_from(Run).where(Run.status == "queued")
            ),
            "active": session.scalar(
                select(func.count()).select_from(Run).where(Run.status.in_(ACTIVE))
            ),
            "device": "cpu",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    @router.get("/datasets")
    def datasets(session=Depends(session_dependency)):
        return [
            dataset_summary(row)
            for row in session.scalars(select(Dataset).order_by(Dataset.created_at.desc()))
        ]

    @router.post("/datasets", status_code=201)
    def add_dataset(request: DatasetInput, session=Depends(session_dependency)):
        try:
            return dataset_summary(register_dataset(session, request))
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @router.get("/datasets/{dataset_id}")
    def dataset(dataset_id: str, session=Depends(session_dependency)):
        row = session.get(Dataset, dataset_id)
        if row is None:
            raise HTTPException(404, "Dataset not found")
        return {**dataset_summary(row), "preview": row.records[:12]}

    @router.get("/runs")
    def runs(session=Depends(session_dependency)):
        rows = session.execute(
            select(Run, Dataset.name, Dataset.synthetic)
            .join(Dataset)
            .order_by(Run.created_at.desc())
            .limit(100)
        )
        return [
            {**run_summary(run), "dataset_name": name, "synthetic": synthetic}
            for run, name, synthetic in rows
        ]

    @router.post("/runs", status_code=201)
    def add_run(request: RunInput, session=Depends(session_dependency)):
        if session.get(Dataset, request.dataset_id) is None:
            raise HTTPException(404, "Dataset not found")
        queued = session.scalar(
            select(func.count()).select_from(Run).where(Run.status.in_(["queued", *ACTIVE]))
        )
        if queued >= 20:
            raise HTTPException(429, "The queue already contains 20 pending runs")
        row = Run(
            id=str(uuid4()),
            name=request.name,
            dataset_id=request.dataset_id,
            config=request.config.model_dump(),
        )
        session.add(row)
        session.commit()
        return run_summary(row)

    @router.get("/runs/{run_id}")
    def run(run_id: str, session=Depends(session_dependency)):
        row = get_run(run_id, session)
        dataset = session.get(Dataset, row.dataset_id)
        points = list(
            session.scalars(
                select(Metric).where(Metric.run_id == run_id).order_by(Metric.id.desc()).limit(1000)
            )
        )
        return {
            **run_summary(row),
            "dataset_name": dataset.name,
            "synthetic": dataset.synthetic,
            "dataset_report": dataset.report,
            "metrics": [
                {"id": point.id, "step": point.step, "epoch": point.epoch, **point.values}
                for point in reversed(points)
            ],
        }

    @router.post("/runs/{run_id}/cancel")
    def cancel(run_id: str, session=Depends(session_dependency)):
        row = get_run(run_id, session)
        if row.status not in TERMINAL:
            row.cancel_requested = True
            if row.status == "queued":
                row.status, row.completed_at = "cancelled", utcnow()
            else:
                row.status = "cancelling"
            session.commit()
        return run_summary(row)

    @router.get("/runs/{run_id}/export")
    def export(run_id: str, session=Depends(session_dependency)):
        if get_run(run_id, session).status != "succeeded":
            raise HTTPException(409, "Only completed runs can be exported")
        path = settings.artifacts_dir / run_id / "model.zip"
        if not path.is_file():
            raise HTTPException(404, "Model files are missing from the artifact store")
        return FileResponse(
            path, filename=f"adapterlab-{run_id[:8]}.zip", media_type="application/zip"
        )

    @router.post("/runs/{run_id}/predict")
    def inference(run_id: str, request: PredictionInput, session=Depends(session_dependency)):
        if get_run(run_id, session).status != "succeeded":
            raise HTTPException(409, "Inference requires a completed run")
        if not inference_lock.acquire(blocking=False):
            raise HTTPException(429, "Inference is busy; retry shortly")
        try:
            from .training import load_artifact, predict

            if run_id not in cache:
                cache[run_id] = load_artifact(settings.artifacts_dir / run_id, settings)
                if len(cache) > 2:
                    cache.popitem(last=False)
            cache.move_to_end(run_id)
            return predict(
                settings.artifacts_dir / run_id, settings, request.text, loaded=cache[run_id]
            )
        except FileNotFoundError as error:
            raise HTTPException(404, "Model files are missing") from error
        finally:
            inference_lock.release()

    app.include_router(router)
    dist = Path(
        os.getenv("FRONTEND_DIST", str(Path(__file__).resolve().parents[2] / "frontend/dist"))
    )
    if dist.is_dir():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/")
        def index():
            return FileResponse(dist / "index.html", headers={"Cache-Control": "no-cache"})

        @app.get("/favicon.svg")
        def favicon():
            return FileResponse(dist / "favicon.svg")

    return app


app = create_app()
