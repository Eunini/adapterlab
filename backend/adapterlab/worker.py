import argparse
import logging
import signal
import threading
from datetime import timedelta
from uuid import uuid4

from sqlalchemy import update

from .config import Settings
from .queue import ACTIVE, TrainingCancelled, claim, finish, progress
from .schemas import TrainConfig
from .storage import Dataset, Run, Worker, setup_database, utcnow

log = logging.getLogger("adapterlab.worker")


def work_once(settings, sessions, stop=None, worker_id=None):
    stop = stop or threading.Event()
    worker_id = worker_id or str(uuid4())
    with sessions() as session:
        worker = session.get(Worker, worker_id)
        if worker is None:
            worker = Worker(id=worker_id)
            session.add(worker)
        worker.last_seen, worker.status = utcnow(), "idle"
        session.commit()
        run_id = claim(session, worker_id)
        if run_id is None:
            return False
        worker.status = "working"
        session.commit()
        run = session.get(Run, run_id)
        dataset = session.get(Dataset, run.dataset_id)
        records, report, config = (
            dataset.records,
            {**dataset.report, "synthetic": dataset.synthetic},
            TrainConfig(**run.config),
        )
    heartbeat_stop, lease_lost = threading.Event(), threading.Event()

    def heartbeat():
        while not heartbeat_stop.wait(15):
            try:
                with sessions() as session:
                    count = session.execute(
                        update(Run)
                        .where(Run.id == run_id, Run.worker_id == worker_id, Run.status.in_(ACTIVE))
                        .values(lease_until=utcnow() + timedelta(seconds=90))
                    ).rowcount
                    session.commit()
                    session.execute(
                        update(Worker).where(Worker.id == worker_id).values(last_seen=utcnow())
                    )
                    session.commit()
                    if count != 1:
                        lease_lost.set()
            except Exception:
                log.warning("Heartbeat failed for run %s", run_id)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()

    def callback(**values):
        if stop.is_set() or lease_lost.is_set():
            raise TrainingCancelled()
        progress(sessions, run_id, worker_id, **values)

    try:
        from .training import train

        result = train(records, report, config, settings, settings.artifacts_dir / run_id, callback)
        callback()
        finish(sessions, run_id, worker_id, "succeeded", result=result)
        log.info("Run %s completed", run_id)
    except TrainingCancelled:
        finish(sessions, run_id, worker_id, "cancelled")
        log.info("Run %s cancelled", run_id)
    except Exception as exception:
        # Dataset bodies, environment values, and model download credentials are not logged.
        message = f"Training failed ({type(exception).__name__}). Check model cache, available memory, and configuration."
        finish(sessions, run_id, worker_id, "failed", error=message)
        log.error("Run %s: %s", run_id, message)
    finally:
        heartbeat_stop.set()
        thread.join(timeout=2)
        with sessions() as session:
            session.execute(
                update(Worker)
                .where(Worker.id == worker_id)
                .values(last_seen=utcnow(), status="idle")
            )
            session.commit()
    return True


def main():
    parser = argparse.ArgumentParser(description="Run AdapterLab's durable training queue")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    settings = Settings()
    settings.validate()
    engine, sessions = setup_database(settings.database_url)
    stop = threading.Event()
    worker_id = str(uuid4())
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    try:
        while not stop.is_set():
            worked = work_once(settings, sessions, stop, worker_id)
            if args.once:
                break
            if not worked:
                stop.wait(2)
    finally:
        with sessions() as session:
            session.execute(update(Worker).where(Worker.id == worker_id).values(status="stopped"))
            session.commit()
        engine.dispose()


if __name__ == "__main__":
    main()
