from datetime import timedelta

from sqlalchemy import select, update

from .storage import Metric, Run, utcnow

ACTIVE = ("preparing", "training", "evaluating", "cancelling")
TERMINAL = ("succeeded", "failed", "cancelled")


class TrainingCancelled(Exception):
    pass


def recover_expired(session):
    count = session.execute(
        update(Run)
        .where(Run.status.in_(ACTIVE), Run.lease_until < utcnow())
        .values(
            status="failed",
            completed_at=utcnow(),
            error="Worker lease expired. Create a new run to retry.",
            lease_until=None,
            worker_id=None,
        )
    ).rowcount
    session.commit()
    return count


def claim(session, worker_id):
    recover_expired(session)
    candidate = session.scalar(
        select(Run.id).where(Run.status == "queued").order_by(Run.created_at, Run.id).limit(1)
    )
    if candidate is None:
        return None
    claimed = session.execute(
        update(Run)
        .where(Run.id == candidate, Run.status == "queued")
        .values(
            status="preparing",
            started_at=utcnow(),
            worker_id=worker_id,
            lease_until=utcnow() + timedelta(seconds=90),
        )
        .returning(Run.id)
    ).scalar_one_or_none()
    session.commit()
    return claimed


def progress(
    sessions,
    run_id,
    worker_id,
    *,
    status=None,
    step=None,
    total_steps=None,
    epoch=None,
    values=None,
):
    with sessions() as session:
        run = session.get(Run, run_id)
        if run.worker_id != worker_id or run.status not in ACTIVE or run.cancel_requested:
            raise TrainingCancelled()
        if status:
            run.status = status
        for key, value in {"step": step, "total_steps": total_steps, "epoch": epoch}.items():
            if value is not None:
                setattr(run, key, value)
        run.lease_until = utcnow() + timedelta(seconds=90)
        if values is not None:
            session.add(Metric(run_id=run_id, step=run.step, epoch=run.epoch, values=values))
        session.commit()


def finish(sessions, run_id, worker_id, status, result=None, error=None):
    with sessions() as session:
        run = session.get(Run, run_id)
        if run.worker_id != worker_id or run.status not in ACTIVE:
            return False
        run.status = "cancelled" if run.cancel_requested else status
        run.result = result if run.status == "succeeded" else None
        run.error = error
        run.completed_at = utcnow()
        run.worker_id = None
        run.lease_until = None
        session.commit()
        return True
