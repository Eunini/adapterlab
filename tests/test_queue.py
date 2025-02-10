from datetime import timedelta
from uuid import uuid4

import pytest
from adapterlab.api import register_dataset
from adapterlab.data import demo_dataset
from adapterlab.queue import TrainingCancelled, claim, finish, progress, recover_expired
from adapterlab.schemas import TrainConfig
from adapterlab.storage import Run, setup_database, utcnow


def job(sessions):
    with sessions() as session:
        dataset = register_dataset(session, demo_dataset())
        row = Run(
            id=str(uuid4()),
            name="Queue test",
            dataset_id=dataset.id,
            config=TrainConfig().model_dump(),
        )
        session.add(row)
        session.commit()
        return row.id


def test_only_one_worker_can_claim_a_job(settings):
    engine, sessions = setup_database(settings.database_url)
    run_id = job(sessions)
    with sessions() as session:
        assert claim(session, "worker-one") == run_id
    with sessions() as session:
        assert claim(session, "worker-two") is None
    with pytest.raises(TrainingCancelled):
        progress(sessions, run_id, "worker-two", step=1)
    engine.dispose()


def test_expired_lease_is_recovered_without_restarting_training(settings):
    engine, sessions = setup_database(settings.database_url)
    run_id = job(sessions)
    with sessions() as session:
        claim(session, "old-worker")
        row = session.get(Run, run_id)
        row.lease_until = utcnow() - timedelta(seconds=1)
        session.commit()
        assert recover_expired(session) == 1
        session.expire_all()
        assert session.get(Run, run_id).status == "failed"
    assert finish(sessions, run_id, "old-worker", "succeeded", result={}) is False
    engine.dispose()


def test_cancellation_prevents_progress_and_late_success(settings):
    engine, sessions = setup_database(settings.database_url)
    run_id = job(sessions)
    with sessions() as session:
        claim(session, "worker")
        row = session.get(Run, run_id)
        row.cancel_requested = True
        session.commit()
    with pytest.raises(TrainingCancelled):
        progress(sessions, run_id, "worker", step=4)
    assert finish(sessions, run_id, "worker", "succeeded", result={"fake": 1}) is True
    with sessions() as session:
        row = session.get(Run, run_id)
        assert row.status == "cancelled" and row.result is None
    engine.dispose()
