from uuid import uuid4

from adapterlab.api import register_dataset
from adapterlab.data import demo_dataset
from adapterlab.schemas import TrainConfig
from adapterlab.storage import Run, setup_database
from adapterlab.worker import work_once


def create_job(sessions):
    with sessions() as session:
        dataset = register_dataset(session, demo_dataset())
        row = Run(
            id=str(uuid4()),
            name="Worker test",
            dataset_id=dataset.id,
            config=TrainConfig().model_dump(),
        )
        session.add(row)
        session.commit()
        return row.id


def test_worker_failure_does_not_publish_exception_contents(settings, monkeypatch):
    from adapterlab import training

    engine, sessions = setup_database(settings.database_url)
    run_id = create_job(sessions)

    def fail(*args, **kwargs):
        raise RuntimeError("private training sample")

    monkeypatch.setattr(training, "train", fail)
    assert work_once(settings, sessions)
    with sessions() as session:
        run = session.get(Run, run_id)
        assert run.status == "failed"
        assert "private training sample" not in run.error
        assert run.result is None and run.worker_id is None
    engine.dispose()


def test_worker_observes_cancellation_before_publishing_results(settings, monkeypatch):
    from adapterlab import training

    engine, sessions = setup_database(settings.database_url)
    run_id = create_job(sessions)

    def cancelled_train(records, report, config, settings, output_dir, callback):
        with sessions() as session:
            run = session.get(Run, run_id)
            run.cancel_requested = True
            session.commit()
        callback(step=1)

    monkeypatch.setattr(training, "train", cancelled_train)
    assert work_once(settings, sessions)
    with sessions() as session:
        assert session.get(Run, run_id).status == "cancelled"
    engine.dispose()
