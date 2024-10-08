import argparse
import json
import time
from uuid import uuid4

from .api import register_dataset
from .config import Settings
from .data import demo_dataset
from .schemas import TrainConfig
from .storage import Run, Worker, setup_database
from .worker import work_once


def main():
    parser = argparse.ArgumentParser(
        description="Train one run on the bundled support-intent dataset"
    )
    parser.add_argument("--strategy", choices=["lora", "head-only", "full"], default="lora")
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()
    settings = Settings()
    engine, sessions = setup_database(settings.database_url)
    run_id = str(uuid4())
    worker_id = str(uuid4())
    with sessions() as session:
        dataset = register_dataset(session, demo_dataset())
        session.add(
            Run(
                id=run_id,
                name=f"{args.strategy} support intents",
                dataset_id=dataset.id,
                config=TrainConfig(strategy=args.strategy, epochs=args.epochs).model_dump(),
            )
        )
        session.commit()
    # Process older queued work first; stop after this requested run becomes terminal.
    while True:
        with sessions() as session:
            run = session.get(Run, run_id)
            if run.status in ["succeeded", "failed", "cancelled"]:
                print(
                    json.dumps(
                        {
                            "id": run_id,
                            "status": run.status,
                            "result": run.result,
                            "error": run.error,
                        },
                        indent=2,
                    )
                )
                code = 0 if run.status == "succeeded" else 1
                break
        if not work_once(settings, sessions, worker_id=worker_id):
            time.sleep(0.5)
    with sessions() as session:
        worker = session.get(Worker, worker_id)
        if worker:
            worker.status = "stopped"
            session.commit()
    engine.dispose()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
