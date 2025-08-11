"""Verify the live queue, a real CPU training run, model export, and inference."""

import argparse
import io
import json
import time
import zipfile

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8010")
    args = parser.parse_args()
    with httpx.Client(base_url=args.url, timeout=60) as client:
        dataset_response = client.get("/api/datasets")
        dataset_response.raise_for_status()
        dataset = dataset_response.json()[0]
        response = client.post(
            "/api/runs",
            json={
                "name": "integration-smoke",
                "dataset_id": dataset["id"],
                "config": {"strategy": "lora", "epochs": 1, "batch_size": 32, "max_length": 32},
            },
        )
        response.raise_for_status()
        run_id = response.json()["id"]
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            run = client.get(f"/api/runs/{run_id}").json()
            if run["status"] in ["succeeded", "failed", "cancelled"]:
                break
            time.sleep(1)
        else:
            raise RuntimeError("Training verification timed out")
        if run["status"] != "succeeded":
            raise RuntimeError(run["error"] or run["status"])
        bundle = client.get(f"/api/runs/{run_id}/export")
        bundle.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(bundle.content)) as archive:
            assert "checksums.json" in archive.namelist()
            assert "BASE_MODEL_LICENSE.txt" in archive.namelist()
        prediction = client.post(
            f"/api/runs/{run_id}/predict", json={"text": "Please help me sign in to my account."}
        )
        prediction.raise_for_status()
        assert prediction.json()["prediction"] in dataset["report"]["labels"]
        assert run["result"]["trainable_parameters"] < run["result"]["total_parameters"]
        print(
            json.dumps(
                {
                    "status": "pass",
                    "run_id": run_id,
                    "steps": run["result"]["steps"],
                    "trainable_parameters": run["result"]["trainable_parameters"],
                    "test_samples": run["result"]["test"]["samples"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
