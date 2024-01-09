"""Check fixture hashes, schemas, and group-separated split fingerprints."""

import hashlib
import json
from pathlib import Path

from adapterlab.data import prepare_dataset
from adapterlab.schemas import DatasetInput


def main():
    base = Path(__file__).resolve().parents[1] / "data/fixtures"
    index = json.loads((base / "index.json").read_text())
    total = 0
    for entry in index["days"]:
        file = (base / entry["file"]).resolve()
        if not file.is_relative_to(base.resolve()):
            raise ValueError("Fixture path escapes the catalog")
        payload = file.read_bytes()
        if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise ValueError(f"Hash mismatch for {entry['file']}")
        rows = [json.loads(line) for line in payload.splitlines()]
        _, report = prepare_dataset(
            DatasetInput(name=entry["date"], license="CC0-1.0", synthetic=True, records=rows)
        )
        if (
            report["fingerprint"] != entry["dataset_fingerprint"]
            or report["examples"] != entry["examples"]
        ):
            raise ValueError("Dataset fingerprint or sample count mismatch")
        total += report["examples"]
    print(
        f"PASS: {len(index['days'])} fixtures, {total} valid synthetic examples, disjoint template-group splits"
    )


if __name__ == "__main__":
    main()
