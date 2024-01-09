"""Generate reproducible support-intent fixtures for dataset validation."""

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

from adapterlab.data import synthetic_examples


def examples(day):
    return synthetic_examples(namespace=f"case-{day:%Y%m%d}-x", variations=5)


def encode(rows):
    return "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    ).encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("data/generated"))
    args = parser.parse_args()
    if not 1 <= args.days <= 366:
        parser.error("days must be between 1 and 366")
    args.output.mkdir(parents=True, exist_ok=True)
    for index in range(args.days):
        day = args.start + timedelta(days=index)
        (args.output / f"{day.isoformat()}.jsonl").write_bytes(encode(examples(day)))
    print(f"Generated {args.days} synthetic dataset fixtures in {args.output}")


if __name__ == "__main__":
    main()
