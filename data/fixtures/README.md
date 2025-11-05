# Dataset-validation fixtures

These JSONL files contain generated support requests, not customer conversations. Dates identify fixture variants; they are not collection dates or training-run timestamps.

Each complete fixture has 240 examples, six labels, and eight template families per label. Template-family IDs keep related wording in one split. Dataset preparation removes normalized-text duplicates and rejects contradictory labels or groups shared across labels.

`index.json` records source-file SHA-256 values and prepared-dataset fingerprints. Run `uv run python scripts/verify_fixtures.py` to check the entire catalog. A fixture can be imported from the Datasets page. The calendar reconstruction uses incremental fixture additions; it does not invent training metrics or experiment results.
