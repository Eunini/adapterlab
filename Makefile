.PHONY: setup build run train test check
setup:
	uv sync --locked --extra ml --extra dev
	npm --prefix frontend ci
build:
	npm --prefix frontend run build
run:
	uv run python scripts/dev.py
train:
	uv run adapterlab-train --strategy lora --epochs 4
test:
	uv run pytest -q
check:
	uv run ruff check backend tests scripts
	uv run ruff format --check backend tests scripts
	npm --prefix frontend run format:check
	npm --prefix frontend run build
