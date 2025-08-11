FROM node:24-bookworm-slim AS frontend
WORKDIR /build/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 FRONTEND_DIST=/app/frontend/dist HF_HOME=/app/data/runtime/hf-cache
WORKDIR /app
RUN useradd --create-home --uid 10001 adapterlab
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./
COPY backend/ backend/
RUN uv sync --frozen --no-dev --no-editable --extra ml
COPY --from=frontend /build/frontend/dist/ frontend/dist/
RUN mkdir -p data/runtime artifacts && chown -R adapterlab:adapterlab /app
USER adapterlab
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"]
CMD ["/app/.venv/bin/uvicorn", "adapterlab.api:app", "--host", "0.0.0.0", "--port", "8000"]
