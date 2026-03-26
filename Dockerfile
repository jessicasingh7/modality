FROM python:3.12-slim AS base

WORKDIR /app

# Install dependencies first (cached layer)
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy application code
COPY modality/ modality/

# No CMD here — each service sets its own entrypoint

# ---------------------------------------------------------------------------
# Data Plane — customer-facing inference API
# ---------------------------------------------------------------------------
FROM base AS data-plane

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "modality.gateway.data_plane:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "4", "--loop", "uvloop", "--http", "httptools"]

# ---------------------------------------------------------------------------
# Control Plane — internal management API
# ---------------------------------------------------------------------------
FROM base AS control-plane

EXPOSE 8001

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')"

CMD ["uvicorn", "modality.gateway.control_plane:app", \
     "--host", "0.0.0.0", "--port", "8001", \
     "--workers", "2"]
