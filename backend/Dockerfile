# Build context is the repository root, not backend/ -- several backend
# services (classification_service.py, stt_service.py)
# resolve `workspace_root = Path(__file__).resolve().parent.parent.parent.parent`
# at import time to reach models_directory/ and project_paths.py, which live
# as siblings of backend/ at the repo root. The image must preserve that same
# relative layout or those imports break.

FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# System dependencies + Microsoft ODBC Driver 18 for SQL Server (db_settings.json
# is configured for "ODBC Driver 18 for SQL Server"). libgomp1 is required at
# runtime by xgboost/torch (OpenMP), not present by default on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        gnupg2 \
        ca-certificates \
        apt-transport-https \
        unixodbc \
        unixodbc-dev \
        build-essential \
        libgomp1 \
    && curl -sSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg \
    && curl -sSL https://packages.microsoft.com/config/debian/12/prod.list -o /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies before copying source, for layer caching.
# torch is installed separately from the CPU-only PyTorch index first --
# this backend does CPU-side classification/embeddings inference, not the
# GPU STT workload, so the default (CUDA-bundled) PyPI wheel would be a
# multi-GB waste. Once installed, requirements.txt's matching torch==
# pin is already satisfied and pip will not reinstall it.
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.9.0 \
    && pip install --no-cache-dir -r backend/requirements.txt

# Application source, preserving repo-root-relative layout.
COPY project_paths.py project_paths.py
COPY models_directory/ models_directory/
COPY backend/ backend/

RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app \
    # Pre-create cache dirs with correct ownership. Named volumes mounted at
    # a path that doesn't yet exist in the image are created root-owned by
    # Docker on first mount; pre-creating them here (with appuser ownership)
    # means Docker copies this ownership into the volume instead. Without
    # this, huggingface_hub/matplotlib get silent Permission Denied errors
    # under the non-root appuser and STT model loading fails open (app still
    # reports healthy, but transcription is broken).
    && mkdir -p /home/appuser/.cache/huggingface /home/appuser/.cache/matplotlib \
    && chown -R appuser:appuser /home/appuser/.cache

WORKDIR /app/backend
USER appuser

EXPOSE 8100

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=5 \
    CMD curl -f http://localhost:8100/api/status || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8100"]
