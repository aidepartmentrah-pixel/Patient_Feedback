# Build context is the repository root (same reason as backend/Dockerfile):
# install_database.py does `sys.path.insert(0, str(BACKEND_DIR))` and imports
# core.deployment_port, so this image needs backend/core/ and
# backend/config/ alongside database/sqlserver/, in the same relative layout
# as the source repo.
#
# This is a one-shot init/migrate service: it runs the install pipeline,
# THEN provisions real organizational units and user accounts (Stage C of the
# org/user migration -- see database/sqlserver/seed/provision.py), against the
# sqlserver service, and exits. It does not need the backend's full ML
# dependency stack (no torch/transformers/etc here) -- just pyodbc + bcrypt.
#
# database/sqlserver/seed/*.local.json and provisioning.v1.* are NOT baked
# into this image (see .dockerignore) -- they are supplied at runtime via a
# read-only volume mount from the release bundle, same handling as the
# Whisper model asset.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# deb.debian.org's plain-HTTP edge (Fastly) intermittently 403s package
# index fetches; https to the same host is reliable. Harmless either way --
# apt trusts the Release file signature, not transport security -- but this
# avoids build-time flakiness.
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null; \
    sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list 2>/dev/null; \
    true

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        gnupg2 \
        ca-certificates \
        apt-transport-https \
        unixodbc \
        unixodbc-dev \
    && curl -sSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg \
    && curl -sSL https://packages.microsoft.com/config/debian/12/prod.list -o /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pyodbc==5.3.0 bcrypt==5.0.0

WORKDIR /app

COPY backend/__init__.py backend/__init__.py
COPY backend/core/ backend/core/
COPY backend/config/ backend/config/
COPY database/sqlserver/ database/sqlserver/

# Fail-hard sequence: schema/static-config install, then org-unit/user
# provisioning. Either step failing must stop the container with a non-zero
# exit so docker-compose's `service_completed_successfully` never triggers
# for a partial or failed install.
CMD ["sh", "-c", "python database/sqlserver/scripts/install_database.py && python database/sqlserver/seed/provision.py"]
