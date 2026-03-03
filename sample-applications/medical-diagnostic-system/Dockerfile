# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ─── Stage 1: Build dependencies ──────────────────────────────────────────────
FROM python:3.11-slim AS python-base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    build-essential curl libffi-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip setuptools poetry

WORKDIR /app
COPY pyproject.toml README.md ./

RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-root && \
    rm -rf ~/.cache

# ─── Stage 2: Runtime image ───────────────────────────────────────────────────
FROM python:3.11-slim

ENV HOME="/home/appuser"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN groupadd -g 1001 appuser && \
    useradd -m -s /bin/bash -u 1001 -g 1001 appuser && \
    mkdir -p $HOME/.cache && \
    chown -R appuser:appuser $HOME

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    libgomp1 libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy installed packages
COPY --from=python-base /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=python-base /usr/local/bin /usr/local/bin

WORKDIR /app

# Copy application source
COPY app/ ./app/
COPY ui/ ./ui/

RUN mkdir -p uploads && chown -R appuser:appuser /app

USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8080/v1/meddiag/health || exit 1

CMD ["python", "-m", "uvicorn", "app.server:app", \
     "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
