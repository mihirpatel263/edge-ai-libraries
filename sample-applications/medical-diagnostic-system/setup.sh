#!/usr/bin/env bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# setup.sh — Install dependencies and start the Medical Diagnostic System
#           in development mode (no Docker required).
#
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

PYTHON=${PYTHON:-python3}
VENV_DIR=".venv"

log() { echo -e "\033[1;34m[setup]\033[0m $*"; }
ok()  { echo -e "\033[1;32m[ ok ]\033[0m $*"; }
err() { echo -e "\033[1;31m[err ]\033[0m $*" >&2; }

# ─── Check Python ──────────────────────────────────────────────────────────────
log "Checking Python version…"
PY_VERSION=$($PYTHON --version 2>&1)
log "Found: $PY_VERSION"

# ─── Create virtual environment ───────────────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
  log "Creating virtual environment at $VENV_DIR…"
  $PYTHON -m venv "$VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
ok "Activated virtual environment"

# ─── Install dependencies ─────────────────────────────────────────────────────
log "Installing dependencies…"
pip install --quiet --upgrade pip
pip install --quiet poetry
poetry install --only main --no-root
ok "Dependencies installed"

# ─── Copy env file ────────────────────────────────────────────────────────────
if [[ ! -f ".env" ]]; then
  cp .env.example .env
  log "Created .env from .env.example (DEMO_MODE=true)"
fi

# ─── Create required directories ──────────────────────────────────────────────
mkdir -p uploads
ok "Upload directory ready"

# ─── Start server ─────────────────────────────────────────────────────────────
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8080}

log "Starting Medical Diagnostic System on http://${HOST}:${PORT}"
echo ""
echo "  ╔══════════════════════════════════════════════╗"
echo "  ║  Intelligent Multimodal Medical Diagnostic  ║"
echo "  ╠══════════════════════════════════════════════╣"
echo "  ║  UI:      http://${HOST}:${PORT}/            ║"
echo "  ║  API:     http://${HOST}:${PORT}/v1/meddiag  ║"
echo "  ║  Docs:    http://${HOST}:${PORT}/v1/meddiag/docs ║"
echo "  ╚══════════════════════════════════════════════╝"
echo ""

DEMO_MODE=true PYTHONPATH="$DIR" \
  uvicorn app.server:app --host "$HOST" --port "$PORT" --reload
