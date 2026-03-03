# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

.PHONY: help install dev build up down logs test clean

IMAGE_NAME ?= meddiag-system:latest

help:
	@echo ""
	@echo "  Intelligent Multimodal Medical Diagnostic System"
	@echo "  ─────────────────────────────────────────────────"
	@echo "  make install   Install Python dependencies (dev)"
	@echo "  make dev       Run backend in development mode"
	@echo "  make build     Build Docker image"
	@echo "  make up        Start all services (docker-compose)"
	@echo "  make down      Stop all services"
	@echo "  make logs      Tail service logs"
	@echo "  make test      Run unit tests"
	@echo "  make clean     Remove containers, volumes and cache"
	@echo ""

install:
	python -m venv .venv && \
	. .venv/bin/activate && \
	pip install --upgrade pip && \
	pip install poetry && \
	poetry install

dev:
	. .venv/bin/activate && \
	DEMO_MODE=true PYTHONPATH=. uvicorn app.server:app \
	  --host 0.0.0.0 --port 8080 --reload

build:
	docker build -t $(IMAGE_NAME) .

up:
	cp -n .env.example .env 2>/dev/null || true
	docker compose up --build -d
	@echo ""
	@echo "  ✅ Services started. Open http://localhost:8200"
	@echo "  📖 API docs: http://localhost:8201/v1/meddiag/docs"
	@echo ""

down:
	docker compose down

logs:
	docker compose logs -f meddiag

test:
	. .venv/bin/activate && \
	PYTHONPATH=. pytest tests/ -v --tb=short

clean:
	docker compose down -v --remove-orphans
	find . -type d -name __pycache__ | xargs rm -rf
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache
