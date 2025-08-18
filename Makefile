# Makefile for Aura Document Analyzer
# Provides common development and deployment tasks

.PHONY: help install install-dev test test-unit test-integration test-e2e lint format type-check clean build run run-dev docker-build docker-run docker-stop setup-dev

# Default target
help:
	@echo "Available targets:"
	@echo "  install          Install production dependencies"
	@echo "  install-dev      Install development dependencies"
	@echo "  test             Run all tests"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-e2e         Run end-to-end tests only"
	@echo "  lint             Run linting checks"
	@echo "  format           Format code with black and isort"
	@echo "  type-check       Run type checking with mypy"
	@echo "  clean            Clean up temporary files"
	@echo "  build            Build the application"
	@echo "  run              Run the application"
	@echo "  run-dev          Run the application in development mode"
	@echo "  docker-build     Build Docker image"
	@echo "  docker-run       Run application with Docker Compose"
	@echo "  docker-stop      Stop Docker Compose services"
	@echo "  setup-dev        Setup development environment"

# Python environment
PYTHON := python3
PIP := pip
VENV := venv
VENV_BIN := $(VENV)/bin

# Install production dependencies
install:
	$(PIP) install -e .

# Install development dependencies
install-dev:
	$(PIP) install -e .[dev]
	python -m spacy download en_core_web_sm

# Setup development environment
setup-dev: install-dev
	cp .env.example .env
	mkdir -p data/{uploads,processed,models,vector_db} logs
	pre-commit install

# Testing
test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-unit:
	pytest tests/unit/ -v -m "not slow"

test-integration:
	pytest tests/integration/ -v

test-e2e:
	pytest tests/e2e/ -v -m "not slow"

test-fast:
	pytest tests/ -v -m "not slow" --maxfail=1

# Code quality
lint:
	flake8 src tests
	black --check src tests
	isort --check-only src tests

format:
	black src tests
	isort src tests

type-check:
	mypy src

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/

# Build
build:
	python -m build

# Run application
run:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000

run-dev:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Docker commands
docker-build:
	docker build -t aura-document-analyzer .

docker-build-dev:
	docker build -t aura-document-analyzer:dev --target development .

docker-run:
	docker-compose up -d

docker-run-full:
	docker-compose --profile monitoring up -d

docker-stop:
	docker-compose down

docker-logs:
	docker-compose logs -f app

docker-shell:
	docker-compose exec app bash

# Database commands
db-upgrade:
	alembic upgrade head

db-downgrade:
	alembic downgrade -1

db-migration:
	alembic revision --autogenerate -m "$(MESSAGE)"

db-reset:
	docker-compose down -v
	docker-compose up -d postgres redis
	sleep 5
	alembic upgrade head

# Development utilities
shell:
	python -c "from src.core.config import settings; print('Settings loaded'); import IPython; IPython.embed()"

download-models:
	python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2'); AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')"
	python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('distilbert-base-uncased'); AutoModel.from_pretrained('distilbert-base-uncased')"
	python -m spacy download en_core_web_sm

# Performance testing
benchmark:
	python scripts/benchmark.py

# Security scanning
security-check:
	bandit -r src/
	safety check

# Documentation
docs-build:
	mkdocs build

docs-serve:
	mkdocs serve

# CI/CD helpers
ci-install:
	pip install -e .[dev]
	python -m spacy download en_core_web_sm

ci-test:
	pytest tests/ --cov=src --cov-report=xml --cov-report=term

ci-quality:
	black --check src tests
	isort --check-only src tests
	flake8 src tests
	mypy src
