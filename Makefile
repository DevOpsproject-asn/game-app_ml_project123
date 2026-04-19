"""Makefile for common development tasks"""

.PHONY: help install dev test lint format clean run docker-up docker-down

help:
	@echo "Game ML Platform - Development Commands"
	@echo "========================================"
	@echo "make install       - Install dependencies"
	@echo "make dev           - Setup development environment"
	@echo "make test          - Run all tests"
	@echo "make test-cov      - Run tests with coverage"
	@echo "make lint          - Run linting checks"
	@echo "make format        - Format code with black and isort"
	@echo "make train         - Run training pipeline"
	@echo "make run           - Start FastAPI server"
	@echo "make quick-demo    - Run quick demo"
	@echo "make docker-build  - Build Docker image"
	@echo "make docker-up     - Start Docker Compose services"
	@echo "make docker-down   - Stop Docker Compose services"
	@echo "make clean         - Clean up cache and artifacts"

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 isort

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/index.html"

lint:
	flake8 src/ tests/ pipelines/ --max-line-length=100
	black --check src/ tests/ pipelines/

format:
	black src/ tests/ pipelines/
	isort src/ tests/ pipelines/

train:
	python pipelines/training_pipeline.py

run:
	uvicorn src.api.main:app --reload --port 8000

quick-demo:
	python quick_start.py

docker-build:
	docker build -t game-ml-platform:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f api

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
