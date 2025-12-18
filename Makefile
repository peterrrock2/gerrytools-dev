# Makefile for managing GerryTools development tasks using 'uv' virtual environment manager.

PYTHON_VERSION = 3.11
VENV_DIR ?= .venv
PKG ?= gerrytools
TEST_PATHS ?= tests

.PHONY: help setup install dev install-docs test type-check lint format precommit docs clean snapshots


help:
	@echo "Available targets:"
	@echo "  setup         - Set up the development environment"
	@echo "  install       - Install the package"
	@echo "  dev           - Install the package with development dependencies"
	@echo "  install-docs  - Install documentation dependencies"
	@echo "  test          - Run the test suite"
	@echo "  snapshots     - Update test snapshots"
	@echo "  type-check    - Run type checking with mypy"
	@echo "  lint          - Run code linters"
	@echo "  format        - Format the codebase"
	@echo "  precommit     - Run pre-commit hooks"
	@echo "  docs          - Build the documentation"
	@echo "  clean         - Clean build artifacts"


check_prereqs:
	@echo "Checking prerequisites..."
	@if ! command -v uv > /dev/null 2>&1; then \
		echo "Error: 'uv' is not installed. Please install it first using the following command:"; \
		echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		exit 1; \
	fi
	@echo "'uv' is installed."

setup: check_prereqs
	@echo "Setting up the environment for GerryTools..."
	@echo
	uv python install $(PYTHON_VERSION)
	@echo "Creating virtual environment and installing dev dependencies..."
	uv sync --python $(PYTHON_VERSION)
	uv sync --all-groups
	uv pip install -e ".[mgrp]"
	uv run pre-commit install
	@echo ""
	@echo "Development environment setup complete!"

install: check_prereqs
	@echo "Installing GerryTools package..."
	uv sync --python $(PYTHON_VERSION)
	uv pip install -e .

install-docs: check_prereqs
	@echo "Installing GerryTools package with all just the documentation dependencies..."
	uv sync --group docs --python $(PYTHON_VERSION)
	uv pip install -e ".[mgrp]"


snapshots:
	@echo "Updating test snapshots..."
	UPDATE_SNAPSHOTS=1 uv run pytest -m latex -v tests	

check:
	$(MAKE) format
	$(MAKE) lint

test:
	@echo "Running test suite..."
	PYTHONHASHSEED=0 uv run pytest -v $(TEST_PATHS)

# Add this in later
# type-check:
# 	@echo "Running type checking with mypy..."
# 	uv run mypy $(PKG) ${TEST_PATHS}

format:
	@echo "Formatting codebase with black..."
	uv run isort $(PKG) $(TEST_PATHS)
	uv run black $(PKG) $(TEST_PATHS)

lint: 
	@echo "Running linters (ruff)..."
	uv run ruff check $(PKG) $(TEST_PATHS)

precommit:
	@echo "Running pre-commit hooks..."
	uv run pre-commit install
	uv run pre-commit run --all-files

docs: install-docs
	@echo "Building documentation..."
	uv run sphinx-build -b html docs/source docs/build

clean:
	@echo "Cleaning build artifacts..."
	@rm -rf build/ \
		dist/ \
		*.egg-info \
		.pytest_cache/ \
		.mypy_cache/ \
		.ruff_cache/ \
		docs/_build/ \
		$(VENV_DIR) \
		.vscode/ \
		.ipynb_checkpoints/ \
		docs/build/
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "Clean complete."
