.PHONY: help setup hooks-install check validate validate-all fix typecheck test

PY_DIRS := docling_mcp tests examples .github/scripts

help: ## Show available targets.
	@awk 'BEGIN {FS = ":.*##"; print "Available targets:"} /^[a-zA-Z0-9_.-]+:.*##/ {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Install CI-style development environment.
	uv sync --frozen --all-extras

hooks-install: ## Install git hooks via prek.
	uv run prek install

check: ## Run read-only local checks.
	uv run ruff format --check --config=pyproject.toml $(PY_DIRS)
	uv run ruff check --config=pyproject.toml $(PY_DIRS)
	uv run --no-sync ty check
	uv lock --locked

validate: ## Run hooks on the current changeset.
	@files="$$( \
		{ \
			git diff --name-only --diff-filter=ACMR; \
			git diff --cached --name-only --diff-filter=ACMR; \
			git ls-files --others --exclude-standard; \
		} | sort -u \
	)"; \
	if [ -z "$$files" ]; then \
		echo "No changed files to validate."; \
	else \
		printf '%s\n' "$$files" | xargs uv run prek run --files; \
	fi

validate-all: ## Run hooks on all files.
	uv run prek run --all-files

fix: ## Run Ruff auto-format and fixers.
	uv run ruff format --config=pyproject.toml $(PY_DIRS)
	uv run ruff check --fix --config=pyproject.toml $(PY_DIRS)

typecheck: ## Run ty.
	uv run --no-sync ty check

test: ## Run the test suite.
	uv run pytest --cov=docling_mcp --cov-report=xml tests
