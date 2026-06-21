# Confidently Wrong - task runner. Everything goes through `uv`.

MODEL    ?= qwen3_4b_instruct
DATASET  ?= afrimmlu
HARDWARE ?= a100
UQ       ?= selective_abstention

.PHONY: help setup test lint smoke run analyze compare gce-submit download download-all clean

help:
	@echo "make setup                         # uv sync (creates .venv + uv.lock)"
	@echo "make test                          # unit tests (metrics on synthetic data)"
	@echo "make lint                          # ruff check"
	@echo "make smoke                         # execute notebooks/smoke_test.ipynb"
	@echo "make run     MODEL= DATASET= HARDWARE=   # local inference + metrics"
	@echo "make analyze MODEL= DATASET=       # per-experiment figures/tables"
	@echo "make compare                       # cross-model aggregation"
	@echo "make gce-submit MODEL= DATASET= HARDWARE=   # launch a GCE run (GCE_ZONE=... overrides .env)"
	@echo "make download EXP=<model>_<dataset>        # pull one experiment from GCS"
	@echo "make download-all                  # pull ALL experiments from GCS (for the combiner)"
	@echo "make clean                         # remove local artifacts/caches"

setup:
	uv sync --extra notebook --extra dev

test:
	uv run pytest -q

lint:
	uv run ruff check src tests

smoke:
	uv run jupyter nbconvert --to notebook --execute --inplace notebooks/smoke_test.ipynb

run:
	uv run python -m src.run model=$(MODEL) dataset=$(DATASET) hardware=$(HARDWARE) uq=$(UQ)

analyze:
	uv run python -m src.analyze --experiment $(MODEL)_$(DATASET)

compare:
	uv run python -m src.compare

gce-submit:
	./scripts/gce_submit.sh $(MODEL) $(HARDWARE) $(DATASET)

download:
	./scripts/download_artifacts.sh $(EXP)

download-all:
	./scripts/download_all.sh

clean:
	rm -rf artifacts/* wandb/ .pytest_cache **/__pycache__
