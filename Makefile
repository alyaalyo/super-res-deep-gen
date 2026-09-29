# Full pipeline. Each step is resumable; see README for timings.
PY ?= python

.PHONY: setup data dev tune test seeds runtime analyze all check

setup:
	pip install -e ".[dev]"
	bash scripts/setup_third_party.sh

data:
	$(PY) scripts/download.py
	$(PY) scripts/prepare_data.py

check:
	$(PY) -m pytest -q
	$(PY) scripts/validate_protocol.py

# dev split: run everything, choose settings, freeze configs/frozen.yaml
dev:
	$(PY) scripts/run_sr.py --split dev
	$(PY) scripts/tune_dev.py
	$(PY) scripts/evaluate.py --split dev
	$(PY) scripts/analyze.py --split dev

# test split: run once with the frozen settings
test:
	$(PY) scripts/run_sr.py --split test
	$(PY) scripts/evaluate.py --split test

seeds:
	$(PY) scripts/seed_experiment.py

runtime:
	$(PY) scripts/benchmark_runtime.py

analyze:
	$(PY) scripts/analyze.py --split test

all: data check dev test seeds runtime analyze
