# Front door for the pilot. GNU Make 3.81 safe: no .ONESHELL, one command per recipe line, $$ escapes.
#   make env         create a python3.12 venv ($(VENV)) and install the hash lock binary-only
#   make lock        regenerate requirements.txt from pyproject.toml (universal, hashed, frozen date)
#   make lock-check  prove the lock resolves binary-only with hashes on Linux x86_64/aarch64 and mac arm64
#   make test        offline pytest in the ACTIVE interpreter (no venv management, no downloads)
#   make test-all    also the network tests: pinned dataset (data) and ~1.1 GB pinned weights (weights)
#   make baselines   score majority, label prior and zero-shot Laya on val and test, then rebuild the reports
#                    (needs `make data` and the pinned weights; about 5 minutes on the M4). It passes --force,
#                    because score.py refuses to overwrite an existing dump in results/dumps (the committed record)
#   make baseline-reports  rebuild the reports and print the results table from the committed dumps (no model)
#   make berturk-postfit   rebuild the Phase 6 BERTurk postfit record from the committed head-LR selection dumps (no model)
PYTHON ?= python
UV ?= uv
PY312 ?= python3.12
VENV ?= .venv
LOCK_DATE := 2026-10-13T00:00:00Z
LAYA_ZS := laya-zeroshot-7b928d82
BERTURK_SELECTION := berturk-091a3daacd74-1c8b6c80-29446b61 berturk-18579d8de257-0ec63c75-29446b61 berturk-3d05bf5d0a66-4f387ff8-29446b61 berturk-ddf9d83bad62-2b5a4009-29446b61 berturk-f47735f1c892-4b07782f-29446b61 berturk-736a4c29ca0e-8a7b46bb-29446b61 berturk-4b6d36a8a788-8580e471-29446b61 berturk-a159207c2727-2582cf0e-29446b61 berturk-37f76167f8e7-f4a040e2-29446b61 berturk-0cf7ae114982-60eedf95-29446b61
SCORE_WORKERS ?= 2
SCORE_CPUS ?= 4
REPORTS := results/reports

.PHONY: env lock lock-check data weights test test-all test-unit clean-data baselines baseline-reports berturk-postfit

env:
	$(PY312) -m venv $(VENV)
	$(VENV)/bin/python -m pip install --require-hashes --only-binary :all: -r requirements.txt
	@echo "now: source $(VENV)/bin/activate"

lock:
	$(UV) pip compile pyproject.toml --extra dev --universal --python-version 3.12 --generate-hashes --exclude-newer $(LOCK_DATE) --custom-compile-command "make lock" -o requirements.txt

lock-check:
	$(UV) pip install --dry-run --python-version 3.12 --python-platform x86_64-manylinux_2_28 --target "$$(mktemp -d)" --require-hashes --only-binary :all: -r requirements.txt
	$(UV) pip install --dry-run --python-version 3.12 --python-platform aarch64-manylinux_2_28 --target "$$(mktemp -d)" --require-hashes --only-binary :all: -r requirements.txt
	MACOSX_DEPLOYMENT_TARGET=14.0 $(UV) pip install --dry-run --python-version 3.12 --python-platform aarch64-apple-darwin --target "$$(mktemp -d)" --require-hashes --only-binary :all: -r requirements.txt

data:
	$(PYTHON) prepare_data.py --out data

weights:
	$(PYTHON) -m pilot.weights

test:
	$(PYTHON) -c "import sys; print('pytest in', sys.executable)"
	$(PYTHON) -m pytest -m "not mps and not data and not weights"

test-all:
	$(PYTHON) -c "import sys; print('pytest in', sys.executable)"
	$(PYTHON) -m pytest -m "not mps"

test-unit:
	$(PYTHON) -m pytest -m unit -q

baselines:
	$(PYTHON) score.py data/splits/val.jsonl --model majority --force
	$(PYTHON) score.py data/splits/test.jsonl --model majority --force
	$(PYTHON) score.py data/splits/val.jsonl --model prior --force
	$(PYTHON) score.py data/splits/test.jsonl --model prior --force
	$(PYTHON) score.py data/splits/val.jsonl --model laya --workers $(SCORE_WORKERS) --cpus-per-worker $(SCORE_CPUS) --force
	$(PYTHON) score.py data/splits/test.jsonl --model laya --workers $(SCORE_WORKERS) --cpus-per-worker $(SCORE_CPUS) --force
	$(MAKE) baseline-reports PYTHON="$(PYTHON)"

baseline-reports:
	mkdir -p $(REPORTS)
	$(PYTHON) report.py results/dumps/majority --split test --out $(REPORTS)/majority.json > /dev/null
	$(PYTHON) report.py results/dumps/prior --split test --out $(REPORTS)/prior.json > /dev/null
	$(PYTHON) report.py results/dumps/$(LAYA_ZS) --split test --out $(REPORTS)/$(LAYA_ZS).json > /dev/null
	$(PYTHON) report.py --compare results/dumps/$(LAYA_ZS) results/dumps/prior --split test --out $(REPORTS)/compare-$(LAYA_ZS)-vs-prior.json > /dev/null
	$(PYTHON) report.py --compare results/dumps/$(LAYA_ZS) results/dumps/$(LAYA_ZS) --split test --out $(REPORTS)/compare-$(LAYA_ZS)-vs-self.json > /dev/null
	$(PYTHON) report.py --table $(REPORTS)

berturk-postfit:
	mkdir -p results/postfit
	$(PYTHON) postfit.py $(addprefix results/dumps/,$(BERTURK_SELECTION)) --out results/postfit/berturk.json > /dev/null

clean-data:
	rm -rf data/splits data/audit
