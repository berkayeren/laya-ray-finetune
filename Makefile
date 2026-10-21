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
#   make laya-postfit      rebuild the Phase 5 Laya postfit record from the committed selection dumps (no model)
#   make berturk-postfit   rebuild the Phase 6 BERTurk postfit record from the committed 30-epoch selection dumps (no model)
PYTHON ?= python
UV ?= uv
PY312 ?= python3.12
VENV ?= .venv
LOCK_DATE := 2026-10-13T00:00:00Z
LAYA_ZS := laya-zeroshot-7b928d82
LAYA_SELECTION := laya-283e9965bce5-70ce7445-cc216d59 laya-5efa46dfddaf-2f6120ee-cc216d59 laya-047381259d3b-cf0f7846-cc216d59 laya-f1ffb616dc78-ebdabc55-cc216d59
BERTURK_SELECTION := berturk-a4dba627ab53-9606916e-29446b61 berturk-7018cd8c39ba-56f1ba60-29446b61 berturk-7e9279793658-a29859ed-29446b61 berturk-a3b0182f1366-302ad85b-29446b61 berturk-d6590855c6b1-d8bf46b2-29446b61 berturk-10912cb0d577-a9c7e9c1-29446b61 berturk-b6119d5e7e32-2b39c313-29446b61 berturk-d84605009dea-32d589c0-29446b61 berturk-6e5acb25e2c1-c4d1d22e-29446b61 berturk-bb2c32535cef-cda54cff-29446b61 berturk-4c412f111dce-db7ddb00-29446b61 berturk-950a551d4bd7-1a12128f-29446b61 berturk-27010875323f-a05aa467-29446b61 berturk-6a2a42f69490-dcf55758-29446b61 berturk-0b980819a020-f584f4a5-29446b61 berturk-9ba66ee64598-5d594954-29446b61 berturk-2f1924130e5d-bd688c6b-29446b61 berturk-3bcedd6ba4f9-b21ac65b-29446b61 berturk-8ee54cba739a-24409e49-29446b61 berturk-82401cb57307-22d46a07-29446b61 berturk-4d4e72354696-955a3508-29446b61 berturk-c2e860d224e0-0d35e06d-29446b61 berturk-cc5662b6cfce-44ddea4c-29446b61 berturk-5f02e1608b5f-1784d55f-29446b61 berturk-25be2a0d4ab9-b99ef664-29446b61 berturk-ccea5efdd206-a1cee246-29446b61 berturk-437d39ceee87-36852e33-29446b61 berturk-565630476668-49edbd79-29446b61 berturk-f24d8fe3c899-50e28523-29446b61 berturk-f1674d98ca65-c5979937-29446b61
SCORE_WORKERS ?= 2
SCORE_CPUS ?= 4
REPORTS := results/reports

.PHONY: env lock lock-check data weights test test-all test-unit clean-data baselines baseline-reports laya-postfit berturk-postfit

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

laya-postfit:
	mkdir -p results/postfit
	$(PYTHON) postfit.py $(addprefix results/dumps/,$(LAYA_SELECTION)) --out results/postfit/laya.json > /dev/null

berturk-postfit:
	mkdir -p results/postfit
	$(PYTHON) postfit.py $(addprefix results/dumps/,$(BERTURK_SELECTION)) --out results/postfit/berturk.json > /dev/null

clean-data:
	rm -rf data/splits data/audit
