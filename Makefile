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
#   make berturk-postfit   rebuild the Phase 6 BERTurk postfit record from the committed selection dumps (no model)
PYTHON ?= python
UV ?= uv
PY312 ?= python3.12
VENV ?= .venv
LOCK_DATE := 2026-10-13T00:00:00Z
LAYA_ZS := laya-zeroshot-7b928d82
BERTURK_SELECTION := berturk-e77a5cd81bdd-10bb633a-29446b61 berturk-6a89b0b079ef-11bcc29d-29446b61 berturk-d9d4bc7b1108-f4929084-29446b61 berturk-90f9d561b7ec-cdca5b18-29446b61 berturk-9d8257de5a33-0267a737-29446b61 berturk-e14e0c7c6f29-aa9f9f52-29446b61 berturk-6418d793ca58-a0bb710a-29446b61 berturk-a85391eb6087-36ac0406-29446b61 berturk-6ecbbbd6d2ba-f2a23b34-29446b61 berturk-611ac90c6d93-951f6f1d-29446b61
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
