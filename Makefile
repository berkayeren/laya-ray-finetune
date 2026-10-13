# Front door for the pilot. GNU Make 3.81 safe: no .ONESHELL, one command per recipe line, $$ escapes.
#   make env         create a python3.12 venv ($(VENV)) and install the hash lock binary-only
#   make lock        regenerate requirements.txt from pyproject.toml (universal, hashed, frozen date)
#   make lock-check  prove the lock resolves binary-only with hashes on Linux x86_64/aarch64 and mac arm64
#   make test        run pytest in the ACTIVE interpreter (no venv management)
PYTHON ?= python
UV ?= uv
PY312 ?= python3.12
VENV ?= .venv
LOCK_DATE := 2026-10-13T00:00:00Z

.PHONY: env lock lock-check data weights test test-unit clean-data

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
	$(PYTHON) -m pytest -m "not mps"

test-unit:
	$(PYTHON) -m pytest -m unit -q

clean-data:
	rm -rf data/splits data/audit
