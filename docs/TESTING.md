<!-- generated-by: gsd-doc-writer -->
# Testing

The test suite is a set of pytest contracts and guards. Most tests run offline on CPU in seconds, using a tiny
synthetic dataset and tiny checkpoints. Others prove that the committed results under `results/` and the generated
README blocks still rebuild from the committed dumps. A few tests need the pinned dataset, the pinned weights, a local
Ray instance or Apple MPS. Each of those is selected by a pytest marker.

## Test framework and setup

- **Framework:** `pytest==9.1.1` with `pytest-timeout==2.4.0`. Both are in the `dev` extra of `pyproject.toml` and
  hash-locked in `requirements.txt`, so `make env` installs them. `torchmetrics==1.9.0` and `scikit-learn` (a runtime
  dependency) serve as test oracles for the metric code.
- **Configuration:** `[tool.pytest]` in `pyproject.toml`. There is no `pytest.ini`, `setup.cfg` or `tox.ini`.

  ```toml
  [tool.pytest]
  minversion = "9.0"
  testpaths = ["tests"]
  pythonpath = ["."]
  addopts = ["-ra", "--strict-markers", "--strict-config"]
  ```

  `pythonpath = ["."]` puts the repo root on `sys.path`. The project is never pip-installed, so tests import the
  scripts (`score`, `report`, `ray_train`, ...) and `pilot.*` directly. `--strict-markers` makes an unregistered
  marker an error.
- **Markers** (registered in `pyproject.toml`):

  | Marker | Meaning |
  |--------|---------|
  | `unit` | Offline and fast. No weights or data downloads. |
  | `data` | Needs the pinned Multi3NLU++ download (network on the first run). |
  | `weights` | Fetches and verifies the pinned HF weights (about 1.1 GB on the first run). |
  | `ray` | Starts a local Ray instance. |
  | `mps` | Needs Apple MPS. |

  Most files set a module-level `pytestmark = pytest.mark.<marker>`. Other files mark single tests with
  `@pytest.mark.data` or `@pytest.mark.weights` inside an otherwise `unit` file. `tests/test_berturk_train.py`,
  `tests/test_berturk_smoke.py` and `tests/test_spike_real.py` have no module marker.
- **Environment:** the suite must run in the locked environment. `tests/test_env_lock.py` compares every installed
  distribution with `requirements.txt` and fails, listing each mismatch, when the active interpreter does not match
  the lock.

  ```bash
  make env && source .venv/bin/activate     # Python 3.12 venv, pip install --require-hashes --only-binary :all:
  ```

## Running tests

The Makefile targets run in the **active** interpreter (`$(PYTHON)`, default `python`) and print `sys.executable`
first. To run without activating the venv, pass the interpreter:

```bash
make test PYTHON=.venv/bin/python
```

| Command | Runs | Notes |
|---------|------|-------|
| `make test` | `pytest -m "not mps and not data and not weights"` | Offline. This includes the `ray` tests and the unmarked files. |
| `make test-all` | `pytest -m "not mps"` | Adds the `data` and `weights` tests, which download the pinned dataset and weights on the first run. |
| `make test-unit` | `pytest -m unit -q` | Only `unit`-marked tests. The `ray` tests and the unmarked files are skipped. |
| `pytest -m mps` | the `mps` tests | Apple MPS only. Never part of a make target. |

Run the network-dependent tests in this order:

```bash
make data          # the data tests read data/splits/*.jsonl; without it they fail or skip, naming `make data`
make weights       # optional: prefetch and verify the ~1.1 GB pinned Laya and BERTurk weights
make test-all
```

Run a subset, one file or one test:

```bash
python -m pytest tests/test_metrics.py                            # one file
python -m pytest tests/test_metrics.py::test_single_row_input     # one test
python -m pytest -m ray                                           # every test that starts a local Ray instance
python -m pytest -m data                                          # the dataset-backed tests (after make data)
python -m pytest -m mps tests/test_score_mps.py                   # one MPS file on an Apple GPU
python -m pytest --collect-only -q -m "not mps and not data and not weights"   # list what make test would run
```

There is no watch mode.

**MPS tests.** The `mps` files (`test_berturk_mps.py`, `test_demo_mps.py`, `test_latency_mps.py`, `test_score_mps.py`,
`test_spike_mps.py`, `test_train_mps.py`, `test_tune_mps.py`, plus one test in `test_reproduce.py` that is also marked `data` and `weights`) run the tiny models,
and some of them the real final seeds, on the Apple GPU. Run one MPS job at a time, on a quiet machine.
`tests/test_demo_mps.py` needs the six final-seed checkpoint dirs (`ckpt-final-s0..2`, `ckpt-berturk-final-s0..2`,
gitignored) in the main checkout and about 6 GiB of free memory. It is skipped, naming the missing dirs, when any is
absent.

**UI build.** `tests/test_demo_ui.py::test_built_bundle_is_served` skips until the UI is built:

```bash
npm --prefix ui ci && npm --prefix ui run build
```

The UI under `ui/` has no JavaScript test runner. Its contracts (exact npm pins, the lockfile, `ui/.npmrc`, the
copy and route checks, and the built bundle served by the demo) are pytest tests in `tests/test_demo_ui.py`.

## Writing new tests

**File naming.** Files are `tests/test_<area>.py`. A suffix names the kind of test:

| Pattern | Kind | Examples |
|---------|------|----------|
| `test_<area>_unit.py`, `test_<area>.py` | Offline logic and refusal tests | `test_runs_unit.py`, `test_metrics.py`, `test_verdict.py` |
| `test_<area>_ray.py`, `test_train_*.py`, `test_*_smoke.py` | Subprocess runs of the real scripts on tiny models, with Ray | `test_score_ray.py`, `test_train_resume.py`, `test_tune_smoke.py` |
| `test_<area>_cpu.py` | End-to-end runs on CPU without Ray | `test_demo_cpu.py`, `test_latency_cpu.py` |
| `test_<area>_mps.py` | The same on Apple MPS | `test_score_mps.py`, `test_train_mps.py` |
| `test_<area>_results.py`, `test_results.py` | Committed records and README blocks rebuild from the committed dumps | `test_postfit_results.py`, `test_final_results.py` |
| `test_<area>_real.py`, `test_real_data.py` | The real pinned dataset or weights | `test_weights_real.py`, `test_spike_real.py` |

**Conventions** (shared with the production code):

- Open every file with a module docstring that states what it proves and which marker or prerequisite it needs.
- Set `pytestmark = pytest.mark.<marker>` at module level. Mark individual tests only when a file mixes kinds.
- Give every test and helper parameter and return type hints, for example `def test_x(tmp_path: Path) -> None:`.
- Assert refusals by their message: `with pytest.raises(SystemExit, match="no messages"):`.
- Tests that run a script in a subprocess carry `@pytest.mark.timeout(TIMEOUT)`. A module constant sets the limit,
  for example `TIMEOUT = 600` in `test_train_ray.py` or `TIMEOUT = 1800` in `test_tune_mps.py`.
- Tests that need a directory the user cannot write are skipped as root or on non-POSIX systems:
  `@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs a dir the user cannot write")`.
- Never write under `results/`, `data/` or `protocol.json`. Build inputs under `tmp_path`, and redirect module-level
  pins with `monkeypatch.setattr`, for example `pilot.data.DEFAULT_MANIFEST`, `pilot.protocol.PROTOCOL_PATH` and
  `PROTOCOL_SHA256`, or `pilot.dumps.DUMP_ROOT`.
- When a module must stay free of a heavy import, guard it in a fresh interpreter, the way
  `tests/test_tune_unit.py::test_import_budget` does:

  ```python
  def test_import_budget() -> None:
      banned = ("torch", "scipy", "laya", "ray", "transformers", "sklearn")
      code = "import sys, pilot.tune; print(sorted(m for m in %r if m in sys.modules))" % (banned,)
      out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
      assert out.stdout.strip() == "[]", out.stdout + out.stderr
  ```

**Shared helpers** (in `tests/`; pytest puts `tests/` on `sys.path`, so test modules import them by name, for example
`from tiny_laya import tiny_checkpoint`):

| Helper | Purpose |
|--------|---------|
| `conftest.py` | The `synthetic_raw` fixture: a tiny synthetic Multi3NLU++ snapshot (20 banking and 20 hotel folds) with the edge rows the data pipeline must handle (no-intent rows, cross-split duplicates, Turkish I/İ casing, conflicting labels, optional duplicate uids). Also `make_raw`, `add_raw_row`, `prepared(tmp_path, synthetic_raw)` (runs `pilot.data.prepare` and returns the output dir and manifest path), `prepared_with_line_separators` and `LINE_SEPARATOR_TEXTS`. |
| `tiny_laya.py` | `tiny_checkpoint(path, temperature=1.0, seed=0)`: an offline 2-layer, 64-wide ModernBERT + Laya head checkpoint in the layout `laya.load` reads, built through `pilot.laya_adapter`. `tiny_tokenizer()`. |
| `tiny_berturk.py` | `tiny_base(path, seed=0)`: an offline cased 2-layer, 32-wide BERT base in the layout of the pinned BERTurk snapshot. `berturk_train.py --base DIR` adds the 48-label head. |
| `fewshot_synthetic.py` | `synthetic_protocol(...)`: a synthetic frozen protocol and compute decision record, pinned by monkeypatch (never the committed pin). `bench(...)`, `plant_laya`, `plant_berturk` and `plant_fewshot_set` plant few-shot checkpoints without training. `plant_runs` and `plant_fewshot_dumps` plant results inputs. `postfit_systems`, `decision_record`, `fewshot_block`, `split_gold`. |
| `tb_events.py` | A crc-checked TensorBoard event reader that needs no tensorboard package: `event_files`, `read_events`, `scalars` and `visible` (TensorBoard's purge rule). |

A typical CPU end-to-end test builds its data and checkpoint in a module-scoped fixture and then runs the real
scripts. `tests/test_smoke_cpu.py` does it like this:

```python
@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str, str, str]:
    d = tmp_path_factory.mktemp("smoke")
    root, raw_sha256 = make_raw(str(d / "raw"))
    out_dir = str(d / "data")
    pilot.data.prepare(root, out_dir, raw_sha256)
    manifest = os.path.join(out_dir, "splits.json")
    base = tiny_checkpoint(d / "base")                      # absolute local --base: the pinned fetch is never taken
    ...
```

**Committed results.** Several `*_results.py` tests compare a README generated block with its render. After a change
that alters a render, regenerate the block with `make report`. Never edit it by hand. Under the `data` marker these
tests also rebuild each committed record byte for byte from the committed dumps. `tests/test_repo_hygiene.py` checks
the tracked tree: only `data/splits.json` is tracked under `data/`, there is no message text or local home path, and
committed dumps hold only float32 logits and uids.

## Coverage requirements

No coverage threshold is configured. There is no coverage tool in the dependencies (no `pytest-cov`, `coverage` or
`.coveragerc`) and no coverage section in `pyproject.toml`. The suite's guarantees come from explicit contract tests
instead. Examples are the adapter signature snapshot in `tests/test_laya_adapter.py`, the 36 verdict status
combinations in `tests/test_verdict.py`, the import-budget guards and the byte-for-byte record rebuilds.

## CI integration

The repository has no CI configuration. There is no `.github/workflows/` directory and no other CI config file. Tests
are run locally:

- `make test` before a commit (offline, in the locked environment).
- `make test-all` after `make data` when a change touches the data, the records or the weights path.
- `pytest -m mps` on an Apple Silicon machine when a change touches a device path.
- `make lock-check` (maintainers, needs `uv`) after `make lock`, to prove that the lock still resolves binary-only
  with hashes on Linux x86_64, Linux aarch64 and macOS arm64.
- `make report` to verify the frozen protocol, the splits and the records, and to rebuild every report and README
  block from the committed dumps. It fails, naming the path, when a rebuilt file differs from the committed one.

See the [README](../README.md) for setup, [ARCHITECTURE.md](ARCHITECTURE.md) for the components
the tests mirror, and [CONFIGURATION.md](CONFIGURATION.md) for the flags and `make` variables.
