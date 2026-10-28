<!-- generated-by: gsd-doc-writer -->
# Development

This page covers working on the code: setting up a development checkout, the `make` targets, the code conventions
the tests enforce, and how to get a change in. The [README](../README.md) is the full command reference,
[ARCHITECTURE.md](ARCHITECTURE.md) describes the components, [CONFIGURATION.md](CONFIGURATION.md) lists the flags and
defaults, and the [guide](GUIDE.md) explains the pipeline and how to read the results.

## Local setup

### Prerequisites

- **Python 3.12**, available as `python3.12` on your `PATH` (`make env` calls it; `pyproject.toml` requires
  `==3.12.*`).
- **git**. `make report` compares rebuilt files against git HEAD and refuses to run without git.
- **GNU Make**. The `Makefile` is written to run on GNU Make 3.81 (the macOS default).
- **Node.js >= 22.12.0 and npm**, only for the demo UI under `ui/` (`ui/package.json` `engines`; `ui/.npmrc` sets
  `engine-strict`, so an older Node is refused).
- **uv**, only for maintainers who regenerate or check the lock (`make lock`, `make lock-check`).
- **Hardware**: the committed results were produced on an Apple M4 Mac with 16 GB unified memory, using PyTorch MPS.
  The offline tests and `make report` need no GPU. One Ray worker holds about 2.6 GB of memory.

### Steps

1. Clone the repository and enter it:

   ```bash
   git clone https://github.com/berkayeren/laya-ray-finetune.git
   cd laya-ray-finetune
   ```

2. Build the locked environment and activate it:

   ```bash
   make env                      # python3.12 -m venv .venv, then pip install --require-hashes --only-binary :all:
   source .venv/bin/activate
   ```

   The install is hash-locked from `requirements.txt` and uses binary wheels only. Do not `pip install` anything
   else into `.venv`: `tests/test_env_lock.py` fails when an installed distribution is not in the lock, or when an
   installed version differs from it.

3. Write the local splits:

   ```bash
   make data                     # pinned Multi3NLU++ -> data/splits/*.jsonl and data/audit/, checked against data/splits.json
   ```

   The split files are gitignored; only the manifest `data/splits.json` is committed. Every consumer refuses a split
   whose sha256 differs from the manifest, so rerun `make data` whenever a split is missing or stale.

4. Run the offline tests:

   ```bash
   make test                     # or, without activating: make test PYTHON=.venv/bin/python
   ```

5. Optional: prefetch the pinned model weights (about 1.1 GB) before running the network tests or a model:

   ```bash
   make weights
   ```

There is no `.env` file to copy and no Python build or compile step. The project is never pip-installed
(`pyproject.toml` has no `[build-system]`): scripts run directly from the repository root (`python score.py ...`),
and pytest finds the modules through `pythonpath = ["."]`. Set `HF_HUB_OFFLINE=1` to reuse the Hugging Face cache
after the first fetch; the weights are still sha256-checked against `pilot/pins.py`.

The only build step is the demo UI, which `make demo` runs for you (`npm --prefix ui ci`, then
`npm --prefix ui run build` into `ui/dist`).

### Local outputs that stay out of git

Training, search and UI outputs are gitignored and stay in your checkout: `ray_results/` (Ray run storage),
`berturk_runs/` (BERTurk run storage), `tensorboard/` (event files of `--tensorboard tensorboard`), `ckpt*/`
(checkpoints, including the final seeds), `tune_runs/` (Ray Tune storage), `data/` apart from `data/splits.json`,
`ui/node_modules/` and `ui/dist/`.

## Build commands

All targets are defined in the `Makefile`. `PYTHON` defaults to `python` (the active interpreter); pass
`PYTHON=.venv/bin/python` to use the venv without activating it.

### Environment and data

| Command | Description |
|---------|-------------|
| `make env` | Create the Python 3.12 venv in `.venv` and install `requirements.txt` with `--require-hashes --only-binary :all:` |
| `make data` | `prepare_data.py --out data`: download the pinned dataset, verify all 40 raw fold hashes, write `data/splits/`, `data/audit/` and check `data/splits.json` |
| `make weights` | `python -m pilot.weights`: fetch and sha256-check the pinned Laya and BERTurk weights (about 1.1 GB) |
| `make clean-data` | Delete `data/splits` and `data/audit` (the committed manifest stays) |
| `make lock` | Maintainer: regenerate `requirements.txt` from `pyproject.toml` with `uv pip compile` (universal, hashed, resolved as of a frozen date) |
| `make lock-check` | Maintainer: prove the lock resolves binary-only with hashes on Linux x86_64, Linux aarch64 and macOS arm64 |

### Tests

| Command | Description |
|---------|-------------|
| `make test` | `pytest -m "not mps and not data and not weights"` in the active interpreter; prints `sys.executable` first |
| `make test-all` | `pytest -m "not mps"`: adds the `data` and `weights` tests, which download the pinned dataset and weights on first run |
| `make test-unit` | `pytest -m unit -q`: the offline unit tests only |
| `python -m pytest -m mps` | The tests that need Apple MPS (no make target) |
| `python -m pytest tests/test_metrics.py` | One test file; add `-k NAME` for one test |

The markers are strict (`--strict-markers`): `unit`, `data`, `weights`, `ray` (starts a local Ray instance) and
`mps`. `make test` includes the `ray` tests, such as the CPU smoke test that trains a tiny checkpoint with
`ray_train.py`, scores it with `score.py` and reports it with `report.py`.

### Rebuild records from the committed dumps (no model)

| Command | Description |
|---------|-------------|
| `make report` | Eval-only: verify the protocol, the splits and every record, rebuild every report, the README's generated blocks and the memo; fails, naming the path, on any rebuilt file that differs from git HEAD |
| `make report VERIFY_CHECKPOINTS=1` | Also re-hash the on-disk final seeds against their manifests (reported, not gated) |
| `make baseline-reports` | Rebuild `results/reports/` and print the baselines table |
| `make laya-postfit` | Rebuild `results/postfit/laya.json` from the 4 committed Laya selection val dumps |
| `make berturk-postfit` | Rebuild `results/postfit/berturk.json` from the 30 committed BERTurk selection val dumps |
| `make confidence-reports` | Rebuild the three val confidence records and their reliability SVGs under `results/confidence/` |
| `make final-reports` | Rebuild `results/final/`: the per-seed test reports, `verdict.json` (with the few-shot record) and the test and HOTELS confidence records |
| `make fewshot-reports` | Rebuild `results/fewshot/fewshot.json` from the committed few-shot dumps |
| `make memo` | Write `results/final/memo.md` from the committed records |

### Steps that run a model

| Command | Description |
|---------|-------------|
| `make reproduce` | The whole pipeline in order, skipping every step its committed, hash-verified artifacts prove done; on a finished tree it only rebuilds the report. `MODE=eval-only` is `make report` |
| `make baselines` | Rescore majority, prior and zero-shot Laya on val and test with `--force`, then `make baseline-reports`. It overwrites committed dumps, so do not run it casually |
| `make final-scoring` | Score the six final seeds once on test and HOTELS behind the protocol gate. It already ran; the gate refuses a rerun |
| `make fewshot-scoring` | Score every few-shot checkpoint listed in `results/fewshot/runs.json` once behind the protocol gate. It already ran; the gate refuses a rerun |

### Demo UI

| Command | Description |
|---------|-------------|
| `make demo` | `npm --prefix ui ci` (skipped while `ui/node_modules` was installed from the current `ui/package-lock.json`), `npm --prefix ui run build`, then `python demo.py --device mps` on http://127.0.0.1:8765 |
| `make demo DEMO_DEVICE=cpu` | The same with the models on the CPU |
| `npm --prefix ui run typecheck` | `tsc -p tsconfig.json` over the UI without building it |

Useful commands that run nothing: `python -m pilot.reproduce --plan` prints the pipeline steps and whether each is
done, `python reproduce.py --dry-run --all` prints every step's commands, and `python ray_train.py ... --print-run-id`
prints a training run's id without starting Ray. For a quick end-to-end run on a CPU, the README's smoke fine-tune
(`ray_train.py ... --workers 2 --freeze-encoder --max-messages 100 --epochs 2`) is the smallest real training command.

## Code style

**No linter or formatter is configured.** There is no ESLint, Prettier, Ruff, Black or `.editorconfig`, and no lint
script. Code is hand-formatted to the conventions below, and several of them are enforced by tests rather than by a
tool.

- **Formatting**: PEP 8-like, 4-space indents, double quotes, lines up to about 120 characters, hand-formatted.
  Messages use `%`-formatting (`"%s: %d messages" % (...)`), not f-strings.
- **Naming**: snake_case modules and functions, UPPER_CASE module constants (`INTENTS`, `QUESTIONS`, `SPLITS`), terse
  locals (`a` for parsed args, `c` for the train config, `tok`, `mb`). The only production class is `Predictor` in
  `score.py`, a Ray Data actor.
- **Type hints**: every function and method carries parameter and return annotations with Python 3.12 builtin
  generics (`list[str]`, `dict[str, Any]`), `X | None`, and `typing.Any` where the precise type would need a heavy
  import. Annotations must be true.
- **Import budgets**: several modules must stay torch-free or numpy-free, and subprocess tests fail when they import
  more. Examples: `pilot/protocol.py` and `pilot/fewshot.py` use the stdlib only, `report.py` is numpy only,
  `pilot/tensorboard.py` imports tensorboardX only inside `open_writer`, and `pilot/defaults.py` imports nothing.
  Never add a heavy runtime import for an annotation.
- **Refusals**: errors are `raise SystemExit("<what is wrong>; <how to fix>")`, raised before any model load or
  `ray.init` where possible. Argument errors use `ap.error(...)`.
- **Docstrings**: every script and `pilot/` module opens with a docstring stating its purpose, design notes and
  `usage:`. Helpers that encode a contract state it and the return value in a docstring.
- **I/O**: open files with `encoding="utf-8"`, write JSON with `ensure_ascii=False`, and write outputs atomically
  with `pilot.data.write_atomic`. Records carry repo-relative paths (`pilot.data.repo_relpath`) and sha256s, never
  local paths.
- **Single sources**: `QUESTIONS` is defined only in `intents.py` (a test fails on a second definition), and Laya's
  private API is used only in `pilot/laya_adapter.py` (an AST guard in `tests/test_laya_adapter.py` fails on a private
  `laya` import anywhere else). When the lock moves `laya`, fix the adapter, not its callers.
- **Frozen records**: never edit `protocol.json`, anything under `data/`, or any `.json`, `.npz` or `.svg` under
  `results/`. A string constant whose text a committed record holds (rule texts, record keys, schema names) must stay
  byte-identical. Markdown under `results/` and the README's generated blocks (between `<!-- NAME:start -->` and
  `<!-- NAME:end -->` markers) are regenerated from the records by `make report`, never edited by hand.
- **Repository hygiene**: `tests/test_repo_hygiene.py` rejects message text from the dataset and local home paths in
  tracked files, and committed dumps that hold anything but float32 logits and uids. Never commit customer data or a
  bank's name.

Some prose is bound by tests too: README sections, `CLAUDE.md` and `.claude/CLAUDE.md` are checked for required
phrases and for withdrawn wording (for example `tests/test_defaults.py`, `tests/test_latency_results.py`), and each
README generated block must equal its render. Run `make test` after editing those files.

## Branch conventions

No branch naming convention is documented. The default branch is `main`, and the history on it is linear (no merge
commits).

Commit messages follow the Conventional Commits shape `type(scope): subject`, with a short lower-case subject. The
types in the history are `feat`, `fix`, `test`, `docs`, `refactor`, `chore`, `style` and `data`. Commit
`pyproject.toml` and `requirements.txt` together whenever a pin changes (`make lock` regenerates the lock).

## PR process

There is no pull request template; `CONTRIBUTING.md` lists the hard rules. Before you
submit a change:

- Run `make test` in the locked environment and make sure it passes. Run `make test-all` as well when you touch data
  preparation, weight handling or anything a committed record is rebuilt from.
- Commit the change, then run `make report` on the clean tree. It must end with `git status clean`; a differing
  rebuilt file means the change altered a committed result, and the change must be reverted or the difference
  explained.
- Keep frozen records byte for byte (see [Code style](#code-style)). A change that would need a new split manifest,
  a rescored gated dump or an edited `protocol.json` is a protocol change, not a fix.
- Keep the import budgets, the type hints and the refusal style, and add or update tests for any new contract.
- If you change a tag wording in `intents.py`, regenerate the data and rerun the baselines: the question text is the
  model input, and `data/splits.json` records `questions_sha256`, so a wording change without a refreeze fails
  `make test`.
