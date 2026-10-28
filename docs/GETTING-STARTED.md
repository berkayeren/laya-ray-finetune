<!-- generated-by: gsd-doc-writer -->
# Getting started

This page takes you from a fresh clone to a working environment: the locked Python environment, the verified data
splits, a passing offline test suite and a check that the committed results rebuild from the committed dumps. The
[README](../README.md) is the full command reference and the [guide](GUIDE.md) explains the pipeline and its numbers.

## Prerequisites

| Requirement | Version | Needed for |
|-------------|---------|------------|
| Python | `==3.12.*`, on your `PATH` as `python3.12` | everything (`pyproject.toml` `requires-python`; `make env` calls `python3.12`) |
| GNU Make | 3.81 or newer | the `make` targets (the `Makefile` is written to be 3.81 safe) |
| git | any recent version | cloning; `make report` compares rebuilt files against git HEAD and refuses to run without git |
| Network access to the Hugging Face Hub | n/a | `make data` (the pinned Multi3NLU++ revision) and the pinned model weights, about 1.1 GB |
| Node.js and npm | `Node.js >= 22.12.0` | the local demo UI only (`ui/package.json` `engines`; `ui/.npmrc` sets `engine-strict=true`) |
| uv | any | maintainers only: `make lock` and `make lock-check` |

Platform and hardware:

- **macOS on Apple silicon (MPS)** is the primary platform. The committed results were produced on an Apple M4 with
  16 GB unified memory. Rebuilding the reports needs no GPU; training and the frozen protocol's scoring use MPS.
- **Linux** installs the default PyPI `torch==2.14.1` wheel, which is built for CUDA 13: it needs an sm_75+ GPU and a
  580-series or newer driver for `--device cuda`. CPU-only runs work without one.
- **Memory:** one Ray worker holds about 2.6 GB; plan for 16 GB or more per worker. Run one MPS job at a time.
- **Disk:** about 1.5 GB for `.venv`, about 1.1 GB for the weights in the Hugging Face cache, a few MB for the data.
  Training runs need much more transient space and refuse to start when free disk is too low.

## Installation steps

1. Clone the repository and enter it:

   ```bash
   git clone https://github.com/berkayeren/laya-ray-finetune.git
   cd laya-ray-finetune
   ```

2. Create the virtual environment and install the hash-locked dependencies (binary wheels only):

   ```bash
   make env                   # python3.12 -m venv .venv, then pip install --require-hashes --only-binary :all: -r requirements.txt
   source .venv/bin/activate
   ```

   The project is never pip-installed (there is no `[build-system]`); the scripts are run directly with `python`.

3. Download and verify the pinned dataset, then write the frozen splits:

   ```bash
   make data                  # pinned Multi3NLU++ -> data/splits/*.jsonl and data/audit/, checked against data/splits.json
   ```

   `make data` checks all 40 raw fold sha256s and every output against the committed manifest `data/splits.json`.
   The split files are local and gitignored; only the manifest is committed.

4. Optionally prefetch the pinned Laya and BERTurk weights (about 1.1 GB, sha256-checked after download):

   ```bash
   make weights
   ```

## First run

Run the offline test suite in the activated environment:

```bash
make test                     # pytest -m "not mps and not data and not weights"; prints the interpreter first
```

Then prove that every committed number follows from the committed dumps and the frozen protocol:

```bash
make report                   # same as: make reproduce MODE=eval-only
```

`make report` never trains or scores. It verifies the protocol pin, the splits and every committed record, rebuilds
every report under `results/`, the README's generated blocks and `results/final/memo.md` from the committed dumps, and
fails, naming the path, if any rebuilt file differs from the committed one. If the local split files are missing it
runs the data step first. A clean pass prints a line starting with `eval-only done:`.

To see a model run, score zero-shot Laya on a 20-message debug subset of val, in-process (the dump goes to a temporary
directory, never `results/dumps`), and print its report:

```bash
python evaluate.py data/splits/val.jsonl --checkpoint multilingual --workers 0 --n 20
```

The first call fetches and verifies the pinned Laya weights if `make weights` has not run.

## Common setup issues

**`make env` fails with `python3.12: No such file or directory`.**
The `Makefile` calls the interpreter named by `PY312` (default `python3.12`). Install Python 3.12, or point the
target at an existing 3.12 interpreter: `make env PY312=/path/to/python3.12`.

**`make test` fails in `tests/test_env_lock.py` with `installed packages differ from requirements.txt` or `the lock is
for 3.12`.**
`make test` runs in whatever interpreter is active, and a test fails when that interpreter does not match the lock.
Activate the venv (`source .venv/bin/activate`), run `make test PYTHON=.venv/bin/python`, or recreate the environment
with `make env`.

**`... not found; check the path, or run `make data` to write the registered files`, or a split's sha256 does not
match `data/splits.json`.**
The split files are gitignored, so a fresh clone has none, and a stale or edited split is refused with no override.
Run `make data`. Do not use `prepare_data.py --rewrite-manifest`: it refreezes the splits and invalidates every
downstream result.

**`--device mps, but MPS is not available here; use --device cpu`.**
The machine has no Apple GPU (or PyTorch cannot reach it). Use `--device cpu`. Scoring on MPS also needs
`--workers 0`, because MPS runs in-process, not inside Ray actors.

**`make report` refuses with `eval-only compares every rebuilt file with the committed one, but these already differ`.**
It needs `results/`, `README.md`, `protocol.json`, the `Makefile`, `reproduce.py` and `pilot/reproduce.py` to match
git HEAD before it starts. Commit, stash or restore them (`git checkout -- PATH`).

**`results/dumps/.../val.npz already exists; pass --force to rescore it ...`.**
`results/dumps/` is the committed audit record and an existing dump is never overwritten silently. Use `--n N`
(written to a temporary directory) or `--out-root DIR` for your own runs; do not pass `--force` over a committed dump.

**No network after the first download.**
Set `HF_HUB_OFFLINE=1` to reuse the Hugging Face cache; the weights are still sha256-checked against `pilot/pins.py`.

**`make demo` fails in `npm ci` with an engine error, or `demo.py` reports `ui/dist/index.html not found`.**
The UI requires `Node.js >= 22.12.0` and `engine-strict` refuses an older one. Upgrade Node, then run `make demo`,
which installs the UI, builds `ui/dist` and starts the server on http://127.0.0.1:8765 (`make demo DEMO_DEVICE=cpu`
runs the models on the CPU). The chat needs the six final-seed checkpoint directories, which are not in git; without
them the chat refuses in one line and the dashboard of committed records still works.

## Next steps

- [README](../README.md): every command (training, scoring, reports, the frozen protocol, few-shot, reproduction,
  the exploratory search and the demo) and the generated result blocks, including the recommendation.
- [Guide](GUIDE.md): what the pilot measures, how to read the results table and the GO block, and a table of common
  refusals with their fixes.
- [Architecture](ARCHITECTURE.md): the components, the data flow from splits to verdict, and the design invariants.
- [Configuration](CONFIGURATION.md): the flags, `make` variables, pinned files and defaults that change behaviour.
- `make test-all` adds the tests that download the pinned dataset and weights; `python -m pilot.reproduce --plan`
  shows the pipeline steps and which are already done.
