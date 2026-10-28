<!-- generated-by: gsd-doc-writer -->
# Contributing

Thanks for helping. This repository produces an auditable go/no-go recommendation. Its committed records are the
audit trail, so most of the rules below protect them. Please read the hard rules before you change anything.

## Development setup

See the [Setup section of the README](README.md#setup) for the environment, data and test commands. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how the modules fit together, and
[docs/CONFIGURATION.md](docs/CONFIGURATION.md) for the CLI flags, make variables and pins. If you are new to machine
learning, start with [docs/GUIDE.md](docs/GUIDE.md). The short version:

```bash
make env && source .venv/bin/activate   # Python 3.12 venv from the hash-locked requirements.txt
make data                               # pinned Multi3NLU++ -> data/splits/ (gitignored except data/splits.json)
make test                               # offline suite: pytest -m "not mps and not data and not weights"
```

## Hard rules

These rules are enforced by tests or by the review of every change. A pull request that breaks one will not be merged.

- **Frozen records stay byte for byte.** Never edit `protocol.json`, anything under `data/`, or any `.json`, `.npz`
  or `.svg` file under `results/`. They are rebuilt from the committed dumps by the make targets, never by hand. If a
  rebuild differs, the code changed a committed result. Fix the code, not the record.
- **Frozen text stays frozen.** A string constant in the code whose text a committed record holds must stay
  byte-identical. This covers the protocol, verdict, metric, latency and few-shot rule texts, record keys and schema
  names. If you change one, the records stop rebuilding.
- **Generated markdown is regenerated, never hand-edited.** That covers the markdown under `results/` and the README
  blocks between `<!-- NAME:start -->` and `<!-- NAME:end -->` markers. `make report` rewrites the README blocks
  from their renders (the registry is `README_BLOCKS` in `pilot/reproduce.py`). Edit the render function, then
  rebuild.
- **No dependency without the lock.** Add an exact pin (`name==X.Y.Z`) to `pyproject.toml`, run `make lock` (needs
  `uv`) and `make lock-check`, and commit `pyproject.toml` and `requirements.txt` together.
  `tests/test_env_lock.py` fails when the installed environment does not match the lock.
- **No real customer data and no bank name.** The dataset is a public stand-in. Never commit customer messages, a
  bank's name, message text from the splits (only `data/splits.json` is tracked under `data/`), or a local absolute
  path. `tests/test_repo_hygiene.py` checks the tracked files for these.
- **Laya's private API stays in one module.** Only `pilot/laya_adapter.py` may import Laya internals. An AST guard in
  `tests/test_laya_adapter.py` fails on any other importer. When the lock moves `laya`, fix the adapter, not the
  callers.
- **Thresholds are never tuned on test.** T, θ and the epoch come from val only. Do not add a path that reads test or
  HOTELS content to select anything, and do not weaken `score.py`'s protocol gate. Never pass `--force` to a gated
  scoring command.
- **Training outputs stay out of git.** `ray_results/`, `berturk_runs/`, `tune_runs/`, `tensorboard/`, `ckpt*/` and
  `*.safetensors` are gitignored. Keep it that way.

## Coding standards

There is no linter or formatter, and there is no CI workflow in the repository. Run the checks yourself before you
open a pull request:

- **Style:** PEP 8-like, 4-space indent, double quotes, lines up to about 120 columns, hand-formatted. Use snake_case
  for modules and functions and UPPER_CASE for module constants. Use `%`-formatting for messages.
- **Type hints:** every function and method carries parameter and return hints in Python 3.12 builtin generics
  (`list[str]`, `X | None`). Use `typing.Any` where a precise type would need a heavy import. Tests guard the import
  budgets of the torch-free and numpy-free modules, so never add a runtime import just for an annotation.
- **Refusals and I/O:** a refusal is `raise SystemExit("<what is wrong>; <how to fix>")`, raised before any model
  load or `ray.init` where possible. Argument errors use `ap.error(...)`. Open files with `encoding="utf-8"` and
  write JSON with `ensure_ascii=False`, atomically (`pilot.data.write_atomic`). Records carry repo-relative paths and
  sha256s.
- **Docstrings:** every script and `pilot/` module opens with a docstring (purpose, design notes, `usage:`). A helper
  that encodes a contract states the contract and its return value.
- **Tests:** `make test` must pass. If you touch scoring, reports, records or renders, also run `make test-all`
  (downloads the pinned dataset and about 1.1 GB of weights on first run) and `make report`. `make report` rebuilds
  every report, README block and the memo from the committed dumps, and fails on any differing file. On an Apple
  Silicon Mac, `pytest -m mps` runs the MPS tests.

## PR guidelines

- **Branch:** the default branch is `main`. Work on a topic branch. No naming convention is documented; a short
  descriptive name such as `fix/score-gate-message` works.
- **Commits:** use conventional-commit style subjects, `type(scope): summary`, with the types the history uses:
  `feat`, `fix`, `test`, `docs`, `refactor`, `chore`, `style` and `data` (a new committed record). Keep each commit to
  one change, and keep a new committed record in its own commit.
- **Tests:** add or update tests with every behaviour change. Pin a new refusal with a test that triggers it. Use the
  offline helpers in `tests/` (`tiny_laya.py`, `tiny_berturk.py`, `fewshot_synthetic.py`) instead of real weights,
  and mark a test with the strict markers (`unit`, `data`, `weights`, `ray`, `mps`) it needs.
- **Docs:** when you change a command, flag or contract, update `README.md` and `CLAUDE.md` in the same pull request.
  Several tests check README prose and generated blocks against the code.
- **Description:** state what changed and why, which test commands you ran (`make test`, `make test-all`,
  `make report`), and whether any committed record or README block would change. A pull request that would change a
  frozen record needs an explicit explanation, and is usually refused.

## Issue reporting

Report bugs and request features at
[github.com/berkayeren/laya-ray-finetune/issues](https://github.com/berkayeren/laya-ray-finetune/issues). There are no
issue templates. For a bug, please include:

- the exact command you ran and its full output, including the one-line refusal if there was one
- what you expected and what happened instead
- your environment: OS and hardware (for example Apple Silicon with MPS, or CPU only), the Python version, and whether
  `make test` passes in that interpreter
- the git commit you ran (`git rev-parse HEAD`) and whether the tree had local changes (`git status`)

Never paste customer messages, a bank's name or message text from the splits into an issue.

## License

By contributing, you agree that your contributions are licensed under the Apache License 2.0. See [LICENSE](LICENSE).
