"""One-command reproduction (ENV-03, Phase 10): the CLI over pilot/reproduce.py, as postfit.py is over pilot/postfit.py.

--mode eval-only (= `make report`) rebuilds the evaluation and every report from the committed dumps, without training
or scoring (D-05, D-06). It prints the interpreter and the step plan of the working tree, then verifies first: the
frozen protocol (require_frozen), every split and manifest hash and every committed record and dump through the step
registry (pilot.reproduce.plan). A missing dump or record is refused before any command, naming its path and the step
that makes it; `make data` runs only when the local split files are absent. Then it rebuilds, through the make
targets and CLIs that exist (pilot.reproduce.REBUILD): the baseline reports, both postfit records, the BERTurk
reference check, the confidence records, the latency markdown, the few-shot record, results/final/ (the verdict with
--fewshot) and the memo. It rewrites the README's generated blocks from their renders (pilot.reproduce.README_BLOCKS)
and fails, naming each path, when a rebuilt file under results/, README.md or protocol.json differs from the committed
one. A tree whose checked paths already differ is refused before anything runs, so a difference is always the
rebuild's.

The gated splits are scored exactly once (protocol.json), so eval-only never rescores a checkpoint: criterion 2's
re-scoring is met by the committed dumps. --verify-checkpoints (`make report VERIFY_CHECKPOINTS=1`) adds a check that
is reported, not gated: each final seed listed in results/postfit/<kind>-final.json is re-hashed under
--checkpoint-root (default the repo root) with the identity functions score.py uses and reported as match, differs or
absent. It loads no model, writes nothing and never changes the exit status (no committed val dump of a final seed
exists to re-score against: the final seeds are scored once, on test and HOTELS).

--dry-run prints the interpreter, the step plan of the working tree and each todo step's exact commands (--all: every
step's, in D-01 order), built from the committed records, protocol.json and the Makefile (pilot.reproduce.commands),
with the D-04 disclosure before the first model step listed (pilot.reproduce.DISCLOSURE). It runs nothing and exits 0.
No printed command passes --force or runs `make baselines`.

--mode full (the default; = `make reproduce`) runs the whole pipeline in D-01 order (pilot.reproduce.run_full): data,
val selection, final training, evaluation, few-shot, report. It skips every step its committed, hash-verified
artifacts prove done, so on a finished tree only the report step (the eval-only rebuild above) runs. A todo step's
commands run in order after the D-04 disclosure; the run stops at the first failing command (a protocol gate refusal
is final) or at the first output that differs from the committed record or a protocol pin, and a gated scoring step
runs only after the training steps that make its checkpoints ran in the same invocation.

usage:
    python reproduce.py                                # = make reproduce PYTHON=.venv/bin/python (--mode full)
    python reproduce.py --mode eval-only               # = make report PYTHON=.venv/bin/python
    python reproduce.py --mode eval-only --verify-checkpoints [--checkpoint-root DIR]
    python reproduce.py --mode eval-only --dry-run [--all]   # print the plan and the commands; run nothing
"""
import argparse
import sys

from pilot import protocol
from pilot import reproduce
from pilot.data import REPO_ROOT


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=reproduce.MODES, default="full",
                    help="full (default): every todo step in order, then the report; eval-only: rebuild every "
                         "report, the memo and the README blocks from the committed dumps")
    ap.add_argument("--verify-checkpoints", action="store_true",
                    help="also re-hash the on-disk final seeds against their manifests (reported, not gated)")
    ap.add_argument("--checkpoint-root", metavar="DIR", default=None,
                    help="where the final seed dirs live (default: the repo root); only with --verify-checkpoints")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the todo steps' commands, with the D-04 disclosure; run nothing")
    ap.add_argument("--all", action="store_true", help="--dry-run: print every step's commands, done ones included")
    a = ap.parse_args(argv)
    if a.checkpoint_root is not None and not a.verify_checkpoints:
        ap.error("--checkpoint-root applies only with --verify-checkpoints")
    if a.all and not a.dry_run:
        ap.error("--all applies only with --dry-run")
    if a.dry_run and a.verify_checkpoints:
        ap.error("--dry-run runs nothing; drop --verify-checkpoints")
    print("reproduce in " + sys.executable, flush=True)
    if a.dry_run:
        print(reproduce.dry_run(REPO_ROOT, all_steps=a.all), end="", flush=True)
        return 0
    rows = reproduce.plan(REPO_ROOT)
    _, pin = protocol.require_frozen()
    print(reproduce.render_plan(rows, pin), end="", flush=True)
    run = reproduce.run_full if a.mode == "full" else reproduce.eval_only
    run(REPO_ROOT, verify=a.verify_checkpoints, checkpoint_root=a.checkpoint_root or REPO_ROOT)


if __name__ == "__main__":
    main()
