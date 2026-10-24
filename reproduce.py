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
re-scoring is met by the committed dumps.

usage:
    python reproduce.py --mode eval-only               # = make report PYTHON=.venv/bin/python
"""
import argparse
import sys

from pilot import protocol
from pilot import reproduce
from pilot.data import REPO_ROOT


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=reproduce.MODES, required=True,
                    help="eval-only: rebuild every report, the memo and the README blocks from the committed dumps")
    a = ap.parse_args(argv)
    print("reproduce in " + sys.executable, flush=True)
    rows = reproduce.plan(REPO_ROOT)
    _, pin = protocol.require_frozen()
    print(reproduce.render_plan(rows, pin), end="", flush=True)
    reproduce.eval_only(REPO_ROOT)


if __name__ == "__main__":
    main()
