"""Single question source (DATA-03): QUESTIONS is the model input, so it must have one definition and match the data.

QUESTIONS is assigned at module level only in intents.py, pilot/data.py and evaluate.py import it from there, its
shape is {tag: {type: noul, instructions: INTENTS[tag]}} in INTENTS order, and the frozen data/splits.json records the
sha256 of the current QUESTIONS (tag wording changed without regenerating the data fails here).
"""
import ast
import json
import os
import subprocess

import pytest

import pilot.data
from intents import INTENTS, QUESTIONS

pytestmark = pytest.mark.unit

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMPORTERS = ("pilot/data.py", "evaluate.py")


def tracked_py_files():
    """Tracked *.py files via git ls-files (never a directory walk, which would descend into .venv)."""
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    files = [f for f in out.stdout.splitlines() if f]
    assert "intents.py" in files, "git ls-files did not list intents.py: %r" % files
    return files


def _parse(rel):
    with open(os.path.join(REPO_ROOT, rel), encoding="utf-8") as f:
        return ast.parse(f.read(), filename=rel)


def _target_names(target):
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            yield from _target_names(elt)
    elif isinstance(target, ast.Starred):
        yield from _target_names(target.value)


def _module_level_statements(body):
    """Statements executed at module level, including inside if/try/with/for blocks but not def/class bodies."""
    for node in body:
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for field in ("body", "orelse", "finalbody"):
            yield from _module_level_statements(getattr(node, field, []) or [])
        for handler in getattr(node, "handlers", []) or []:
            yield from _module_level_statements(handler.body)


def module_level_assignments(tree, name):
    hits = []
    for node in _module_level_statements(tree.body):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        else:
            continue
        if any(name in _target_names(t) for t in targets):
            hits.append(node.lineno)
    return hits


def test_questions_defined_only_in_intents():
    found = {}
    for rel in tracked_py_files():
        lines = module_level_assignments(_parse(rel), "QUESTIONS")
        if lines:
            found[rel] = lines
    assert found == {"intents.py": found.get("intents.py")} and len(found["intents.py"]) == 1, \
        "QUESTIONS must be assigned at module level exactly once, in intents.py; found %r" % found


@pytest.mark.parametrize("rel", IMPORTERS)
def test_consumers_import_questions_from_intents(rel):
    imported = [node for node in ast.walk(_parse(rel)) if isinstance(node, ast.ImportFrom)
                and node.module == "intents" and any(a.name == "QUESTIONS" for a in node.names)]
    assert imported, "%s must use `from intents import ... QUESTIONS` (single question source, DATA-03)" % rel


def test_questions_shape_matches_intents():
    assert len(INTENTS) == 48
    assert list(QUESTIONS) == list(INTENTS)
    bad = {tag: q for tag, q in QUESTIONS.items() if q != {"type": "noul", "instructions": INTENTS[tag]}}
    assert not bad, "QUESTIONS entries not equal to {type: noul, instructions: INTENTS[tag]}: %r" % bad


def test_frozen_manifest_matches_current_questions():
    with open(pilot.data.DEFAULT_MANIFEST, encoding="utf-8") as f:
        frozen = json.load(f)["questions_sha256"]
    current = pilot.data.questions_sha256()
    assert frozen == current, (
        "data/splits.json questions_sha256 %s != current QUESTIONS %s: tag wording in intents.py changed without "
        "regenerating the data. Run `make data` with --rewrite-manifest (python prepare_data.py --out data "
        "--rewrite-manifest), commit the new data/splits.json, and re-run the baselines (CLAUDE.md rule)."
        % (frozen, current))
