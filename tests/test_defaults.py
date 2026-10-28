"""The tuned defaults of new training runs (pilot/defaults.py) and what they must never move.

- pilot.defaults.LAYA and BERTURK are exactly the `best.config` of the committed search records results/tune/laya.json
  and results/tune/berturk.json, read through pilot.tune.read_record at their pinned sha256s and compared by JSON text,
  so a rounded value or an int written for a float (0 for 0.0) fails; SOURCES names each record, its sha256, the best
  trial and its epoch
- pilot.defaults imports nothing outside the stdlib, no other pilot module, and reads no file
- ray_train.py's --epochs, --encoder-lr, --head-lr and --neg default to pilot.defaults.LAYA, an explicit flag wins
- every committed command of both systems, as pilot.reproduce prints it, keeps its committed run id, rebuilt offline
  through the trainer's own parser, resolution and identity_values with the bindings run on it: the selection runs,
  the final seeds (check_final), the masked bases (check_base) and the continued few-shot runs (check_continued),
  whose pruned masked bases enter by the identity their gated sidecars record (JSON only), the whole rebuilt canonical
  config equal to the continued run's sidecar config; without its recipe flags a command names another run
- README.md, CLAUDE.md and .claude/CLAUDE.md name the tuned defaults and spell out the frozen BERTurk selection recipe

Every test is unit-marked and offline (the committed records, protocol.json, data/splits.json and the pins only).
"""
import ast
import json
import os
import re
import subprocess
import sys
from typing import Any

import pytest

from pilot import defaults
from pilot import tune as pt
from pilot.data import REPO_ROOT

pytestmark = pytest.mark.unit

SYSTEMS = ("laya", "berturk")
CONSTANTS = {"laya": defaults.LAYA, "berturk": defaults.BERTURK}
# the committed records' bytes, pinned (the same sha256s tests/test_tune_results.py pins)
RECORD_SHA256 = {"laya": "b111b6e21db971fa765650c46a9534d530317fce620cea7f9e9f0c61f26feef5",
                 "berturk": "acb143c13bc3b64dfbab5c819a435790e64671c39422654cc2342f2f10d3e70d"}
INT_KEYS = ("neg", "epochs", "batch")
DEFAULTS_PY = os.path.join(REPO_ROOT, "pilot", "defaults.py")


def test_defaults_are_the_best_trials_of_the_committed_search() -> None:
    for s in SYSTEMS:
        src = defaults.SOURCES[s]
        assert src["record"] == "results/tune/%s.json" % s
        rec, sha = pt.read_record(os.path.join(REPO_ROOT, src["record"]))
        assert sha == RECORD_SHA256[s] == src["sha256"]
        best, constant = rec["best"], CONSTANTS[s]
        assert json.dumps(constant, sort_keys=True) == json.dumps(best["config"], sort_keys=True)
        assert list(constant) == list(best["config"])
        assert src["trial"] == best["index"]
        assert src["epoch"] == best["epoch"] == constant["epochs"]
        for key, v in constant.items():
            if key in INT_KEYS:
                assert type(v) is int, (s, key, v)
            else:
                assert type(v) is float, (s, key, v)
    assert repr(defaults.BERTURK["weight_decay"]) == "0.0"
    assert set(defaults.SOURCES) == set(SYSTEMS)


def test_defaults_module_imports_the_stdlib_only() -> None:
    code = ("import sys; before = set(sys.modules); import pilot.defaults; new = set(sys.modules) - before; "
            "bad = sorted(m for m in new if m.split('.')[0] not in sys.stdlib_module_names and m.split('.')[0] != "
            "'pilot') + sorted(m for m in new if m.startswith('pilot.') and m != 'pilot.defaults'); print(bad)")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "[]"
    with open(DEFAULTS_PY, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
            assert name != "open", "pilot/defaults.py reads a file at line %d" % node.lineno
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] in sys.stdlib_module_names, alias.name
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0 and (node.module or "").split(".")[0] in sys.stdlib_module_names, node.module


# ---- Laya: ray_train.py's parser defaults ----

ENTRY = {"path": "splits/selection_train.jsonl", "sha256": "a" * 64, "role": "train"}
BASE_IDENT = {"repo": "r", "subfolder": "s", "revision": "c" * 40, "model_sha256": "b" * 64}
LAYA_DESTS = {"encoder_lr": "encoder_lr", "head_lr": "head_lr", "neg": "neg", "epochs": "epochs"}   # key: argparse dest


def _laya_canon(*argv: str) -> dict[str, Any]:
    """The canonical config of a ray_train.py command on a placeholder split entry and base identity."""
    import ray_train
    from pilot import runs
    a = ray_train.parser().parse_args(list(argv))
    return runs.canonical(ray_train.identity_values(a, ray_train.resolve(a), ENTRY, "multilingual", BASE_IDENT))


def test_ray_train_parser_defaults_are_the_tuned_laya_config() -> None:
    ray_train = pytest.importorskip("ray_train")
    dests = {act.dest: act.default for act in ray_train.parser()._actions}
    got = {key: dests[dest] for key, dest in LAYA_DESTS.items()}
    assert list(got) == list(defaults.LAYA)
    assert json.dumps(got, sort_keys=True) == json.dumps(defaults.LAYA, sort_keys=True)


def test_a_plain_laya_command_names_the_tuned_recipe() -> None:
    pytest.importorskip("ray_train")
    from pilot import runs
    base = ["--data", "d.jsonl", "--out", "o"]
    canon = _laya_canon(*base)
    got = {key: canon[key] for key in defaults.LAYA}
    assert json.dumps(got, sort_keys=True) == json.dumps(defaults.LAYA, sort_keys=True)
    assert runs.run_id(canon).startswith("laya-selection_train-neg16-s0-")
    for flag, value, key, want in (("--epochs", "4", "epochs", 4), ("--encoder-lr", "3e-05", "encoder_lr", 3e-05),
                                   ("--head-lr", "0.0002", "head_lr", 0.0002), ("--neg", "8", "neg", 8)):
        explicit = _laya_canon(*(base + [flag, value]))
        assert explicit[key] == want and type(explicit[key]) is type(want), flag
        assert {k: explicit[k] for k in defaults.LAYA if k != key} == {k: canon[k] for k in defaults.LAYA if k != key}


# ---- Laya: every committed command names its own recipe ----

LAYA_RECIPE_FLAGS = ("--epochs", "--encoder-lr", "--head-lr", "--neg")


def _read_json(*parts: str) -> Any:
    with open(os.path.join(REPO_ROOT, *parts), encoding="utf-8") as f:
        return json.load(f)


def _laya_commands(step: str) -> list[tuple[str, ...]]:
    from pilot import reproduce as rp
    return [c for c in rp.commands(step) if c[0] == "ray_train.py"]


def test_committed_laya_commands_pass_every_tuned_flag() -> None:
    counts = {"laya-selection": 2, "laya-final": 3, "fewshot-bases": 3, "fewshot-continued": 6}
    for step, n in counts.items():
        cmds = _laya_commands(step)
        assert len(cmds) == n, (step, len(cmds))
        for c in cmds:
            for flag in LAYA_RECIPE_FLAGS:
                assert c.count(flag) == 1, (step, flag, c)


def _committed_laya_canon(argv: list[str], ident: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """(canonical config, postfit record or None) of a committed ray_train.py command, offline: the committed
    data/splits.json entry named by the --data file's stem, the pinned base identity, and for a final seed the sha256
    of its postfit record."""
    import ray_train
    from pilot import postfit, runs
    a = ray_train.parser().parse_args(argv)
    entry = _read_json("data", "splits.json")["splits"][os.path.splitext(os.path.basename(a.data))[0]]
    record, sha = (None, None)
    if a.postfit is not None:
        record, sha = postfit.read_record(os.path.join(REPO_ROOT, a.postfit))
    vals = ray_train.identity_values(a, ray_train.resolve(a), entry, "multilingual", ident, sha)
    return runs.canonical(vals), record


def test_committed_laya_run_ids_recompute_offline() -> None:
    pytest.importorskip("ray_train")
    from pilot import postfit, runs
    from pilot.weights import identity
    ident = identity("multilingual")
    chosen = _read_json("results", "postfit", "laya.json")["chosen"]["train_run_id"]
    assert chosen == "laya-selection_train-neg8-s0-d92447f4"
    sel = list(_laya_commands("laya-selection")[1][1:])
    assert "--print-run-id" not in sel
    canon, _ = _committed_laya_canon(sel, ident)
    assert runs.run_id(canon) == chosen
    final_ids = {s["seed"]: s["train_run_id"] for s in _read_json("results", "postfit", "laya-final.json")["seeds"]}
    finals = _laya_commands("laya-final")
    assert len(finals) == 3
    for c in finals:
        argv = list(c[1:])
        canon, record = _committed_laya_canon(argv, ident)
        assert record is not None
        postfit.check_final(record, canon)
        assert runs.run_id(canon) == final_ids[canon["seed"]]
    assert sorted(final_ids) == [0, 1, 2]
    # negative control: without its four recipe flags the selection command names another run, at the tuned neg
    bare = list(sel)
    for flag in LAYA_RECIPE_FLAGS:
        i = bare.index(flag)
        del bare[i:i + 2]
    other = runs.run_id(_committed_laya_canon(bare, ident)[0])
    assert other != chosen and other.startswith("laya-selection_train-neg%d-s0-" % defaults.LAYA["neg"])


# ---- BERTurk and the Laya masked bases: every printable committed command resolves to its committed run id ----

def _training_lines(step: str, script: str) -> list[list[str]]:
    """The argv (script name dropped) of every `script` training line pilot.reproduce prints for `step`."""
    from pilot import reproduce as rp
    return [list(c[1:]) for c in rp.commands(step) if c[0] == script]


def _entry_of(data: str) -> dict[str, Any]:
    """The committed data/splits.json entry named by a --data file's stem."""
    return _read_json("data", "splits.json")["splits"][os.path.splitext(os.path.basename(data))[0]]


def _berturk_canon(argv: list[str]) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """(parsed args, canonical config, context) of a committed berturk_train.py command on the pinned base, built
    exactly as berturk_train.main builds it: the frozen protocol, mask and binding of a masked base, the --postfit
    record and its sha256 of a final seed, then recipe_context and resolve_recipe before identity_values. context holds
    proto, pin, chosen and record for the bindings."""
    import berturk_train
    from pilot import berturk, fewshot, pins, postfit, protocol
    a = berturk_train.parser().parse_args(argv)
    assert a.base == "berturk", argv
    proto = pin = chosen = record = sha = fs_binding = None
    mask: Any = ()
    if a.fewshot_base:
        proto, pin = protocol.require_frozen()
        mask = berturk.check_mask_tags(proto["fewshot"]["tags"])
        fs_binding = fewshot.binding(pin, "base")
        chosen = fewshot.chosen_canonical(proto, "berturk")
    if a.postfit is not None:
        record, sha = postfit.read_record(os.path.join(REPO_ROOT, a.postfit))
    berturk_train.resolve_recipe(a, *berturk_train.recipe_context(a, record, chosen, None))
    canon = berturk.canonical(berturk_train.identity_values(a, _entry_of(a.data), dict(pins.BERTURK["sha256"]), mask,
                                                            sha, fs_binding))
    return a, canon, {"proto": proto, "pin": pin, "chosen": chosen, "record": record}


def _base_entries(kind: str) -> dict[int, dict[str, Any]]:
    """{seed: runs.json entry} of the committed masked bases of `kind`."""
    entries = _read_json("results", "fewshot", "runs.json")["entries"]
    return {e["seed"]: e for e in entries if e["kind"] == kind and e["stage"] == "base"}


def test_committed_berturk_run_ids_resolve_offline() -> None:
    pytest.importorskip("berturk_train")
    from pilot import berturk, fewshot, postfit
    chosen_rid = _read_json("results", "postfit", "berturk.json")["chosen"]["train_run_id"]
    assert chosen_rid == "berturk-selection_train-s0-4f36ae05"
    sel = _training_lines("berturk-selection", "berturk_train.py")
    assert len(sel) == 2 and sel[0] == sel[1] + ["--print-run-id"]
    _, canon, _ = _berturk_canon(sel[1])
    assert berturk.run_id(canon) == chosen_rid
    # negative control: without its three recipe pairs the selection command resolves them to the tuned defaults
    # (its explicit epochs and LRs stay the frozen ones) and names another run
    bare = list(sel[1])
    for flag in ("--batch", "--warmup-ratio", "--weight-decay"):
        i = bare.index(flag)
        del bare[i:i + 2]
    _, plain, _ = _berturk_canon(bare)
    dropped = ("batch", "warmup_ratio", "weight_decay")
    assert json.dumps({k: plain[k] for k in dropped}) == json.dumps({k: defaults.BERTURK[k] for k in dropped})
    assert {k: plain[k] for k in ("epochs", "encoder_lr", "head_lr")} == {k: canon[k] for k in
                                                                          ("epochs", "encoder_lr", "head_lr")}
    assert berturk.run_id(plain) != chosen_rid
    # and the whole command without any recipe flag resolves every value to the tuned defaults
    for flag in ("--epochs", "--lr", "--head-lr"):
        i = bare.index(flag)
        del bare[i:i + 2]
    _, plain, _ = _berturk_canon(bare)
    got = {key: plain[key] for key in defaults.BERTURK}
    assert json.dumps(got, sort_keys=True) == json.dumps(defaults.BERTURK, sort_keys=True)
    finals = _training_lines("berturk-final", "berturk_train.py")
    final_ids = {s["seed"]: s["train_run_id"] for s in _read_json("results", "postfit", "berturk-final.json")["seeds"]}
    assert len(finals) == 3 and sorted(final_ids) == [0, 1, 2]
    for argv in finals:
        a, canon, ctx = _berturk_canon(argv)
        assert ctx["record"] is not None
        postfit.check_final(ctx["record"], canon)
        assert berturk.run_id(canon) == final_ids[a.seed]
    bases, entries = _training_lines("fewshot-bases", "berturk_train.py"), _base_entries("berturk")
    assert len(bases) == 3 and sorted(entries) == [0, 1, 2]
    for argv in bases:
        a, canon, ctx = _berturk_canon(argv)
        fewshot.check_base(ctx["proto"], ctx["pin"], "berturk", canon, ctx["chosen"], fewshot.read_mode(ctx["pin"]))
        assert berturk.run_id(canon) == entries[a.seed]["train_run_id"]


def test_committed_laya_bases_recompute_offline() -> None:
    ray_train = pytest.importorskip("ray_train")
    from pilot import fewshot, protocol, runs
    from pilot.weights import identity
    proto, pin = protocol.require_frozen()
    mode, chosen = fewshot.read_mode(pin), fewshot.chosen_canonical(proto, "laya")
    ident = identity("multilingual")
    bases, entries = _training_lines("fewshot-bases", "ray_train.py"), _base_entries("laya")
    assert len(bases) == 3 and sorted(entries) == [0, 1, 2]
    for argv in bases:
        a = ray_train.parser().parse_args(argv)
        assert a.fewshot_base and a.base == "multilingual", argv
        vals = ray_train.identity_values(a, ray_train.resolve(a), _entry_of(a.data), "multilingual", ident, None,
                                         mask_tags=sorted(proto["fewshot"]["tags"]),
                                         fewshot=fewshot.binding(pin, "base"))
        canon = runs.canonical(vals)
        fewshot.check_base(proto, pin, "laya", canon, chosen, mode)
        assert runs.run_id(canon) == entries[a.seed]["train_run_id"]


# ---- the continued few-shot runs: their bases were pruned, so their gated sidecars hold the identities ----

def _sidecar(dump_run_id: str) -> dict[str, Any]:
    """A committed gated test-dump sidecar (JSON only, never the .npz)."""
    return _read_json("results", "dumps", dump_run_id, "test.json")


def _base_digests(kind: str, ident: dict[str, Any]) -> dict[str, Any]:
    """The base identity a continued run's main hashes from its masked base dir, rebuilt from the base's gated sidecar
    checkpoint_identity: BERTurk's per-file digests (berturk_train.main: sha256_file of each pilot.berturk.BASE_FILES
    entry), Laya's pilot.weights.identity fields (ray_train.identity_values reads model_sha256, config_sha256 and
    files_sha256)."""
    if kind == "berturk":
        return {"model.safetensors": ident["model_sha256"], "config.json": ident["config_sha256"],
                **ident["files_sha256"]}
    return {"model_sha256": ident["model_sha256"], "config_sha256": ident["config_sha256"],
            "files_sha256": ident["files_sha256"]}


def test_committed_continued_runs_rebuild_their_run_ids_from_the_command() -> None:
    """Each of the 12 printed continued few-shot commands is parsed by its trainer, its omitted recipe flags resolved
    as main resolves them (BERTurk: from the masked base's canonical config), and its whole canonical config rebuilt by
    the trainer's own identity_values from the command, the committed split entry, the masked base's identity (from the
    base's gated sidecar, since the base checkpoints were pruned) and the few-shot binding of the frozen draw and the
    base's run id: it equals the continued run's gated sidecar canonical config key for key, its run id is the runs
    manifest's, and check_continued accepts it against the base's canonical config. So a change in how a continued
    command maps any key (seed, draw and k, max_len, micro-batch, neg_seed, the base digests), not only the recipe
    keys, fails here."""
    berturk_train = pytest.importorskip("berturk_train")
    ray_train = pytest.importorskip("ray_train")
    from pilot import berturk, fewshot, protocol, runs
    proto, pin = protocol.require_frozen()
    mode = fewshot.read_mode(pin)
    entries = _read_json("results", "fewshot", "runs.json")["entries"]
    by_out = {e["checkpoint_name"]: e for e in entries if e["stage"] == "continued"}
    bases = {(e["kind"], e["seed"]): e for e in entries if e["stage"] == "base"}
    seen = {"berturk": 0, "laya": 0}
    for script, kind in (("berturk_train.py", "berturk"), ("ray_train.py", "laya")):
        for argv in _training_lines("fewshot-continued", script):
            p = (berturk_train if kind == "berturk" else ray_train).parser()
            a = p.parse_args(argv)
            entry = by_out[a.out]
            assert entry["kind"] == kind and (a.fewshot_draw, a.fewshot_k) == (entry["draw"], entry["k"])
            assert a.base != ("berturk" if kind == "berturk" else "multilingual"), argv   # a local masked base
            base = bases[(kind, entry["base_seed"])]
            assert base["train_run_id"] == entry["base_run_id"]
            base_side, side = _sidecar(base["dump_run_id"]), _sidecar(entry["dump_run_id"])
            if base_side["training"].get("run_id") is not None:
                assert base_side["training"]["run_id"] == base["train_run_id"]
            assert base_side["system"]["checkpoint"] == base["checkpoint_name"]
            base_canon, canon = base_side["training"]["canonical"], side["training"]["canonical"]
            assert (runs.run_id if kind == "laya" else berturk.run_id)(base_canon) == base["train_run_id"]
            digests = _base_digests(kind, base_side["system"]["checkpoint_identity"])
            fsb = fewshot.binding(pin, "continued", a.fewshot_draw, a.fewshot_k,
                                  proto["fewshot"]["draws"][a.fewshot_draw]["digest"], base["train_run_id"])
            if kind == "berturk":
                context = {"run_id": base["train_run_id"], "canonical": base_canon}
                berturk_train.resolve_recipe(a, *berturk_train.recipe_context(a, None, None, context))
                built = berturk.canonical(berturk_train.identity_values(a, _entry_of(a.data), digests, (), None, fsb))
                rid = berturk.run_id(built)
            else:
                vals = ray_train.identity_values(a, ray_train.resolve(a), _entry_of(a.data), a.base, digests, None,
                                                 fewshot=fsb)
                built = runs.canonical(vals)
                rid = runs.run_id(built)
            assert built == canon, (a.out, sorted(k for k in set(built) | set(canon) if built.get(k) != canon.get(k)))
            assert json.dumps(built, sort_keys=True) == json.dumps(canon, sort_keys=True), a.out   # types too
            assert rid == entry["train_run_id"], a.out
            fewshot.check_continued(proto, pin, kind, built, base_canon, base["train_run_id"], entry["draw"],
                                    entry["k"], mode)
            seen[kind] += 1
    assert seen == {"berturk": 6, "laya": 6}


# ---- the docs name the tuned defaults and the frozen recipe of the committed runs ----

README = os.path.join(REPO_ROOT, "README.md")
RECIPE_TEXT = "--batch 32 --warmup-ratio 0.1 --weight-decay 0.01"


def _text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _part(text: str, heading: str, end: str) -> str:
    """The text from `heading` to the next `end` heading after it (exactly one heading)."""
    assert text.count(heading) == 1, heading
    i = text.index(heading)
    return text[i:text.index(end, i + len(heading))]


def _berturk_lines_with_recipe(part: str) -> list[str]:
    """The berturk_train.py command lines (a command and its backslash continuations, joined) holding RECIPE_TEXT."""
    joined = part.replace("\\\n", " ")
    return [l for l in joined.splitlines() if "berturk_train.py" in l and RECIPE_TEXT in " ".join(l.split())]


def test_docs_name_the_tuned_defaults() -> None:
    from pilot import reproduce as rp
    readme = _text(README)
    files = _part(readme, "\n## Files\n", "\n## Run\n")
    assert re.search(r"^\| `pilot/defaults\.py` \| ", files, re.M), "the README Files table has no pilot/defaults.py row"
    assert _berturk_lines_with_recipe(_part(readme, "\n## Run\n", "\n## Scoring path\n"))
    assert _berturk_lines_with_recipe(_part(readme, "\n## BERTurk baseline\n", "\n## Confidence"))
    search = _part(readme, "\n## Exploratory hyperparameter search (Ray Tune)\n", "\n## Demo UI\n")
    for phrase in ("pilot/defaults.py", "new training runs", "frozen recipe"):
        assert phrase in search, phrase
    assert "<!-- " not in search
    assert len(re.findall(r"<!-- [a-z-]+:start -->", readme)) == len(rp.README_BLOCKS) == 11
    claude = _text(os.path.join(REPO_ROOT, "CLAUDE.md"))
    for name in ("pilot/defaults.py", "recipe_context", "resolve_recipe", "results/tune/", RECIPE_TEXT):
        assert name in claude, name
    dot = _text(os.path.join(REPO_ROOT, ".claude", "CLAUDE.md"))
    for name in ("pilot/defaults.py", "results/tune/"):
        assert name in dot, name


def test_docs_say_the_trial_fixed_settings_are_not_defaults() -> None:
    """A plain command trains the tuned keys only: the trials' fixed settings (pilot.tune.FIXED) and device are not
    defaults. The claim's basis is checked against the parsers (Laya's plain --max-len None and --workers 2 against the
    trial's 128 and 1, BERTurk's plain --max-len the trial's 128, both plain devices cpu against the trials' mps), and
    pilot/defaults.py, the ray_train.py docstring and the README search section name pilot.tune.FIXED and the flags a
    run of the Laya trial's recipe passes; the withdrawn "trains them" wording is gone."""
    berturk_train = pytest.importorskip("berturk_train")
    ray_train = pytest.importorskip("ray_train")
    laya = {act.dest: act.default for act in ray_train.parser()._actions}
    bert = {act.dest: act.default for act in berturk_train.parser()._actions}
    fixed = pt.FIXED
    assert fixed["laya"]["max_len"] == 128 and laya["max_len"] is None
    assert fixed["laya"]["workers"] == 1 and laya["workers"] == 2
    assert (fixed["laya"]["micro_batch"], fixed["laya"]["grad_accum"]) == (laya["micro_batch"], laya["grad_accum"])
    assert fixed["berturk"]["max_len"] == bert["max_len"] == 128
    plain = ray_train.parser().parse_args(["--data", "d", "--out", "o"])
    assert laya["device"] is None and ray_train.resolve(plain)["device"] == "cpu"
    assert bert["device"] == "cpu"
    for rec in ("laya", "berturk"):
        assert pt.read_record(os.path.join(REPO_ROOT, defaults.SOURCES[rec]["record"]))[0]["device"] == "mps", rec
    search = _part(_text(README), "\n## Exploratory hyperparameter search (Ray Tune)\n", "\n## Demo UI\n")
    flags = "--device mps --workers 1 --max-len 128"
    for name, text in (("pilot/defaults.py", defaults.__doc__ or ""), ("ray_train.py", ray_train.__doc__ or ""),
                       ("README.md", search)):
        flat = " ".join(text.split())
        assert "pilot.tune.FIXED" in flat and flags in flat, name
    assert "command trains them" not in " ".join(search.split())
