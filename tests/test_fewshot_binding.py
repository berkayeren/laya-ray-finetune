"""Contracts of the few-shot protocol binding: every training command is bound to the frozen protocol by refusal, never
by transcription.

- check_base passes the committed chosen recipe (results/postfit/<kind>.json, read through the protocol-pinned path
  and sha256 by chosen_canonical) with only the base changes (seed and neg_seed s, the frozen stop epoch, the six
  sorted held-out tags, the base binding) and refuses every other difference, naming the key; the first differing
  key in sorted order is named; a Laya base may sit on exactly one fallback ladder configuration and the rung is
  returned.
- check_continued passes a continued command whose canonical config equals its base's except the continued-training
  keys, and refuses another k, another draw's digest, a seed off the draw seed, a base that is not a base of this
  protocol, a base seed other than base_seed_of(block, draw, mode), and any other differing key (the rung carries
  over).
- the recorded mode decides the base pairing: primary pairs base seed d with draw
  d, fallback pairs every draw with the compute fallback's seed 0.
- read_mode reads the compute decision record at call time and refuses a missing record, another schema or pin, an
  unknown mode and a mode that disagrees with its trigger fields; FALLBACK_HOURS is the frozen trigger's 10 h.
Offline: the committed protocol.json and postfit records are tracked files; read_mode runs on the synthetic protocol.
"""
import json
import os
from pathlib import Path
from typing import Any

import pytest

from conftest import SyntheticRaw
from fewshot_synthetic import DECISION_SCHEMA, decision_record, postfit_systems, synthetic_protocol, write_decision
from pilot import fewshot as fs
from pilot import protocol as pr
from pilot import runs
from pilot.data import REPO_ROOT

pytestmark = pytest.mark.unit

TAGS = ("savings", "loan", "overdraft", "appointment", "contactless", "direct_debit")
DROP = object()


def frozen() -> tuple[dict[str, Any], str]:
    return pr.require_frozen()


def edit(canon: dict[str, Any], change: dict[str, Any]) -> dict[str, Any]:
    out = dict(canon)
    for key, value in change.items():
        if value is DROP:
            out.pop(key, None)
        else:
            out[key] = value
    return out


def base_canon(proto: dict[str, Any], pin: str, kind: str, base_seed: int, **change: Any) -> dict[str, Any]:
    """The canonical config of a masked base command: the chosen selection run's, plus the base changes."""
    c = dict(fs.chosen_canonical(proto, kind))
    c.update(seed=base_seed, stop_after_epoch=proto["fewshot"]["rules"]["base_runs"][kind]["stop_after_epoch"],
             mask_tags=sorted(TAGS), fewshot=fs.binding(pin, "base"))
    if kind == "laya":
        c["neg_seed"] = base_seed
    return edit(c, change)


def continued_canon(proto: dict[str, Any], pin: str, kind: str, base: dict[str, Any], draw: int, k: Any,
                    base_run_id: str = "base-run", **change: Any) -> dict[str, Any]:
    """The canonical config of a continued command from `base`: the continued-training keys, the local base
    digests and the continued binding replace the base's."""
    d = proto["fewshot"]["draws"][draw]
    c = {key: v for key, v in base.items() if key not in ("stop_after_epoch", "mask_tags")}
    c.update(epochs=proto["fewshot"]["rules"]["continued_training"]["epochs"], seed=d["seed"],
             base_model_sha256="a" * 64, base_config_sha256="b" * 64,
             fewshot=fs.binding(pin, "continued", draw, k, d["digest"], base_run_id))
    if kind == "laya":
        c.update(neg_seed=d["seed"], base_files_sha256="c" * 64)
    return edit(c, change)


# ---- check_base ----

def test_check_base_accepts_the_chosen_recipe_with_the_base_changes() -> None:
    proto, pin = frozen()
    for kind in ("laya", "berturk"):
        with open(os.path.join(REPO_ROOT, proto["systems"][kind]["postfit"]["path"]), encoding="utf-8") as f:
            assert fs.chosen_canonical(proto, kind) == json.load(f)["chosen"]["canonical"]
        chosen = fs.chosen_canonical(proto, kind)
        for s in (0, 1, 2):
            assert fs.check_base(proto, pin, kind, base_canon(proto, pin, kind, s), chosen, "primary") == "none"


BASE_REFUSALS = [
    ("laya", "epochs", {"epochs": 5}), ("laya", "stop_after_epoch", {"stop_after_epoch": 4}),
    ("laya", "stop_after_epoch", {"stop_after_epoch": DROP}), ("laya", "encoder_lr", {"encoder_lr": 3e-05}),
    ("laya", "head_lr", {"head_lr": 0.0002}), ("laya", "neg", {"neg": 4}), ("laya", "max_len", {"max_len": 64}),
    ("laya", "device", {"device": "cpu"}), ("laya", "workers", {"workers": 2}), ("laya", "loss", {"loss": "soft_ce"}),
    ("laya", "data_sha256", {"data_sha256": "0" * 64}), ("laya", "split", {"split": "final_train"}),
    ("laya", "mask_tags", {"mask_tags": sorted(TAGS)[:5]}), ("laya", "mask_tags", {"mask_tags": list(TAGS)}),
    ("laya", "mask_tags", {"mask_tags": DROP}), ("laya", "seed", {"seed": 3, "neg_seed": 3}),
    ("laya", "seed", {"seed": True, "neg_seed": True}), ("laya", "neg_seed", {"neg_seed": 2}),
    ("laya", "fewshot", {"fewshot": {"protocol_sha256": "f" * 64, "stage": "base"}}),
    ("laya", "fewshot", {"fewshot": DROP}), ("laya", "weight_decay", {"weight_decay": 0.0}),
    ("laya", "postfit_sha256", {"postfit_sha256": "a" * 64}), ("laya", "max_messages", {"max_messages": 100}),
    ("berturk", "batch", {"batch": 16}), ("berturk", "epochs", {"epochs": 10}),
    ("berturk", "stop_after_epoch", {"stop_after_epoch": 28}), ("berturk", "encoder_lr", {"encoder_lr": 1e-05}),
    ("berturk", "optim", {"optim": "adamw_hf"}), ("berturk", "warmup_ratio", {"warmup_ratio": 0.0}),
    ("berturk", "seed", {"seed": 5}), ("berturk", "mask_tags", {"mask_tags": ["loan"]}),
    ("berturk", "device", {"device": "cuda"}),
]


@pytest.mark.parametrize("kind,key,change", BASE_REFUSALS)
def test_check_base_refuses_a_differing_setting(kind: str, key: str, change: Any) -> None:
    proto, pin = frozen()
    canon = base_canon(proto, pin, kind, 1, **change)
    with pytest.raises(SystemExit, match=key) as e:
        fs.check_base(proto, pin, kind, canon, fs.chosen_canonical(proto, kind), "primary")
    assert "\n" not in str(e.value)


def test_check_base_names_the_first_differing_key_in_sorted_order() -> None:
    proto, pin = frozen()
    canon = base_canon(proto, pin, "laya", 0, weight_decay=0.0, clip_grad_norm=2.0, shuffle_options=True)
    with pytest.raises(SystemExit, match="^clip_grad_norm is 2.0 here but 1.0"):
        fs.check_base(proto, pin, "laya", canon, fs.chosen_canonical(proto, "laya"), "primary")


def test_check_base_compares_json_values_exactly() -> None:
    proto, pin = frozen()
    canon = base_canon(proto, pin, "laya", 0, max_len=128.0)              # 128.0 is not the frozen 128
    with pytest.raises(SystemExit, match="max_len"):
        fs.check_base(proto, pin, "laya", canon, fs.chosen_canonical(proto, "laya"), "primary")


LADDER = [({"freeze_embeddings": True}, "embeddings_frozen"),
          ({"freeze_embeddings": True, "micro_batch": 4, "grad_accum": 8}, "embeddings_frozen+smaller_batch"),
          ({"freeze_encoder": True}, "encoder_frozen")]


@pytest.mark.parametrize("change,rung", LADDER)
def test_check_base_accepts_exactly_one_ladder_configuration(change: Any, rung: str) -> None:
    proto, pin = frozen()
    canon = base_canon(proto, pin, "laya", 2, **change)
    assert fs.check_base(proto, pin, "laya", canon, fs.chosen_canonical(proto, "laya"), "primary") == rung


@pytest.mark.parametrize("change", [{"micro_batch": 4, "grad_accum": 8}, {"freeze_embeddings": True,
                                                                         "freeze_encoder": True},
                                    {"micro_batch": 4}, {"freeze_encoder": True, "micro_batch": 4, "grad_accum": 8},
                                    {"grad_accum": 8}])
def test_check_base_refuses_a_configuration_off_the_ladder(change: Any) -> None:
    proto, pin = frozen()
    canon = base_canon(proto, pin, "laya", 0, **change)
    with pytest.raises(SystemExit, match="fallback ladder"):
        fs.check_base(proto, pin, "laya", canon, fs.chosen_canonical(proto, "laya"), "primary")


def test_ladder_configs_are_the_laya03_rungs() -> None:
    proto, _ = frozen()
    rule = proto["fewshot"]["rules"]["base_runs"]["laya"]
    cfg = fs.ladder_configs(rule)
    assert set(cfg) <= set(runs.RUNG_NEXT)
    assert cfg == {"none": (False, False, 8, 4), "embeddings_frozen": (True, False, 8, 4),
                   "embeddings_frozen+smaller_batch": (True, False, 4, 8), "encoder_frozen": (False, True, 8, 4)}
    for label, (emb, enc, mb, ga) in cfg.items():
        assert runs.rung_label({"freeze_embeddings": emb, "freeze_encoder": enc, "micro_batch": mb}) == label


def test_check_base_refuses_an_unknown_kind_or_mode() -> None:
    proto, pin = frozen()
    canon = base_canon(proto, pin, "laya", 0)
    with pytest.raises(SystemExit, match="kind"):
        fs.check_base(proto, pin, "bert", canon, canon, "primary")
    with pytest.raises(SystemExit, match="mode"):
        fs.check_base(proto, pin, "laya", canon, fs.chosen_canonical(proto, "laya"), "both")


# ---- check_continued ----

@pytest.mark.parametrize("kind", ["laya", "berturk"])
def test_check_continued_accepts_the_bound_continued_run(kind: str) -> None:
    proto, pin = frozen()
    for draw in range(3):
        base = base_canon(proto, pin, kind, draw)
        for k in (5, 10):
            canon = continued_canon(proto, pin, kind, base, draw, k)
            assert fs.check_continued(proto, pin, kind, canon, base, "base-run", draw, k, "primary") is None


def test_check_continued_carries_the_base_rung_over() -> None:
    proto, pin = frozen()
    base = base_canon(proto, pin, "laya", 1, freeze_embeddings=True)
    canon = continued_canon(proto, pin, "laya", base, 1, 10)
    assert fs.check_continued(proto, pin, "laya", canon, base, "base-run", 1, 10, "primary") is None
    with pytest.raises(SystemExit, match="freeze_embeddings"):
        fs.check_continued(proto, pin, "laya", dict(canon, freeze_embeddings=False), base, "base-run", 1, 10,
                           "primary")


def _continued_refusals(proto: dict[str, Any], pin: str) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    d0, d1 = proto["fewshot"]["draws"][0], proto["fewshot"]["draws"][1]
    other = fs.binding(pin, "continued", 0, 10, d1["digest"], "base-run")
    return [
        ("digest|fewshot", {"fewshot": other}, {}),
        ("fewshot", {"fewshot": fs.binding(pin, "continued", 0, 5, d0["digest"], "base-run")}, {}),
        ("fewshot", {"fewshot": fs.binding(pin, "continued", 0, 10, d0["digest"], "another-base")}, {}),
        ("seed", {"seed": 1, "neg_seed": 1}, {}),
        ("neg_seed", {"neg_seed": 2}, {}),
        ("epochs", {"epochs": 9}, {}),
        ("encoder_lr", {"encoder_lr": 1e-05}, {}),
        ("head_lr", {"head_lr": 0.001}, {}),
        ("stop_after_epoch", {"stop_after_epoch": 3}, {}),
        ("mask_tags", {"mask_tags": sorted(TAGS)}, {}),
        ("postfit_sha256", {"postfit_sha256": "a" * 64}, {}),
        ("max_len", {"max_len": 64}, {}),
        ("data_sha256", {"data_sha256": "0" * 64}, {}),
        ("base", {}, {"fewshot": fs.binding(pin, "continued", 0, 10, d0["digest"], "x")}),
        ("base", {}, {"fewshot": {"protocol_sha256": "f" * 64, "stage": "base"}}),
        ("base", {}, {"seed": 1, "neg_seed": 1}),
        ("base", {}, {"mask_tags": ["loan"]}),
    ]


@pytest.mark.parametrize("i", range(17))
def test_check_continued_refuses_a_differing_command(i: int) -> None:
    proto, pin = frozen()
    match, change, base_change = _continued_refusals(proto, pin)[i]
    base = base_canon(proto, pin, "laya", 0)
    canon = continued_canon(proto, pin, "laya", base, 0, 10, **change)
    with pytest.raises(SystemExit, match=match) as e:
        fs.check_continued(proto, pin, "laya", canon, edit(base, base_change), "base-run", 0, 10, "primary")
    assert "\n" not in str(e.value)


@pytest.mark.parametrize("draw,k", [(0, 3), (3, 5), (-1, 10)])
def test_check_continued_refuses_a_k_or_draw_off_the_protocol(draw: Any, k: Any) -> None:
    proto, pin = frozen()
    base = base_canon(proto, pin, "laya", 0)
    canon = continued_canon(proto, pin, "laya", base, 0, 10)
    with pytest.raises(SystemExit, match="draw|k "):
        fs.check_continued(proto, pin, "laya", canon, base, "base-run", draw, k, "primary")


def test_base_seed_follows_the_recorded_mode() -> None:
    """The assumption-delta invariant: for both modes every continued binding check accepts exactly the base
    base_seed_of names and refuses every other base seed."""
    proto, pin = frozen()
    block = proto["fewshot"]
    assert [fs.base_seed_of(block, d, "primary") for d in range(3)] == [0, 1, 2]
    assert [fs.base_seed_of(block, d, "fallback") for d in range(3)] == [0, 0, 0]
    assert fs.allowed_base_seeds(block, "primary") == [0, 1, 2] and fs.allowed_base_seeds(block, "fallback") == [0]
    for mode in fs.MODES:
        for draw in range(3):
            for bs in (0, 1, 2):
                base = base_canon(proto, pin, "berturk", bs)
                canon = continued_canon(proto, pin, "berturk", base, draw, 10)
                if bs == fs.base_seed_of(block, draw, mode):
                    assert fs.check_continued(proto, pin, "berturk", canon, base, "base-run", draw, 10, mode) is None
                else:
                    with pytest.raises(SystemExit, match="base"):
                        fs.check_continued(proto, pin, "berturk", canon, base, "base-run", draw, 10, mode)
    chosen = fs.chosen_canonical(proto, "laya")
    assert fs.check_base(proto, pin, "laya", base_canon(proto, pin, "laya", 0), chosen, "fallback") == "none"
    for bs in (1, 2):
        with pytest.raises(SystemExit, match="seed"):
            fs.check_base(proto, pin, "laya", base_canon(proto, pin, "laya", bs), chosen, "fallback")


# ---- binding, chosen_canonical ----

def test_binding_shapes_and_refusals() -> None:
    assert fs.binding("p", "base") == {"protocol_sha256": "p", "stage": "base"}
    assert fs.binding("p", "continued", 1, 5, "d", "r") == {"protocol_sha256": "p", "stage": "continued", "draw": 1,
                                                            "k": 5, "digest": "d", "base_run_id": "r"}
    assert fs.STAGES == ("base", "continued") and fs.MODES == ("primary", "fallback")
    with pytest.raises(SystemExit, match="stage"):
        fs.binding("p", "final")
    with pytest.raises(SystemExit, match="base_run_id"):
        fs.binding("p", "continued", 1, 5, "d")
    with pytest.raises(SystemExit, match="base"):
        fs.binding("p", "base", draw=1)


def test_chosen_canonical_refuses_a_record_off_its_pin(tmp_path: Path, synthetic_raw: SyntheticRaw,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    systems = postfit_systems(tmp_path, {"laya": {"seed": 0, "epochs": 4}})
    ns = synthetic_protocol(tmp_path, synthetic_raw, monkeypatch, systems=systems)
    assert fs.chosen_canonical(ns.protocol, "laya") == {"seed": 0, "epochs": 4}
    path = systems["laya"]["postfit"]["path"]
    with open(path, "ab") as f:
        f.write(b" ")
    with pytest.raises(SystemExit, match="sha256"):
        fs.chosen_canonical(ns.protocol, "laya")
    with pytest.raises(SystemExit, match="berturk"):
        fs.chosen_canonical(ns.protocol, "berturk")
    os.remove(path)
    with pytest.raises(SystemExit, match="not found"):
        fs.chosen_canonical(ns.protocol, "laya")


# ---- read_mode and the frozen trigger ----

def test_fallback_hours_is_the_frozen_trigger() -> None:
    proto, _ = frozen()
    trigger = proto["fewshot"]["rules"]["compute_fallback"]["trigger"]
    assert fs.FALLBACK_HOURS == 10 and "exceeds %d h" % fs.FALLBACK_HOURS in trigger
    assert fs.DECISION_SCHEMA == DECISION_SCHEMA
    assert fs.DECISION_PATH == os.path.join(REPO_ROOT, "results", "fewshot", "projection.json")


@pytest.mark.parametrize("mode,ladder", [("primary", False), ("fallback", False), ("fallback", True)])
def test_read_mode_returns_the_decision_mode(tmp_path: Path, synthetic_raw: SyntheticRaw,
                                             monkeypatch: pytest.MonkeyPatch, mode: str, ladder: bool) -> None:
    ns = synthetic_protocol(tmp_path, synthetic_raw, monkeypatch, mode=mode)
    if ladder:
        write_decision(ns.decision_path, decision_record(ns.pin, mode, projection_fired=False, ladder_failure=True))
    assert fs.read_mode(ns.pin) == mode
    assert fs.read_mode(ns.pin, ns.decision_path) == mode


def test_read_mode_refuses_a_missing_record(tmp_path: Path, synthetic_raw: SyntheticRaw,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    ns = synthetic_protocol(tmp_path, synthetic_raw, monkeypatch)
    os.remove(ns.decision_path)
    with pytest.raises(SystemExit, match="fewshot.py --projection"):
        fs.read_mode(ns.pin)


@pytest.mark.parametrize("change,match", [
    ({"schema": "fewshot-projection/2"}, "schema"),
    ({"protocol_sha256": "f" * 64}, "protocol"),
    ({"decision": {"mode": "both"}}, "mode"),
    ({"decision": {"mode": "fallback"}}, "trigger"),
    ({"trigger": {"projection_fired": True, "ladder_failure": False}}, "trigger"),
    ({"trigger": {"projection_fired": "no", "ladder_failure": False}}, "trigger"),
    ({"trigger": None}, "trigger"),
])
def test_read_mode_refuses_a_record_that_does_not_bind(tmp_path: Path, synthetic_raw: SyntheticRaw,
                                                       monkeypatch: pytest.MonkeyPatch, change: Any,
                                                       match: str) -> None:
    ns = synthetic_protocol(tmp_path, synthetic_raw, monkeypatch)
    write_decision(ns.decision_path, dict(decision_record(ns.pin, "primary"), **change))
    with pytest.raises(SystemExit, match=match):
        fs.read_mode(ns.pin)


def test_read_mode_reads_the_path_at_call_time(tmp_path: Path, synthetic_raw: SyntheticRaw,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    ns = synthetic_protocol(tmp_path, synthetic_raw, monkeypatch)
    other = write_decision(str(tmp_path / "other" / "projection.json"), decision_record(ns.pin, "fallback"))
    monkeypatch.setattr(fs, "DECISION_PATH", other)
    assert fs.read_mode(ns.pin) == "fallback"
