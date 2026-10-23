"""Contracts of pilot/protocol.py (REP-01, ROADMAP Phase 8 criterion 1): the frozen decision protocol as code.

- protocol_bytes: the identity of a protocol is the sha256 of its exact bytes (indent 2, ensure_ascii False, one
  trailing newline, UTF-8); require_frozen hashes the bytes it parses and refuses a missing file, an unset pin, a
  wrong pin, a re-serialised but equal file, non-JSON, a non-object, another schema; it reads PROTOCOL_PATH and
  PROTOCOL_SHA256 at call time, so a consumer never captures a stale pin.
- write_protocol refuses an existing path (the protocol is frozen once).
- gated_split names test or hotels_probe only for an entry whose sha256 and uids_sha256 are the protocol's;
  final_seed_allowed is true for exactly the three dump run ids of a system's final manifest.
- pilot.protocol imports the stdlib only.
- data-marked: build_protocol on the committed inputs (deterministic bytes, the CLI).
- render_markdown(protocol, sha256), the README protocol block (08-06): every value read from the dict passed in,
  deterministic, display rounding only (2 decimals for points, 4 for metrics); `--render PATH` previews a file with its
  own sha256, `--render` alone renders require_frozen()'s protocol and refuses while unpinned.
Offline except the `data` tests.
"""
import copy
import json
import os
import subprocess
import sys

import pytest

import pilot.data
from pilot import protocol as pr

pytestmark = pytest.mark.unit

HEAVY = ("torch", "numpy", "scipy", "sklearn", "laya", "ray", "transformers")
CODE = {"git_head": "test", "dirty": False}
LAYA_FINAL = os.path.join(pilot.data.REPO_ROOT, "results", "postfit", "laya-final.json")
BERTURK_FINAL = os.path.join(pilot.data.REPO_ROOT, "results", "postfit", "berturk-final.json")
LAYA_CHOSEN = "laya-047381259d3b-cf0f7846-cc216d59"
BERTURK_CHOSEN = "berturk-f24d8fe3c899-50e28523-29446b61"


# every top-level key of protocol/1, in build order (written here, independently of pilot.protocol.REQUIRED_KEYS)
KEYS = ("schema", "title", "units", "frozen", "inputs", "splits", "systems", "references", "scoring", "variant",
        "seed_averaging", "bootstrap", "criteria", "collapse", "verdict", "definitions", "fewshot", "known_at_freeze",
        "power_context", "assumptions", "delegation", "code")


def small():
    """A hand-written protocol dict with every top-level key: enough for the byte identity and the reader."""
    p = {k: {} for k in KEYS}
    p.update({"schema": pr.SCHEMA, "title": "a hand-written protocol", "units": "fractions; ünits"})
    return p


def frozen(tmp_path, p=None):
    p = small() if p is None else p
    path = str(tmp_path / "protocol.json")
    pr.write_protocol(p, path)
    return path, pilot.data.sha256_bytes(pr.protocol_bytes(p))


def committed_entries():
    with open(pilot.data.DEFAULT_MANIFEST, encoding="utf-8") as f:
        return json.load(f)["splits"]


def final_ids(path):
    with open(path, encoding="utf-8") as f:
        return [s["dump_run_id"] for s in json.load(f)["seeds"]]


@pytest.fixture(scope="module")
def gate():
    """The two data-free blocks the gate helpers read (no split file needed)."""
    return {"splits": pr.splits_block(pilot.data.DEFAULT_MANIFEST), "systems": pr.systems_block(pilot.data.REPO_ROOT)}


# ---- bytes and the pinned reader ----

def test_protocol_bytes_are_indent_2_utf8_with_one_newline():
    p = small()
    raw = pr.protocol_bytes(p)
    assert raw == (json.dumps(p, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    assert "ünits".encode("utf-8") in raw and raw.endswith(b"}\n") and not raw.endswith(b"\n\n")


def test_require_frozen_returns_the_parsed_bytes_and_their_sha(tmp_path):
    path, sha = frozen(tmp_path)
    p, got = pr.require_frozen(path, sha)
    assert p == small() and got == sha
    with open(path, "rb") as f:
        assert pilot.data.sha256_bytes(f.read()) == sha


def test_require_frozen_refuses_a_missing_file(tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        pr.require_frozen(str(tmp_path / "protocol.json"), "0" * 64)


def test_require_frozen_refuses_an_unset_pin(tmp_path):
    path, _ = frozen(tmp_path)
    with pytest.raises(SystemExit, match="not frozen"):
        pr.require_frozen(path, None)


def test_require_frozen_refuses_a_wrong_pin_naming_both_hashes(tmp_path):
    path, sha = frozen(tmp_path)
    wrong = "f" * 64
    with pytest.raises(SystemExit) as e:
        pr.require_frozen(path, wrong)
    assert sha in str(e.value) and wrong in str(e.value)


def test_require_frozen_refuses_a_reserialised_equal_file(tmp_path):
    path, sha = frozen(tmp_path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(small(), ensure_ascii=False, indent=4) + "\n")
    with open(path, encoding="utf-8") as f:
        assert json.load(f) == small()                      # the same dict, other bytes
    with pytest.raises(SystemExit, match=sha):
        pr.require_frozen(path, sha)


@pytest.mark.parametrize("raw, what", [(b"{not json", "not valid JSON"), (b"[1, 2]\n", "not a JSON object"),
                                       (b"\xff\xfe\n", "not valid JSON")])
def test_require_frozen_refuses_non_json_or_non_object_bytes(tmp_path, raw, what):
    path = str(tmp_path / "protocol.json")
    with open(path, "wb") as f:
        f.write(raw)
    with pytest.raises(SystemExit, match=what):
        pr.require_frozen(path, pilot.data.sha256_bytes(raw))


def test_require_frozen_refuses_another_schema(tmp_path):
    p = dict(small(), schema="protocol/0")
    path, sha = frozen(tmp_path, p)
    with pytest.raises(SystemExit, match="protocol/0"):
        pr.require_frozen(path, sha)


def test_require_frozen_reads_the_module_pin_at_call_time(tmp_path, monkeypatch):
    path, sha = frozen(tmp_path)
    monkeypatch.setattr(pr, "PROTOCOL_PATH", path)
    monkeypatch.setattr(pr, "PROTOCOL_SHA256", None)
    with pytest.raises(SystemExit, match="not frozen"):
        pr.require_frozen()
    monkeypatch.setattr(pr, "PROTOCOL_SHA256", sha)
    assert pr.require_frozen() == (small(), sha)


def test_the_repo_protocol_is_not_frozen_by_this_plan():
    assert pr.PROTOCOL_SHA256 is None
    assert pr.PROTOCOL_PATH == os.path.join(pilot.data.REPO_ROOT, "protocol.json")


def test_write_protocol_refuses_an_existing_path_and_keeps_it(tmp_path):
    path, _ = frozen(tmp_path)
    with open(path, "rb") as f:
        before = f.read()
    with pytest.raises(SystemExit, match="exists"):
        pr.write_protocol(dict(small(), title="another"), path)
    with open(path, "rb") as f:
        assert f.read() == before
    assert sorted(os.listdir(tmp_path)) == ["protocol.json"]  # no stray temp file


# ---- the gate helpers on the committed entries ----

def test_gated_split_names_the_committed_test_and_hotels_entries(gate):
    e = committed_entries()
    assert pr.gated_split(gate, e["test"]) == "test"
    assert pr.gated_split(gate, e["hotels_probe"]) == "hotels_probe"
    for name in ("val", "selection_train", "final_train"):
        assert pr.gated_split(gate, e[name]) is None


def test_gated_split_needs_both_hashes(gate):
    e = committed_entries()
    other = dict(e["test"], uids_sha256=e["val"]["uids_sha256"])
    assert pr.gated_split(gate, other) is None
    other = dict(e["hotels_probe"], sha256=e["val"]["sha256"])
    assert pr.gated_split(gate, other) is None
    assert pr.gated_split(gate, {}) is None


def test_final_seed_allowed_is_exactly_the_three_final_dump_run_ids(gate):
    laya, berturk = final_ids(LAYA_FINAL), final_ids(BERTURK_FINAL)
    assert len(laya) == len(set(laya)) == 3 and len(berturk) == len(set(berturk)) == 3
    for rid in laya:
        assert pr.final_seed_allowed(gate, "laya", rid)
        assert not pr.final_seed_allowed(gate, "berturk", rid)
    for rid in berturk:
        assert pr.final_seed_allowed(gate, "berturk", rid)
        assert not pr.final_seed_allowed(gate, "laya", rid)
    for rid in (LAYA_CHOSEN, BERTURK_CHOSEN, "laya-zeroshot-7b928d82", "prior", "majority", ""):
        assert not pr.final_seed_allowed(gate, "laya", rid) and not pr.final_seed_allowed(gate, "berturk", rid)
    assert not pr.final_seed_allowed(gate, "majority", laya[0])
    assert [s["dump_run_id"] for s in pr.final_seeds(gate, "laya")] == laya


def test_systems_block_binds_the_final_manifests_and_records(gate):
    for kind, chosen in (("laya", LAYA_CHOSEN), ("berturk", BERTURK_CHOSEN)):
        s = gate["systems"][kind]
        assert s["kind"] == kind and s["seeds"] == [0, 1, 2]
        assert s["final_manifest"] == {"path": "results/postfit/%s-final.json" % kind,
                                       "sha256": pilot.data.sha256_file(os.path.join(
                                           pilot.data.REPO_ROOT, "results", "postfit", "%s-final.json" % kind))}
        assert s["postfit"]["path"] == "results/postfit/%s.json" % kind
        assert s["chosen"]["dump_run_id"] == chosen
        assert s["dump_run_ids"] == [x["dump_run_id"] for x in s["final_seeds"]]
        assert [x["seed"] for x in s["final_seeds"]] == [0, 1, 2]


def test_systems_block_refuses_a_final_manifest_bound_to_another_record(tmp_path):
    root = tmp_path / "repo"
    (root / "results" / "postfit").mkdir(parents=True)
    for name in ("laya.json", "laya-final.json", "berturk.json", "berturk-final.json"):
        with open(os.path.join(pilot.data.REPO_ROOT, "results", "postfit", name), "rb") as f:
            raw = f.read()
        with open(root / "results" / "postfit" / name, "wb") as f:
            f.write(raw)
    with open(root / "results" / "postfit" / "laya.json", "ab") as f:
        f.write(b" ")                                       # another sha256 than the manifest's binding
    with pytest.raises(SystemExit, match="laya-final.json"):
        pr.systems_block(str(root))


def test_splits_block_holds_the_committed_entries():
    e = committed_entries()
    s = pr.splits_block(pilot.data.DEFAULT_MANIFEST)
    assert list(s) == ["test", "hotels_probe", "val", "selection_train", "final_train"]
    for name, b in s.items():
        assert b["path"] == "data/" + e[name]["path"]
        for key in ("role", "sha256", "uids_sha256", "n_messages", "n_no_intent"):
            assert b[key] == e[name][key]
    with open(pilot.data.DEFAULT_MANIFEST, encoding="utf-8") as f:
        assert s["hotels_probe"]["ood_scoring_tags"] == json.load(f)["ood_scoring_tags"]


def test_splits_block_refuses_another_manifest(tmp_path):
    with open(pilot.data.DEFAULT_MANIFEST, encoding="utf-8") as f:
        m = json.load(f)
    path = tmp_path / "splits.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=4)
    with pytest.raises(SystemExit, match="committed"):
        pr.splits_block(str(path))


def test_protocol_module_is_stdlib_only_at_import():
    code = "import sys, pilot.protocol; print(sorted(m for m in %r if m in sys.modules))" % (HEAVY,)
    out = subprocess.run([sys.executable, "-c", code], cwd=pilot.data.REPO_ROOT, capture_output=True, text=True,
                         check=True).stdout.strip()
    assert out == "[]"


def test_cli_check_refuses_while_unpinned(capsys):
    with pytest.raises(SystemExit, match="not frozen|not found"):
        pr.main(["--check"])


def test_cli_build_needs_out():
    with pytest.raises(SystemExit):
        pr.main(["--build"])


# ---- data: the build on the committed inputs ----

@pytest.mark.data
def test_build_protocol_is_deterministic_and_round_trips(tmp_path):
    a, b = pr.build_protocol(code=CODE), pr.build_protocol(code=copy.deepcopy(CODE))
    assert pr.protocol_bytes(a) == pr.protocol_bytes(b)
    assert a["schema"] == pr.SCHEMA and a["code"] == CODE
    path, sha = frozen(tmp_path, a)
    assert pr.require_frozen(path, sha) == (a, sha)
    assert a["scoring"] == pr.SCORING
    assert a["bootstrap"]["B"] == 10000 and a["bootstrap"]["seed"] == 0 and a["bootstrap"]["level"] == 0.95
    for rel, sha_in in a["inputs"].items():
        assert not os.path.isabs(rel) and pilot.data.sha256_file(os.path.join(pilot.data.REPO_ROOT, rel)) == sha_in
    for kind, chosen in (("laya", LAYA_CHOSEN), ("berturk", BERTURK_CHOSEN)):
        assert "results/dumps/%s/val.json" % chosen in a["inputs"]
    assert pr.gated_split(a, committed_entries()["test"]) == "test"


@pytest.mark.data
def test_cli_build_writes_the_protocol_and_refuses_an_existing_out(tmp_path, capsys):
    out = str(tmp_path / "p.json")
    pr.main(["--build", "--out", out])
    sha = capsys.readouterr().out.strip().split()[-1]
    with open(out, "rb") as f:
        assert pilot.data.sha256_bytes(f.read()) == sha and len(sha) == 64
    with pytest.raises(SystemExit, match="exists"):
        pr.main(["--build", "--out", out])


# ---- Task 2: the decision blocks ----

def test_required_keys_are_every_top_level_key_in_build_order():
    assert pr.REQUIRED_KEYS == KEYS


@pytest.mark.parametrize("key", ["criteria", "collapse", "verdict", "fewshot", "known_at_freeze", "code"])
def test_require_frozen_refuses_a_protocol_missing_a_required_key(tmp_path, key):
    p = {k: x for k, x in small().items() if k != key}
    path, sha = frozen(tmp_path, p)
    with pytest.raises(SystemExit, match=key):
        pr.require_frozen(path, sha)


def test_bootstrap_coverage_is_criterion_d_abstention_coverage():
    from pilot import verdict
    path = [x for x in verdict.DEFAULT_CRITERIA["d"]["paths"] if x["path"] == "abstention"][0]
    assert pr.bootstrap_block()["coverage"] == path["coverage"] == 80


def test_power_context_is_a_labelled_research_estimate():
    pc = pr.POWER_CONTEXT
    assert "research estimate, normal approximation (A19)" in pc["label"]
    assert pc["n"] == 413 and pc["micro_f1_delta_se"] == 0.0098
    assert pc["p_a_lower_at_least_margin"] == {"0": 0.53, "-0.005": 0.34, "-0.01": 0.17, "-0.0155": 0.07}
    assert pc["margin_80_power"] == -0.0274
    assert pc["alternative_go_success"] == {"0.03": 0.5, "0.04": 0.85, "0.05": 0.98}
    assert pc["macro_f1_ci_half_width"] == 0.035 and pc["precision_at_80_ci_half_width"] == 0.03
    assert "the -0.02 margin is locked (REP-01) and kept" in pc["margin"]


def test_assumptions_cover_a1_to_a19_and_the_three_open_questions():
    ids = [a["id"] for a in pr.ASSUMPTIONS]
    assert ids == ["A%d" % i for i in range(1, 20)] + ["OQ1", "OQ2", "OQ3"]
    for a in pr.ASSUMPTIONS:
        assert set(a) == {"id", "decision", "taken"} and "\n" not in a["decision"] and a["decision"]
        assert a["taken"] == "taken as 08-RESEARCH recommended under the standing delegation"
    by = {a["id"]: a["decision"] for a in pr.ASSUMPTIONS}
    assert "Brier" in by["A1"] and "520/2163" in by["A6"] and "ddof 1" in by["A16"]


def test_delegation_quotes_are_verbatim():
    d = json.dumps(pr.DELEGATION, ensure_ascii=False)
    for quote in ("do the programatical control. i will not do manual control until all phases done.",
                  "i already told you that i will not made a manual control untill all phases are done",
                  "assume it okey for now. defer for further learning rate grid search. we know that at least one "
                  "hyperparameter works.",
                  "Single LR now (Recommended)",
                  "the bank-name check is deferred to the end-of-milestone review"):
        assert quote in d, quote


@pytest.fixture(scope="module")
def built():
    return pr.build_protocol(code=CODE)


@pytest.mark.data
def test_build_has_every_required_key_in_order(built):
    assert tuple(built) == KEYS


@pytest.mark.data
def test_criteria_match_verdict_defaults(built):
    from pilot import verdict
    assert built["criteria"] == verdict.DEFAULT_CRITERIA
    assert built["criteria"] is not verdict.DEFAULT_CRITERIA
    vb = built["verdict"]
    assert vb["rules"] == list(verdict.VERDICT_RULES)
    assert vb["inconclusive_means"] == verdict.INCONCLUSIVE_MEANS and vb["provisional"] == verdict.PROVISIONAL_TEXT
    assert vb["statuses"]["a"] == list(verdict.A_STATUSES) and vb["statuses"]["d"] == list(verdict.D_STATUSES)
    assert vb["statuses"]["overall"] == [verdict.GO, verdict.NO_GO, verdict.INCONCLUSIVE]


def _definitions():
    from pilot import fewshot, metrics
    return {"brier": metrics.CALIBRATION_RULES["brier"], "nll": metrics.CALIBRATION_RULES["nll"],
            "classwise_ece": metrics.CALIBRATION_RULES["classwise_ece"], "bins": metrics.CALIBRATION_RULES["bins"],
            "confidence": metrics.CONFIDENCE_RULE, "coverage": metrics.COVERAGE_RULE, "aurc": metrics.AURC_RULE,
            "theta_tie": metrics.TIE_RULE, "bootstrap_method": metrics.BOOT_METHOD,
            "bootstrap_extra": metrics.BOOT_EXTRA_RULE, "false_positive": metrics.FP_RULE,
            "average_precision": metrics.AP_RULE, "fewshot_sampler": fewshot.SAMPLER_RULE,
            "fewshot_digest": fewshot.DIGEST_RULE}


@pytest.mark.data
@pytest.mark.parametrize("key", sorted(_definitions()))
def test_definitions_match_code_constants(built, key):
    assert set(built["definitions"]) == set(_definitions())
    assert built["definitions"][key] == _definitions()[key]


@pytest.mark.data
def test_collapse_holds_the_rule_and_the_prior_fractions(built):
    from pilot import verdict
    c = built["collapse"]
    assert {k: x for k, x in c.items() if not k.startswith("prior_")} == verdict.DEFAULT_COLLAPSE
    assert c["prior_theta_global"] == 0.12
    assert (c["prior_val_fraction"], c["prior_test_fraction"]) == ("10/41", "520/2163")
    assert c["prior_val_micro_f1"] == 10 / 41 and c["prior_test_micro_f1"] == 520 / 2163
    assert c["prior_counts"]["val"] == {"tp": 135, "fp": 489, "fn": 348}
    assert verdict.parse_fraction(c["prior_val_fraction"]) == verdict.micro_fraction(135, 489, 348)


@pytest.mark.data
def test_known_at_freeze_holds_the_in_sample_val_deltas(built):
    k = built["known_at_freeze"]
    assert (k["split"], k["in_sample"], k["n"]) == ("val", True, 208)
    assert (k["laya"], k["berturk"]) == (LAYA_CHOSEN, BERTURK_CHOSEN)
    for variant in ("headline", "raw"):
        m = k[variant]["metrics"]
        assert list(m) == ["micro_f1", "macro_f1", "exact_match", "top1_hit", "brier", "classwise_ece",
                           "precision_at_80"]
        assert k[variant]["B"] == 10000 and k[variant]["seed"] == 0 and k[variant]["n"] == 208
        for name, x in m.items():
            assert x["ci"][0] <= x["ci"][1], (variant, name)
    h = k["headline"]["metrics"]
    assert abs(h["micro_f1"]["delta"] + 0.0155) < 5e-4
    assert abs(h["macro_f1"]["delta"] - 0.1100) < 5e-4 and abs(h["brier"]["delta"] - 0.0050) < 5e-4
    assert abs(h["precision_at_80"]["delta"] + 0.1362) < 5e-4
    assert abs(k["raw"]["metrics"]["micro_f1"]["delta"] + 0.0269) < 5e-4
    assert "in-sample" in k["note"] and "not tuned" in k["note"]
    assert k["headline"]["temperature"] == {"laya": built["systems"]["laya"]["chosen"]["temperature"],
                                            "berturk": built["systems"]["berturk"]["chosen"]["temperature"]}


@pytest.mark.data
def test_fewshot_is_the_sampler_output_and_its_draws_check(built):
    from pilot import fewshot
    assert built["fewshot"] == fewshot.protocol_block()
    path = os.path.join(pilot.data.REPO_ROOT, built["splits"]["selection_train"]["path"])
    with open(path, "rb") as f:
        rows = pilot.data.parse_jsonl(f.read())
    fewshot.check_draws(built["fewshot"], rows)


@pytest.mark.data
def test_power_assumptions_and_delegation_are_the_constants(built):
    assert built["power_context"] == pr.POWER_CONTEXT
    assert built["assumptions"] == pr.ASSUMPTIONS
    assert built["delegation"] == pr.DELEGATION


# ---- 08-06: render_markdown and --render ----

HEADING = "### Frozen decision protocol (protocol.json)"


def _system(kind):
    return {"kind": kind,
            "final_manifest": {"path": "results/postfit/%s-final.json" % kind, "sha256": "1" * 64},
            "postfit": {"path": "results/postfit/%s.json" % kind, "sha256": "2" * 64},
            "chosen": {"dump_run_id": "%s-chosen" % kind, "epoch": 1, "epochs_schedule": 2, "temperature": 1.5,
                       "theta": {"shipped": {"global": 0.55, "n_per_tag": 1}, "raw": {"global": 0.45, "n_per_tag": 0}},
                       "encoder_lr": 2e-05, "head_lr": 0.0001},
            "seeds": [0],
            "final_seeds": [{"seed": 0, "train_run_id": "%s-train-s0" % kind, "checkpoint_name": "ckpt-%s-s0" % kind,
                             "dump_run_id": "%s-final-s0" % kind}],
            "dump_run_ids": ["%s-final-s0" % kind]}


def _deltas(T):
    return {"n": 4, "B": 7, "seed": 3, "level": 0.9, "method": "percentile", "coverage": 50,
            "metrics": {"micro_f1": {"a": 0.5, "b": 0.75, "delta": -0.25, "ci": [-0.5, 0.125], "width": 0.625,
                                     "n_undefined": 0}},
            "temperature": {"laya": T, "berturk": T}, "theta": {"laya": "theta.shipped", "berturk": "theta.shipped"}}


def tiny():
    """A hand-written protocol/1 dict holding every block render_markdown reads, in small form (no build, no data)."""
    p = small()
    p.update({
        "frozen": {"when": "before any tiny test number", "why": "nothing tiny is tuned to the result",
                   "identity": "the sha256 of the tiny bytes"},
        "inputs": {"data/splits.json": "a" * 64},
        "splits": {"test": {"path": "data/splits/test.jsonl", "role": "test", "sha256": "b" * 64,
                            "uids_sha256": "c" * 64, "n_messages": 4, "n_no_intent": 1},
                   "hotels_probe": {"path": "data/splits/hotels_probe.jsonl", "role": "ood_probe", "sha256": "d" * 64,
                                    "uids_sha256": "e" * 64, "n_messages": 3, "n_no_intent": 0,
                                    "ood_scoring_tags": ["card", "loan"]}},
        "systems": {"laya": _system("laya"), "berturk": _system("berturk")},
        "references": {"rule": "tiny references are reported, never gated",
                       "rows": {"prior": {"run_id": "prior", "report": "results/reports/prior.json", "gated": False,
                                          "hotels_probe": None}}},
        "scoring": {"splits": ["test"], "device": "mps", "workers": 0, "refused": ["--force", "--n"],
                    "why": "the tiny scorer of the val dumps"},
        "variant": {"headline": "the tiny record T", "raw": "tiny T = 1, reported only"},
        "seed_averaging": {"rule": "tiny probabilities are averaged before any threshold"},
        "bootstrap": {"B": 7, "seed": 3, "level": 0.9, "coverage": 50, "unit": "message", "method": "percentile",
                      "procedure": "a tiny paired bootstrap", "extra": "one tiny draw"},
        "criteria": {
            "a": {"name": "non-inferiority", "rule": "PASS when the tiny CI lower bound >= micro_ci_lower_min",
                  "micro_ci_lower_min": -0.02, "macro_point_min": -0.03,
                  "macro_ci_label": "not resolvable at this tiny size"},
            "c": {"name": "latency", "rule": "tiny latency is reported", "label": "reported, not gated"},
            "d": {"name": "an edge", "rule": "PASS when any tiny path is MET",
                  "paths": [{"path": "calibration", "metric": "brier", "delta_below": 0.0},
                            {"path": "fewshot", "metric": "new_tag_macro_ap", "k": 10, "mean_delta_at_least": 0.05}],
                  "reported_only": {"classwise_ece": "tiny classwise ECE is reported, not gated"}},
            "alternative_go": {"name": "alternative GO", "rule": "MET when the tiny delta > delta_above",
                               "delta_above": 0.03, "requires": "b PASS"}},
        "collapse": {"rule": "a tiny run is collapsed when any component is",
                     "applies_to": [{"runs": "each tiny final seed", "source": "its tiny dump",
                                     "components": ["loss", "test_floor"]}],
                     "components": {"loss": {"rule": "collapsed on a NaN tiny loss", "note": "a tiny note"},
                                    "test_floor": {"rule": "collapsed below the tiny prior",
                                                   "addition": "a tiny marked addition"}},
                     "not_applied": "no tiny val floor on final seeds",
                     "prior_theta_global": 0.12, "prior_counts": {"test": {"tp": 1, "fp": 2, "fn": 3}},
                     "prior_test_fraction": "2/7", "prior_test_micro_f1": 2 / 7,
                     "prior_source": "the tiny prior's dumps"},
        "verdict": {"rules": ["(b) FAIL -> NO-GO", "(a) INCONCLUSIVE -> INCONCLUSIVE (counts as NO-GO; tiny)"],
                    "inconclusive_means": "counts as NO-GO; tiny bank data",
                    "provisional": "PROVISIONAL: %s if the tiny path is met, else %s",
                    "units": "tiny deltas are fractions", "statuses": {"overall": ["GO", "NO-GO", "INCONCLUSIVE"]}},
        "definitions": {"brier": "tiny mean of (p - y)^2", "classwise_ece": "tiny |sum of y - sum of p| / n"},
        "fewshot": {"tags": ["loan", "card"], "k_values": [1, 2], "draw_seeds": [0], "neg_per_pos": 4, "replay": 2,
                    "epochs": 10,
                    "eligibility": {"rule": {"min_test_positives": 10},
                                    "counts": {"test": {"loan": 11, "card": 12}, "final_train": {"loan": 40, "card": 41}},
                                    "note": "tiny gold labels only"},
                    "source": {"split": "selection_train", "sha256": "f" * 64, "uids_sha256": "0" * 64},
                    "sampler": {"schema": "fewshot-draw/1", "rule": "tiny sorted uids", "digest": "tiny sha256"},
                    "draws": [{"draw": 0, "seed": 0, "base_seed": 0, "support": {"1": ["u1", "u2"], "2": ["u1", "u2", "u5"]},
                               "replay": ["u3", "u4"], "positives": {"1": {"loan": 1, "card": 1},
                                                                    "2": {"loan": 2, "card": 3}},
                               "digest": "9876543210ab" + "c" * 52}],
                    "rules": {"masking": "the tiny tags are masked",
                              "base_runs": {"laya": {"epochs": 4, "encoder_lr": 2e-05}, "rule": "one tiny base run"},
                              "phase_9_notes": ["add a tiny --mask-tags"]}},
        "known_at_freeze": {"split": "val", "in_sample": True, "n": 4, "laya": "laya-chosen",
                            "berturk": "berturk-chosen", "headline": _deltas(1.5), "raw": _deltas(1.0),
                            "note": "tiny in-sample val"},
        "power_context": {"label": "tiny research estimate", "n": 4, "p_a_lower_at_least_margin": {"0": 0.5},
                          "margin": "the tiny margin is locked"},
        "assumptions": [{"id": "A1", "decision": "tiny Brier names (d)", "taken": "tiny taken"},
                        {"id": "OQ1", "decision": "a tiny open question", "taken": "tiny taken"}],
        "delegation": {"standing": {"given": "today", "quote": "do the tiny control."}, "bank_name": "tiny deferred"},
        "code": {"git_head": "abc123", "dirty": False},
    })
    return p


def test_render_starts_with_the_heading_and_shows_the_sha():
    md = pr.render_markdown(tiny(), "5" * 64)
    assert md.startswith(HEADING + "\n") and md.endswith("\n")
    assert "5" * 64 in md
    for text in ("before any tiny test number", "nothing tiny is tuned to the result", "#### Criteria",
                 "#### Collapse rule (b)", "#### Verdict"):
        assert text in md, text


def test_render_is_deterministic_on_a_hand_written_dict():
    a = pr.render_markdown(tiny(), "5" * 64)
    assert a == pr.render_markdown(tiny(), "5" * 64) == pr.render_markdown(copy.deepcopy(tiny()), "5" * 64)


def test_render_shows_thresholds_in_points_and_the_decision_texts():
    md = pr.render_markdown(tiny(), "5" * 64)
    for text in ("-2.00 points", "-3.00 points", "+3.00 points", "+5.00 points", "+0.00 points",
                 "not resolvable at this tiny size", "reported, not gated", "tiny classwise ECE is reported, not gated",
                 "2/7", "0.2857", "a tiny marked addition", "a tiny note", "no tiny val floor on final seeds",
                 "counts as NO-GO; tiny bank data", "PROVISIONAL: %s if the tiny path is met, else %s",
                 "tiny deltas are fractions"):
        assert text in md, text
    p = tiny()
    for rule in p["verdict"]["rules"]:
        assert rule in md, rule
    for c in p["criteria"].values():
        assert c["rule"] in md, c["rule"]


def test_render_reads_every_value_from_the_dict_passed_in():
    p = tiny()
    p["criteria"]["a"]["micro_ci_lower_min"] = -0.04
    p["collapse"]["prior_test_fraction"], p["collapse"]["prior_test_micro_f1"] = "3/11", 3 / 11
    p["verdict"]["rules"] = ["(z) a changed rule -> GO"]
    md = pr.render_markdown(p, "5" * 64)
    assert "-4.00 points" in md and "-2.00 points" not in md
    assert "3/11" in md and "0.2727" in md and "2/7" not in md
    assert "(z) a changed rule -> GO" in md and "(b) FAIL -> NO-GO" not in md


def test_cli_render_previews_a_path_with_its_own_sha(tmp_path, capsys):
    path, sha = frozen(tmp_path, tiny())
    pr.main(["--render", path])
    assert capsys.readouterr().out == pr.render_markdown(tiny(), sha)


def test_cli_render_without_path_refuses_while_unpinned():
    assert pr.PROTOCOL_SHA256 is None
    with pytest.raises(SystemExit, match="not frozen|not found"):
        pr.main(["--render"])


def test_cli_render_without_path_renders_the_pinned_protocol(tmp_path, monkeypatch, capsys):
    path, sha = frozen(tmp_path, tiny())
    monkeypatch.setattr(pr, "PROTOCOL_PATH", path)
    monkeypatch.setattr(pr, "PROTOCOL_SHA256", sha)
    pr.main(["--render"])
    assert capsys.readouterr().out == pr.render_markdown(tiny(), sha)


@pytest.mark.parametrize("raw, what", [(b"{not json", "not valid JSON"), (b"[1]\n", "not a JSON object"),
                                       (b'{"schema": "other/1"}\n', "schema")])
def test_cli_render_refuses_a_file_that_is_not_a_protocol(tmp_path, raw, what):
    path = tmp_path / "p.json"
    path.write_bytes(raw)
    with pytest.raises(SystemExit, match=what):
        pr.main(["--render", str(path)])


def test_cli_render_refuses_out():
    with pytest.raises(SystemExit):
        pr.main(["--render", "--out", "x.json"])


@pytest.mark.data
def test_render_shows_the_decision_sections(built):
    md = pr.render_markdown(built, "5" * 64)
    assert md.startswith(HEADING + "\n") and "5" * 64 in md
    for text in ("-2.00 points", "-3.00 points", "+3.00 points", "+5.00 points", "Brier", "10/41", "520/2163",
                 "not resolvable at this test size (n = 413)", "counts as NO-GO; collect bank data", "PROVISIONAL"):
        assert text in md, text
    assert md == pr.render_markdown(copy.deepcopy(built), "5" * 64)


@pytest.mark.data
def test_cli_render_of_a_built_protocol(tmp_path, capsys):
    out = str(tmp_path / "p.json")
    pr.main(["--build", "--out", out])
    capsys.readouterr()
    with open(out, "rb") as f:
        raw = f.read()
    pr.main(["--render", out])
    md = capsys.readouterr().out
    assert md == pr.render_markdown(json.loads(raw.decode("utf-8")), pilot.data.sha256_bytes(raw))
    assert "counts as NO-GO; collect bank data" in md


# ---- 08-06 Task 2: the remaining sections ----

SECTIONS = ("#### Criteria", "#### Collapse rule (b)", "#### Verdict", "#### Seed averaging, θ and bootstrap",
            "#### Systems, splits and inputs", "#### Few-shot protocol (Phase 9 executes it unchanged)",
            "#### Known at freeze (val, in-sample, one seed per model)", "#### Power at the test size (research estimates)",
            "#### Definitions", "#### Assumptions and delegation")


def _strings(x):
    """Every string value under x (dict values and list items, never dict keys)."""
    if isinstance(x, str):
        return [x]
    if isinstance(x, dict):
        return [s for v in x.values() for s in _strings(v)]
    if isinstance(x, list):
        return [s for v in x for s in _strings(v)]
    return []


def frozen_texts(p):
    """Every rule, definition, note and quote text of a protocol dict that the README block must show verbatim."""
    texts = list(p["definitions"].values()) + list(p["verdict"]["rules"])
    texts += [p["verdict"]["inconclusive_means"], p["verdict"]["provisional"], p["verdict"]["units"]]
    for c in p["criteria"].values():
        texts += [c["rule"]] + list((c.get("reported_only") or {}).values())
    texts += [p["collapse"]["rule"], p["collapse"]["not_applied"], p["collapse"]["prior_source"]]
    texts += _strings(p["collapse"]["components"])
    texts += _strings(p["variant"]) + _strings(p["seed_averaging"]) + _strings(p["scoring"])
    texts += [p["references"]["rule"], p["bootstrap"]["procedure"], p["bootstrap"]["extra"]]
    texts += _strings(p["fewshot"]["rules"]) + _strings(p["fewshot"]["sampler"]) + [p["fewshot"]["eligibility"]["note"]]
    texts += [p["known_at_freeze"]["note"]] + _strings(p["power_context"])
    texts += [a["decision"] for a in p["assumptions"]] + [a["taken"] for a in p["assumptions"]]
    texts += _strings(p["delegation"]) + _strings(p["frozen"])
    return texts


def test_render_shows_every_section_in_order_on_a_hand_written_dict():
    md = pr.render_markdown(tiny(), "5" * 64)
    at = [md.index(s) for s in SECTIONS]
    assert at == sorted(at)


def test_render_shows_every_frozen_text_of_a_hand_written_dict():
    md = pr.render_markdown(tiny(), "5" * 64)
    for text in frozen_texts(tiny()):
        assert text in md, text


def test_render_shows_the_remaining_values_of_a_hand_written_dict():
    md = pr.render_markdown(tiny(), "5" * 64)
    for text in ("B = 7", "seed 3", "level 0.9", "coverage 50%", "loan", "card", "9876543210ab", "A1", "OQ1",
                 "tiny research estimate", "in-sample", "-0.2500", "[-0.5000, +0.1250]", "laya-chosen",
                 "berturk-chosen", "laya-final-s0", "berturk-final-s0", "1.5000", "0.55", "0.45",
                 "data/splits/test.jsonl", "ood_scoring_tags: card, loan", "do the tiny control."):
        assert text in md, text
    assert "9876543210abc" not in md                    # digests show their first 12 hex characters only
    assert "u1" not in md and "u5" not in md            # support and replay uids are counted, never listed
    assert "| 0 | 0 | 0 | 2 | 3 | 2 | loan 2, card 3 | 9876543210ab |" in md


@pytest.mark.data
def test_render_shows_the_remaining_sections(built):
    md = pr.render_markdown(built, "5" * 64)
    for s in SECTIONS:
        assert s in md, s
    assert "B = 10000" in md and "research estimate" in md and "in-sample" in md
    for tag in built["fewshot"]["tags"]:
        assert tag in md, tag
    for d in built["fewshot"]["draws"]:
        assert d["digest"][:12] in md and d["digest"] not in md
    for a in built["assumptions"]:
        assert "- %s: " % a["id"] in md, a["id"]
    delta = built["known_at_freeze"]["headline"]["metrics"]["micro_f1"]["delta"]
    assert "%+.4f" % delta in md and "%+.4f" % delta == "-0.0155"
    for kind in pr.KINDS:
        for s in built["systems"][kind]["final_seeds"]:
            assert s["dump_run_id"] in md
    assert "%.4f" % built["systems"]["laya"]["chosen"]["temperature"] in md


@pytest.mark.data
def test_render_shows_every_frozen_text(built):
    md = pr.render_markdown(built, "5" * 64)
    for text in list(built["definitions"].values()) + list(built["verdict"]["rules"]):
        assert text in md, text
    for c in built["criteria"].values():
        assert c["rule"] in md, c["rule"]
    for text in frozen_texts(built):
        assert text in md, text


@pytest.mark.data
def test_render_holds_no_local_path_or_message_text(built):
    from test_repo_hygiene import home_dir, local_split_texts
    md = pr.render_markdown(built, "5" * 64)
    home = home_dir()
    assert home is None or home not in md
    assert pilot.data.REPO_ROOT not in md
    texts = local_split_texts()
    assert texts, "make data first: the message-text check needs the committed splits"
    # a message text never appears inside the block; a few messages are a bare number or word ("40"), so those
    # short ones are checked as whole lines, bullet values and table cells instead of as substrings
    pieces = set()
    for line in md.split("\n"):
        pieces |= {line.strip(), line.lstrip("- ").strip()} | {c.strip() for c in line.split("|")}
        pieces |= {x.strip() for x in line.split(": ")}
    assert [t for t in texts if len(t) >= 12 and t in md] == []
    assert sorted(t for t in texts if t in pieces) == []
