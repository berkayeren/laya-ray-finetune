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


def small():
    """A hand-written protocol dict: enough for the byte identity and the reader's refusals."""
    return {"schema": pr.SCHEMA, "title": "a hand-written protocol", "units": "fractions; ünits"}


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
