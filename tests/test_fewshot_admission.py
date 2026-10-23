"""Few-shot test admission (FEW-01, FEW-02; D-01, D-04, D-09, D-14; 09-05): planted few-shot checkpoints ->
`fewshot.py --inspect` -> `fewshot.py --runs-manifest` -> score.py's protocol gate scores a listed checkpoint on test
exactly once; every allow-list refusal; the guarded deletion modes.

tests/fewshot_synthetic.bench installs the synthetic frozen protocol (tag card, k 1 and 2, one draw, base seed 0, tiny
recipes, scoring on cpu with workers 0, a D-08 decision record) and plant_fewshot_set plants, without training, one
consistent masked base and continued set per kind (tiny Laya checkpoints with a finished ray_train.py run record,
tiny BERTurk files with berturk_train.json). pilot.fewshot.RUNS_PATH and pilot.dumps.DUMP_ROOT point at temp paths.
Offline, no Ray: runs in `make test`.
"""
import json
import os

import pytest

import fewshot as runner
import pilot.data
import score
from fewshot_synthetic import bench, plant_fewshot_set
from pilot import dumps
from pilot import fewshot as fs

pytestmark = pytest.mark.unit


@pytest.fixture
def keep_threads():
    import torch
    n = torch.get_num_threads()
    yield
    torch.set_num_threads(n)


@pytest.fixture
def planted(tmp_path, synthetic_raw, monkeypatch, keep_threads):
    """The synthetic protocol, a planted set, a temp RUNS_PATH and DUMP_ROOT."""
    ns = bench(tmp_path, synthetic_raw, monkeypatch)
    ns.ckpts = plant_fewshot_set(ns, str(tmp_path / "ckpt"))
    ns.runs_path = str(tmp_path / "fewshot" / "runs.json")
    monkeypatch.setattr(fs, "RUNS_PATH", ns.runs_path, raising=False)
    ns.root = str(tmp_path / "dumps")
    monkeypatch.setattr(dumps, "DUMP_ROOT", ns.root)
    ns.tmp = tmp_path
    return ns


def split(ns, name):
    return os.path.join(ns.out_dir, "splits", name + ".jsonl")


def laya_argv(ns, name, ckpt, *extra):
    return [split(ns, name), "--model", "laya", "--checkpoint", ckpt, "--workers", "0", "--device", "cpu",
            "--manifest", ns.manifest] + list(extra)


def sidecar(run_dir, name):
    with open(os.path.join(run_dir, name + ".json"), encoding="utf-8") as f:
        return json.load(f)


def test_runs_manifest_admits_a_listed_checkpoint_on_test_once(planted, capsys):
    ns = planted
    names = sorted(ns.ckpts)
    assert names == ["berturk-base-s0", "berturk-d0-k1", "berturk-d0-k2", "laya-base-s0", "laya-d0-k1", "laya-d0-k2"]
    dirs = [ns.ckpts[n] for n in names]

    # --inspect: each checkpoint re-validated on its own, one entry each, its dump run id and proof printed
    capsys.readouterr()
    shown = runner.main(["--inspect"] + dirs)
    printed = json.loads(capsys.readouterr().out)
    assert printed == shown and len(shown) == len(dirs)
    by_name = {e["checkpoint_name"]: e for e in shown}
    assert sorted(by_name) == names
    lc = by_name["laya-d0-k2"]
    assert lc["kind"] == "laya" and lc["stage"] == "continued" and (lc["draw"], lc["k"]) == (0, 2)
    assert lc["digest"] == ns.block["draws"][0]["digest"] and lc["base_seed"] == 0
    assert lc["base_run_id"] == by_name["laya-base-s0"]["train_run_id"]
    assert lc["dump_run_id"].startswith("laya-")
    assert lc["proof"]["new_tag_items"] == {"card": lc["proof"]["support_rows"]}
    assert by_name["laya-base-s0"]["proof"]["held_out_items"] == {"card": 0}
    assert by_name["berturk-base-s0"]["proof"]["unmasked_held_out_labels"] == {"card": 0}
    assert by_name["berturk-d0-k1"]["dump_run_id"].startswith("berturk-")
    assert lc["wall_clock_s"] == pytest.approx(2.0 + 0.5 + 1.0 + 0.5)      # WALL_RULE: train_s + save_s + load + save
    assert by_name["berturk-d0-k1"]["wall_clock_s"] == pytest.approx(2.0 + 1.0)
    assert set(lc) == set(fs.ENTRY_KEYS)
    for text in (json.dumps(shown),):
        assert "bank_" not in text and "hotel_" not in text                    # no uid list or text

    # --runs-manifest: written once, check_runs accepts it, entries sorted by (kind, stage, seed, draw, k)
    runs = runner.main(["--runs-manifest"] + list(reversed(dirs)) + ["--out", ns.runs_path])
    with open(ns.runs_path, "rb") as f:
        raw = f.read()
    assert json.loads(raw) == runs and raw.endswith(b"\n")
    assert runs["schema"] == fs.RUNS_SCHEMA and runs["protocol_sha256"] == ns.pin and runs["mode"] == "primary"
    assert runs["wall_rule"] == fs.WALL_RULE
    with open(ns.decision_path, "rb") as f:
        assert runs["decision"]["sha256"] == pilot.data.sha256_bytes(f.read())
    key = [(e["kind"], e["stage"], e["seed"], -1 if e["draw"] is None else e["draw"], -1 if e["k"] is None else e["k"])
           for e in runs["entries"]]
    assert key == sorted(key) and len(key) == 6
    assert fs.check_runs(runs, ns.protocol, ns.pin, "primary") == runs["entries"]
    got, sha = fs.read_runs(ns.protocol, ns.pin)
    assert got == runs and sha == pilot.data.sha256_bytes(raw)
    assert fs.admitted(ns.protocol, ns.pin, "laya", lc["dump_run_id"]) == (by_name["laya-d0-k2"], sha)
    assert fs.admitted(ns.protocol, ns.pin, "berturk", lc["dump_run_id"]) is None

    # score.py: the listed continued checkpoint on test is admitted once, bound to the pin and the allow-list
    run_dir, name = score.main(laya_argv(ns, "test", ns.ckpts["laya-d0-k2"]))
    assert (run_dir, name) == (os.path.join(ns.root, lc["dump_run_id"]), "test")
    side = sidecar(run_dir, "test")
    assert side["protocol_sha256"] == ns.pin and side["fewshot_runs_sha256"] == sha
    logits, _, _, entry = dumps.load_dump(run_dir, "test", ns.manifest)
    assert logits.shape == (entry["n_messages"], 48)
    with pytest.raises(SystemExit) as e:
        score.main(laya_argv(ns, "test", ns.ckpts["laya-d0-k2"]))
    text = str(e.value.code)
    assert text.startswith(score.GATE) and "already exists" in text and "\n" not in text

    # the masked base on val: ungated, as every val dump is (D-05)
    run_dir, _ = score.main(laya_argv(ns, "val", ns.ckpts["laya-base-s0"]))
    side = sidecar(run_dir, "val")
    assert "protocol_sha256" not in side and "fewshot_runs_sha256" not in side
    assert os.path.basename(run_dir) == by_name["laya-base-s0"]["dump_run_id"]


# ---- check_runs: one checkpoint per slot, the frozen draws and pairing, the proofs (09-05 Task 2) ----

@pytest.fixture
def listed(planted):
    """planted plus its runs manifest written by fewshot.py --runs-manifest (ns.runs)."""
    planted.runs = runner.main(["--runs-manifest"] + sorted(planted.ckpts.values()) + ["--out", planted.runs_path])
    return planted


def edited(runs, edit):
    out = json.loads(json.dumps(runs))
    edit(out)
    return out


def entry(runs, kind, stage, draw=None, k=None):
    return next(e for e in runs["entries"] if (e["kind"], e["stage"], e["draw"], e["k"]) == (kind, stage, draw, k))


def check_refused(ns, runs, *needles, mode="primary"):
    with pytest.raises(SystemExit) as e:
        fs.check_runs(runs, ns.protocol, ns.pin, mode)
    text = str(e.value.code)
    assert "\n" not in text, text
    for n in needles:
        assert n in text, (n, text)
    assert "--runs-manifest" in text, text                                # names the fix
    return text


def test_check_runs_refuses_a_duplicate_base_seed(listed):
    def edit(r):
        b = entry(r, "laya", "base")
        r["entries"].insert(r["entries"].index(b) + 1, dict(b, dump_run_id="laya-x", checkpoint_name="x",
                                                            train_run_id="laya-y"))
    check_refused(listed, edited(listed.runs, edit), "second entry", "(kind laya, base seed 0)")


def test_check_runs_refuses_a_duplicate_draw_k(listed):
    def edit(r):
        c = entry(r, "berturk", "continued", 0, 2)
        r["entries"].append(dict(c, dump_run_id="berturk-x", checkpoint_name="x", train_run_id="berturk-y"))
    check_refused(listed, edited(listed.runs, edit), "second entry", "(kind berturk, draw 0, k 2)")


@pytest.mark.parametrize("key", ["dump_run_id", "checkpoint_name", "train_run_id"])
def test_check_runs_refuses_a_repeated_run_id_or_name(listed, key):
    def edit(r):
        entry(r, "laya", "continued", 0, 1)[key] = entry(r, "laya", "continued", 0, 2)[key]
    check_refused(listed, edited(listed.runs, edit), "repeats %s" % key)


def test_check_runs_refuses_a_wrong_base_pairing(listed):
    def seed(r):
        entry(r, "laya", "continued", 0, 1)["base_seed"] = 1
    check_refused(listed, edited(listed.runs, seed), "base_seed 1", "base_seed_of")

    def run(r):
        entry(r, "laya", "continued", 0, 1)["base_run_id"] = "laya-selection_train-neg2-s0-00000000"
    check_refused(listed, edited(listed.runs, run), "is not the listed laya base of seed 0")


def test_check_runs_refuses_a_fallback_manifest_with_another_base(listed):
    """Under the fallback mode the bases are the compute fallback's seed 0 only: a base of seed 1 is refused."""
    def edit(r):
        r["mode"] = "fallback"
        entry(r, "berturk", "base")["seed"] = 1
    check_refused(listed, edited(listed.runs, edit), "base seed 1", "fallback protocol", mode="fallback")
    check_refused(listed, listed.runs, "lists the 'primary' protocol", mode="fallback")


def test_check_runs_refuses_a_changed_digest_or_draw(listed):
    def digest(r):
        entry(r, "laya", "continued", 0, 2)["digest"] = "0" * 64
    check_refused(listed, edited(listed.runs, digest), "digest", "frozen draw 0")

    def k(r):
        entry(r, "laya", "continued", 0, 2)["k"] = 3
    with pytest.raises(SystemExit, match="few-shot k 3 is not one of the frozen k values"):
        fs.check_runs(edited(listed.runs, k), listed.protocol, listed.pin, "primary")


@pytest.mark.parametrize("kind, stage, key, value, needle", [
    ("laya", "base", "held_out_items", {"card": 1}, "held_out_items card is 1, not 0"),
    ("berturk", "base", "unmasked_held_out_labels", {"card": 2}, "unmasked_held_out_labels card is 2, not 0"),
    ("laya", "continued", "new_tag_items", {"card": 0}, "not one per support row"),
    ("berturk", "continued", "replay_unmasked_held_out_labels", 1, "replay_unmasked_held_out_labels is 1"),
    ("laya", "continued", "support_rows", 1, "support_rows 1"),
], ids=["laya_base", "berturk_base", "laya_continued", "berturk_replay", "support_size"])
def test_check_runs_refuses_a_nonzero_or_wrong_proof(listed, kind, stage, key, value, needle):
    def edit(r):
        e = entry(r, kind, stage, *((0, 2) if stage == "continued" else ()))
        e["proof"][key] = value
        if key == "support_rows":
            e["proof"]["new_tag_items"] = {"card": value}
    check_refused(listed, edited(listed.runs, edit), needle)


def test_check_runs_refuses_an_unknown_entry_key_or_top_key(listed):
    check_refused(listed, edited(listed.runs, lambda r: entry(r, "laya", "base").update(extra=1)),
                  "unknown ['extra']")
    check_refused(listed, edited(listed.runs, lambda r: r.update(note="x")), "expected exactly")
    check_refused(listed, edited(listed.runs, lambda r: r.update(protocol_sha256="f" * 64)),
                  "protocol sha256 " + "f" * 64)


def test_check_runs_refuses_an_incomplete_or_unsorted_manifest(listed):
    def drop(r):
        r["entries"].remove(entry(r, "berturk", "continued", 0, 1))
    check_refused(listed, edited(listed.runs, drop), "incomplete", "berturk draw 0 k 1")
    check_refused(listed, edited(listed.runs, lambda r: r["entries"].reverse()), "not sorted")


def test_admitted_is_none_without_a_runs_manifest(planted):
    assert not os.path.exists(planted.runs_path)
    assert fs.read_runs(planted.protocol, planted.pin) == (None, None)
    assert fs.admitted(planted.protocol, planted.pin, "laya", "laya-anything") is None


def test_read_runs_refuses_a_changed_decision_record(listed):
    with open(listed.decision_path, "a", encoding="utf-8") as f:
        f.write("\n")
    with pytest.raises(SystemExit, match="decision record sha256") as e:
        fs.read_runs(listed.protocol, listed.pin)
    assert "\n" not in str(e.value.code)


# ---- the CLI refusals: --runs-manifest and --inspect ----

def cli_refused(argv, *needles):
    with pytest.raises(SystemExit) as e:
        runner.main(argv)
    text = str(e.value.code)
    assert "\n" not in text, text
    for n in needles:
        assert n in text, (n, text)
    return text


def test_runs_manifest_refuses_an_incomplete_set(planted):
    dirs = [d for n, d in sorted(planted.ckpts.items()) if n not in ("laya-d0-k1", "berturk-d0-k2")]
    cli_refused(["--runs-manifest"] + dirs + ["--out", planted.runs_path], "incomplete",
                "laya draw 0 k 1", "berturk draw 0 k 2")
    assert not os.path.exists(planted.runs_path)


def test_runs_manifest_refuses_a_continued_run_without_its_base(planted):
    dirs = [d for n, d in sorted(planted.ckpts.items()) if n != "laya-base-s0"]
    cli_refused(["--runs-manifest"] + dirs + ["--out", planted.runs_path], "not among the checkpoints given")
    assert not os.path.exists(planted.runs_path)


def test_runs_manifest_refuses_a_base_whose_digests_differ(planted):
    """A base with the same run record but other weights (a retrained base of the same canonical config) is not the
    base the continued runs started from: their canonical base_model_sha256 names the original."""
    from fewshot_synthetic import plant_laya
    base = planted.ckpts["laya-base-s0"]
    with open(os.path.join(base, "rl_agent_config.json"), encoding="utf-8") as f:
        rt = json.load(f)["training"]["ray_train"]
    other = plant_laya(str(planted.tmp / "other" / "laya-base-s0"), rt["canonical"], rt["record"], seed=7)
    dirs = [other if d == base else d for d in sorted(planted.ckpts.values())]
    cli_refused(["--runs-manifest"] + dirs + ["--out", planted.runs_path], "base_model_sha256", "laya-base-s0")
    assert not os.path.exists(planted.runs_path)


def test_runs_manifest_refuses_an_existing_out(listed):
    with open(listed.runs_path, "rb") as f:
        before = f.read()
    cli_refused(["--runs-manifest"] + sorted(listed.ckpts.values()) + ["--out", listed.runs_path], "exists",
                "written once")
    with open(listed.runs_path, "rb") as f:
        assert f.read() == before


def test_inspect_refuses_an_unfinished_laya_checkpoint(planted):
    os.remove(os.path.join(planted.ckpts["laya-d0-k1"], "final.json"))
    cli_refused(["--inspect", planted.ckpts["laya-d0-k1"]], "not a finished run", "final.json missing")


def test_inspect_refuses_a_checkpoint_whose_binding_no_longer_passes(planted):
    """A continued checkpoint trained at another encoder LR than the frozen continued_training (its run id is its
    canonical's) no longer binds to the protocol."""
    from fewshot_synthetic import plant_laya
    src = planted.ckpts["laya-d0-k2"]
    with open(os.path.join(src, "rl_agent_config.json"), encoding="utf-8") as f:
        rt = json.load(f)["training"]["ray_train"]
    other = plant_laya(str(planted.tmp / "lr" / "laya-d0-k2"), dict(rt["canonical"], encoder_lr=3e-05), rt["record"])
    cli_refused(["--inspect", other], "no longer binds to the frozen protocol", "encoder_lr")


def test_inspect_refuses_a_record_that_is_not_its_identity(planted):
    """A checkpoint that is not a Phase 9 run (no few-shot binding) is refused; so is one holding both kinds' files."""
    ckpt = planted.ckpts["berturk-base-s0"]
    with open(os.path.join(ckpt, "rl_agent_config.json"), "w", encoding="utf-8") as f:
        f.write("{}")
    cli_refused(["--inspect", ckpt], "holds both")
