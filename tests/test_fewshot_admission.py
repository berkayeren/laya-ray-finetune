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
import shutil

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


# ---- guarded deletion: --prune after verified dumps, --drop-storage of a finished Laya run (D-14; 09-05 Task 3) ----

def gated_test(ns, name):
    """score.py of a listed checkpoint on test (the gate admits it once); returns the run dir."""
    return score.main(laya_argv(ns, "test", ns.ckpts[name]))[0]


def set_side(run_dir, split_name, edit):
    path = os.path.join(run_dir, split_name + ".json")
    with open(path, encoding="utf-8") as f:
        side = json.load(f)
    edit(side)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(side, f)


def test_prune_deletes_a_listed_continued_checkpoint_after_its_gated_test_dump(listed, capsys):
    gated_test(listed, "laya-d0-k2")
    keep = sorted(n for n in listed.ckpts if n != "laya-d0-k2")
    capsys.readouterr()
    got = runner.main(["--prune", listed.ckpts["laya-d0-k2"]])
    out = capsys.readouterr().out
    rid = entry(listed.runs, "laya", "continued", 0, 2)["dump_run_id"]
    assert got == [rid] and "laya-d0-k2" in out and rid in out
    assert not os.path.exists(listed.ckpts["laya-d0-k2"])
    assert all(os.path.isdir(listed.ckpts[n]) for n in keep)
    assert dumps.load_dump(os.path.join(listed.root, rid), "test", listed.manifest)[2]["fewshot_runs_sha256"]


def test_prune_refuses_a_continued_checkpoint_before_its_test_dump(listed):
    cli_refused(["--prune", listed.ckpts["laya-d0-k1"]], "laya-d0-k1", "no verified test dump", "score")
    assert os.path.isdir(listed.ckpts["laya-d0-k1"])


@pytest.mark.parametrize("case, needle", [
    ("identity", "another checkpoint's"),
    ("ungated", "not scored through the protocol gate"),
    ("other_runs", "not scored through the protocol gate"),
    ("subset", "subset dump"),
])
def test_prune_refuses_a_dump_that_is_not_this_checkpoints_gated_one(listed, case, needle):
    run_dir = gated_test(listed, "laya-d0-k2")
    if case == "identity":
        set_side(run_dir, "test", lambda s: s["system"]["checkpoint_identity"].update(model_sha256="0" * 64))
    elif case == "ungated":
        set_side(run_dir, "test", lambda s: (s.pop("protocol_sha256"), s.pop("fewshot_runs_sha256")))
    elif case == "other_runs":
        set_side(run_dir, "test", lambda s: s.update(fewshot_runs_sha256="e" * 64))
    else:
        set_side(run_dir, "test", lambda s: s.update(n_subset=7))
    cli_refused(["--prune", listed.ckpts["laya-d0-k2"]], needle)
    assert os.path.isdir(listed.ckpts["laya-d0-k2"])


def test_prune_refuses_an_unlisted_dir_a_symlink_and_no_manifest(listed):
    from tiny_laya import tiny_checkpoint
    final = tiny_checkpoint(listed.tmp / "ckpt-final-s0")             # a Phase 8 final checkpoint stand-in
    cli_refused(["--prune", final], "not listed in results/fewshot/runs.json")
    assert os.path.isdir(final)
    gated_test(listed, "laya-d0-k2")
    link = str(listed.tmp / "link-d0-k2")
    os.symlink(listed.ckpts["laya-d0-k2"], link)
    cli_refused(["--prune", link], "symlink")
    assert os.path.isdir(listed.ckpts["laya-d0-k2"]) and os.path.islink(link)
    os.remove(listed.runs_path)
    cli_refused(["--prune", listed.ckpts["laya-d0-k2"]], "no results/fewshot/runs.json")
    assert os.path.isdir(listed.ckpts["laya-d0-k2"])


def test_prune_deletes_a_base_only_after_its_val_test_and_continued_dumps(listed):
    base = listed.ckpts["laya-base-s0"]
    text = cli_refused(["--prune", base], "laya-base-s0", "val dump", "test dump")
    assert "laya-d0-k1" in text and "laya-d0-k2" in text            # names every missing continued test dump
    score.main(laya_argv(listed, "val", base))                        # ungated val dump
    gated_test(listed, "laya-base-s0")
    gated_test(listed, "laya-d0-k1")
    text = cli_refused(["--prune", base], "laya-d0-k2")
    assert "laya-d0-k1" not in text and "val dump" not in text
    assert os.path.isdir(base)
    gated_test(listed, "laya-d0-k2")
    rid = entry(listed.runs, "laya", "base")["dump_run_id"]
    assert runner.main(["--prune", base]) == [rid]
    assert not os.path.exists(base) and os.path.isdir(listed.ckpts["berturk-base-s0"])


def test_prune_refuses_a_base_whose_val_dump_is_gated(listed):
    """The base's θ source is its own ungated val dump (D-05); a val dump carrying protocol_sha256 is not it."""
    base = listed.ckpts["laya-base-s0"]
    run_dir, _ = score.main(laya_argv(listed, "val", base))
    set_side(run_dir, "val", lambda s: s.update(protocol_sha256=listed.pin))
    cli_refused(["--prune", base], "val dump", "gated")
    assert os.path.isdir(base)


def ray_storage(ns, name):
    """A Ray storage dir holding the run dir of planted checkpoint `name` and another run's dir; returns (storage,
    run dir, other dir)."""
    from pilot import runs as rr
    rid = rr.read_record(ns.ckpts[name])["run_id"]
    storage = ns.tmp / "ray"
    for d in (storage / rid / "epoch-001", storage / "laya-selection_train-neg8-s0-other"):
        d.mkdir(parents=True)
        (d / "state.bin").write_bytes(b"x")
    return str(storage), str(storage / rid), str(storage / "laya-selection_train-neg8-s0-other")


def test_drop_storage_deletes_the_run_dir_of_a_finished_laya_checkpoint(planted, capsys):
    storage, run_dir, other = ray_storage(planted, "laya-d0-k1")
    capsys.readouterr()
    got = runner.main(["--drop-storage", planted.ckpts["laya-d0-k1"], "--storage", storage])
    assert got == os.path.basename(run_dir) and os.path.basename(run_dir) in capsys.readouterr().out
    assert not os.path.exists(run_dir) and os.path.isdir(other) and os.path.isdir(planted.ckpts["laya-d0-k1"])


def test_drop_storage_refuses_a_missing_or_symlinked_run_dir(planted):
    storage, run_dir, other = ray_storage(planted, "laya-d0-k1")
    cli_refused(["--drop-storage", planted.ckpts["laya-d0-k2"], "--storage", storage], "not found")
    os.rename(run_dir, run_dir + "-real")
    os.symlink(run_dir + "-real", run_dir)
    cli_refused(["--drop-storage", planted.ckpts["laya-d0-k1"], "--storage", storage], "symlink")
    assert os.path.isdir(run_dir + "-real") and os.path.isdir(other)


def test_drop_storage_refuses_an_unfinished_checkpoint_or_one_inside_the_run_dir(planted):
    storage, run_dir, _ = ray_storage(planted, "laya-d0-k1")
    os.remove(os.path.join(planted.ckpts["laya-d0-k1"], "final.json"))
    cli_refused(["--drop-storage", planted.ckpts["laya-d0-k1"], "--storage", storage], "not a finished run")
    assert os.path.isdir(run_dir)
    from pilot import runs as rr
    src = planted.ckpts["laya-d0-k2"]
    rid = rr.read_record(src)["run_id"]
    inside = planted.tmp / "ray2" / rid / "final"
    shutil.copytree(src, str(inside))
    cli_refused(["--drop-storage", str(inside), "--storage", str(planted.tmp / "ray2")], "holds the checkpoint")
    assert os.path.isdir(str(inside))


def test_drop_storage_refuses_a_berturk_or_non_phase_9_checkpoint(planted):
    from tiny_laya import tiny_checkpoint
    storage, run_dir, _ = ray_storage(planted, "laya-d0-k1")
    cli_refused(["--drop-storage", planted.ckpts["berturk-d0-k1"], "--storage", storage], "ray_train.py")
    plain = tiny_checkpoint(planted.tmp / "plain")
    cli_refused(["--drop-storage", plain, "--storage", storage], "ray_train.py run record")
    assert os.path.isdir(run_dir)


@pytest.mark.parametrize("argv, needle", [
    (["--prune", "x", "--out", "y"], "--out"),
    (["--prune", "x", "--storage", "y"], "--storage belongs to --drop-storage"),
    (["--drop-storage", "x"], "--drop-storage needs --storage"),
    (["--drop-storage", "x", "--storage", "y", "--manifest", "m"], "--manifest"),
    (["--inspect", "x", "--storage", "y"], "--storage belongs to --drop-storage"),
    (["--counts", "--storage", "y"], "--storage belongs to --drop-storage"),
], ids=["prune_out", "prune_storage", "drop_no_storage", "drop_manifest", "inspect_storage", "counts_storage"])
def test_prune_and_drop_storage_refuse_other_modes_flags(argv, needle, capsys):
    with pytest.raises(SystemExit) as e:
        runner.main(argv)
    assert e.value.code == 2 and needle in capsys.readouterr().err


def test_no_mode_error_lists_every_mode(capsys):
    with pytest.raises(SystemExit) as e:
        runner.main([])
    err = capsys.readouterr().err
    assert e.value.code == 2
    for flag in ("--support-sets", "--counts", "--projection", "--inspect", "--runs-manifest", "--prune",
                 "--drop-storage"):
        assert flag in err, flag
