"""Unit contracts of the latency harness (EVAL-06): integer nearest rank, summary edges, the warm-up and sync contract,
the label and the render, every refusal before a model load, and torch-free imports.

The refusals run on the synthetic snapshot and a tiny Laya checkpoint (tests/test_latency_cpu.py's helpers). Each must
come before the quiet-machine preflight (which runs right before torch is imported) and before any model load: the
preflight, score.resolve_checkpoint, score.Predictor, score.check_berturk and score.load_bert all fail the test if
they are reached. Offline, no Ray, no weights: runs in `make test`.
"""
import json
import os
import subprocess
import sys

import pytest

import latency
import pilot.data
from pilot import latency as plat
from test_latency_cpu import PREFLIGHT, laya_setup, run_args, val_cases
from test_score_berturk import _relabelled

pytestmark = pytest.mark.unit

HEAVY = ("torch", "transformers", "laya", "ray")


def _boom(what):
    def boom(*a, **k):
        raise AssertionError("%s was reached before the refusal" % what)
    return boom


@pytest.fixture
def no_load(monkeypatch):
    """Fail the test if the preflight or any checkpoint load is reached (a refusal must come first)."""
    monkeypatch.setattr(latency, "preflight", _boom("the preflight"))
    for name in ("resolve_checkpoint", "Predictor", "check_berturk", "load_bert"):
        monkeypatch.setattr(latency.score, name, _boom("score." + name))


@pytest.fixture
def laya(tmp_path, synthetic_raw):
    """(splits dir, splits manifest, final manifest, checkpoint root) of one tiny Laya seed with T = 1.3."""
    return laya_setup(tmp_path, synthetic_raw)


def _write(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return str(path)


def _toy_report(**meta):
    """A latency/1 report of two seeds with known times (no model): seed 0 at 1..20 ms, seed 1 at 21..40 ms."""
    base = {"system": "laya", "device": "cpu", "dtype": "torch.float32", "amp_enabled": False, "threads": 4,
            "warmup": 10, "batch": 1, "versions": {"torch": "x", "numpy": "y"}, "mps_fallback_env": None,
            "code": {"git_sha": "a" * 40, "git_dirty": False, "git_diff_sha256": "b" * 64},
            "preflight": dict(PREFLIGHT), "swap_used_bytes": {"before": 100, "after": 100},
            "split": {"name": "val", "path": "splits/val.jsonl", "sha256": "c" * 64, "uids_sha256": "d" * 64,
                      "n_messages": 20, "manifest_is_committed": True},
            "final_manifest": {"path": "results/postfit/laya-final.json", "sha256": "e" * 64, "is_committed": True}}
    base.update(meta)
    seeds = []
    for seed in (0, 1):
        times = [(20 * seed + i) / 1000.0 for i in range(1, 21)]
        seeds.append({"seed": seed, "checkpoint_name": "ckpt-final-s%d" % seed, "times_s": times,
                      "summary": plat.summarize(times, "toy")})
    return plat.build_report(base, seeds)


# ---- statistics ----

def test_nearest_rank_uses_integer_ranks():
    for n, p50, p95 in ((208, 104, 198), (20, 10, 19), (100, 50, 95), (11, 6, 11), (1, 1, 1)):
        s = list(range(1, n + 1))
        assert plat.nearest_rank(s, 50) == p50 and plat.nearest_rank(s, 95) == p95
    assert plat.nearest_rank(list(range(1, 101)), 7) == 7        # math.ceil(7 / 100 * 100) is 8 (Pitfall 3)
    assert plat.nearest_rank([7.5], 50) == plat.nearest_rank([7.5], 95) == 7.5
    with pytest.raises(SystemExit, match="empty"):
        plat.nearest_rank([], 50)
    for q in (0, 101, 9.5, True):
        with pytest.raises(SystemExit, match="integer percent"):
            plat.nearest_rank([1, 2], q)


def test_summarize_single_and_empty():
    with pytest.raises(SystemExit, match="laya seed 2 on mps: no timings to summarise"):
        plat.summarize([], "laya seed 2 on mps")
    assert plat.summarize([0.25], "x") == {"n": 1, "min": 0.25, "p50": 0.25, "p95": 0.25, "mean": 0.25, "max": 0.25}
    s = plat.summarize([0.3, 0.1, 0.2], "x")                # unsorted in, nearest rank on the sorted list
    assert (s["min"], s["p50"], s["p95"], s["max"], s["n"]) == (0.1, 0.2, 0.3, 0.3, 3)


def test_time_messages_excludes_warmup_and_syncs_around_each_message():
    events, calls, syncs = [], [], []
    ticks = iter([0.0, 1.0, 10.0, 13.0, 20.0, 26.0, 30.0, 34.0, 40.0, 45.0])

    def fn(t):
        calls.append(t)
        events.append("fn:" + t)

    def sync():
        syncs.append(1)
        events.append("sync")

    def clock():
        events.append("clock")
        return next(ticks)
    texts = ["a", "b", "c", "d", "e"]
    out = plat.time_messages(fn, texts, 2, sync, clock=clock)
    assert calls == ["a", "b"] + texts and len(calls) == 7
    assert len(syncs) == 1 + 2 * 5
    assert out == [1.0, 3.0, 6.0, 4.0, 5.0]                # warm-up calls are not timed
    assert events[:3] == ["fn:a", "fn:b", "sync"]           # warm-up, then one sync
    assert events[3:8] == ["sync", "clock", "fn:a", "sync", "clock"]   # sync before t0, sync after the host copy
    assert plat.time_messages(fn, ["x"], 0, sync, clock=iter([5.0, 5.5]).__next__) == [0.5]


# ---- label and render ----

def test_label_in_report_and_markdown():
    r = _toy_report()
    assert plat.LABEL == "reported, not gated" and r["label"] == plat.LABEL and r["schema"] == "latency/1"
    assert r["pooled"]["n"] == 40 and r["pooled"]["p50"] == 0.020 and r["pooled"]["p95"] == 0.038
    md = plat.render_markdown([r])
    assert md.startswith("### Latency per message (Phase 7, EVAL-06): reported, not gated\n")
    assert "| 0 | ckpt-final-s0 | 20 | 10.0 | 19.0 |" in md and "| pooled | 2 seed(s) | 40 | 20.0 | 38.0 |" in md
    assert "fp16 autocast default on MPS (laya_default_amp) is not what is timed" in md


def test_render_from_json_shows_busy_banner(tmp_path, capsys):
    quiet = json.loads(json.dumps(_toy_report()))
    assert "Busy-machine" not in plat.render_markdown([quiet])
    overridden = _toy_report(preflight=dict(PREFLIGHT, override=True, min_available_gib=0))
    idle_swap = _toy_report(preflight=dict(PREFLIGHT, swap_grew_while_idle=True))
    loop_swap = _toy_report(swap_used_bytes={"before": 100, "after": 2 ** 30})
    for r in (overridden, idle_swap, loop_swap):
        assert "**Busy-machine run:**" in plat.render_markdown([json.loads(json.dumps(r))])
    d = tmp_path / "latency"
    d.mkdir()
    _write(d / "laya-cpu.json", quiet)
    _write(d / "berturk-mps.json", _toy_report(system="berturk", device="mps"))
    md = latency.main(["--render", str(d), "--out", str(tmp_path / "latency.md")])
    assert capsys.readouterr().out == md == latency.main(["--render", str(d)])   # identical on every render
    assert (tmp_path / "latency.md").read_bytes() == md.encode("utf-8")
    assert md.index("#### berturk on mps") < md.index("#### laya on cpu")    # sorted by system, then device
    _write(d / "other.json", {"schema": "confidence/1"})
    with pytest.raises(SystemExit, match="is not a latency/1 report"):
        latency.main(["--render", str(d)])
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit, match="holds no latency report"):
        latency.main(["--render", str(tmp_path / "empty")])
    with pytest.raises(SystemExit, match="is not a directory of latency reports"):
        latency.main(["--render", str(tmp_path / "missing")])


# ---- refusals before any load ----

def test_refuses_non_val_split_before_load(laya, no_load, tmp_path):
    splits, manifest, m, root = laya
    for name, why in (("test", "has role test; expected one of val"), ("hotels_probe", "has role ood_probe"),
                      ("selection_train", "has role train")):
        args = run_args("laya", splits, manifest, m, root, tmp_path / "out.json")
        args[args.index("--split") + 1] = os.path.join(splits, name + ".jsonl")
        with pytest.raises(SystemExit, match=why):
            latency.main(args)
    assert not (tmp_path / "out.json").exists()


def test_refuses_committed_test_content_before_load(laya, no_load, tmp_path, monkeypatch):
    """A manifest copy registering the committed test or HOTELS rows (or a part of them) as val is refused by content.
    The synthetic data stands in for the committed split (as tests/test_score_berturk.py's interlock test does)."""
    splits, manifest, m, root = laya
    monkeypatch.setattr(pilot.data, "DEFAULT_MANIFEST", manifest)
    for held in ("test", "hotels_probe"):
        with open(os.path.join(splits, held + ".jsonl"), "rb") as f:
            lines = f.read().splitlines(keepends=True)
        for part, rows in (("whole", lines), ("part", lines[1:])):
            val, other = _relabelled((splits, manifest, None), tmp_path / ("%s-%s" % (held, part)), rows)
            assert pilot.data.verify_split_file(val, manifest_path=other, roles=("val",))["role"] == "val"
            args = run_args("laya", splits, other, m, root, tmp_path / "out.json")
            args[args.index("--split") + 1] = val
            with pytest.raises(SystemExit, match="holds committed %s content; latency is measured on val" % held):
                latency.main(args)
    assert not (tmp_path / "out.json").exists()


def test_refuses_manifest_of_other_system_or_schema(laya, no_load, tmp_path):
    splits, manifest, m, root = laya
    with open(m, encoding="utf-8") as f:
        good = json.load(f)
    cases = [("laya", _write(tmp_path / "b.json", dict(good, system_kind="berturk")),
              "final manifest of berturk, not laya"),
             ("berturk", m, "final manifest of laya, not berturk"),
             ("laya", _write(tmp_path / "s.json", dict(good, schema="postfit/1")), "is not a postfit-final/1"),
             ("laya", _write(tmp_path / "e.json", dict(good, seeds=[])), "lists no final seeds"),
             ("laya", _write(tmp_path / "t.json", dict(good, seeds=[dict(good["seeds"][0], temperature=1)])),
              "no float temperature"),
             ("laya", _write(tmp_path / "d.json", dict(good, seeds=good["seeds"] * 2)), "repeats seed 0"),
             ("laya", str(tmp_path / "missing.json"), "cannot be read")]
    for system, path, why in cases:
        with pytest.raises(SystemExit, match=why):
            latency.main(run_args(system, splits, manifest, path, root, tmp_path / "out.json"))
    assert not (tmp_path / "out.json").exists()


@pytest.mark.parametrize("name", ["../ckpt-s0", "a/b", "a\\b", "..", ".", "", "/abs"])
def test_refuses_unsafe_checkpoint_name(laya, no_load, tmp_path, name):
    splits, manifest, m, root = laya
    with open(m, encoding="utf-8") as f:
        good = json.load(f)
    bad = _write(tmp_path / "bad.json", dict(good, seeds=[dict(good["seeds"][0], checkpoint_name=name)]))
    with pytest.raises(SystemExit, match="not a plain directory name"):
        latency.main(run_args("laya", splits, manifest, bad, root, tmp_path / "out.json"))


def test_refuses_identity_mismatch_before_load(laya, no_load, tmp_path):
    splits, manifest, m, root = laya
    with open(m, encoding="utf-8") as f:
        good = json.load(f)
    seed = good["seeds"][0]
    ident = json.loads(json.dumps(seed["checkpoint_identity"]))
    ident["model_sha256"] = "0" * 64
    cases = [(dict(seed, checkpoint_identity=ident), "not the checkpoint the final manifest names for seed 0"),
             (dict(seed, dump_run_id="laya-000000000000-00000000-00000000"), "not the checkpoint the final manifest"),
             (dict(seed, checkpoint_name="ckpt-s9"), r"--checkpoint-root/ckpt-s9 \(seed 0\) not found")]
    for i, (s, why) in enumerate(cases):
        bad = _write(tmp_path / ("bad%d.json" % i), dict(good, seeds=[s]))
        with pytest.raises(SystemExit, match=why):
            latency.main(run_args("laya", splits, manifest, bad, root, tmp_path / "out.json"))
    # a second, valid seed listed after a bad one: still refused before the first seed is loaded
    bad = _write(tmp_path / "late.json", dict(good, seeds=[seed, dict(seed, seed=1, checkpoint_name="ckpt-s9")]))
    with pytest.raises(SystemExit, match="ckpt-s9"):
        latency.main(run_args("laya", splits, manifest, bad, root, tmp_path / "out.json"))
    assert not (tmp_path / "out.json").exists()


def test_refuses_served_temperature_other_than_the_manifest(laya, tmp_path, monkeypatch):
    """After the (real, tiny) load: a final seed must serve the manifest's T, else nothing is timed."""
    import torch
    threads = torch.get_num_threads()
    monkeypatch.setattr(latency, "preflight", lambda g: dict(PREFLIGHT))
    monkeypatch.setattr(plat, "time_messages", _boom("the timed loop"))
    splits, manifest, m, root = laya
    with open(m, encoding="utf-8") as f:
        good = json.load(f)
    bad = _write(tmp_path / "t.json", dict(good, seeds=[dict(good["seeds"][0], temperature=1.3 + 1e-9)]))
    try:
        with pytest.raises(SystemExit, match="serves T 1.3, the final manifest records"):
            latency.main(run_args("laya", splits, manifest, bad, root, tmp_path / "out.json"))
    finally:
        torch.set_num_threads(threads)


def test_argument_errors(capsys):
    for argv in (["--system", "laya", "--render", "x"], [], ["--system", "laya", "--threads", "0"],
                 ["--system", "laya", "--warmup", "-1"], ["--system", "laya", "--min-available-gib", "-1"],
                 ["--system", "other"], ["--system", "laya", "--device", "tpu"]):
        with pytest.raises(SystemExit) as e:
            latency.main(argv)
        assert e.value.code == 2, argv
    capsys.readouterr()


def test_latency_modules_are_torch_free():
    code = ("import sys, pilot.latency, latency; bad = [m for m in %r if m in sys.modules]; "
            "assert not bad, bad" % (HEAVY,))
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "-c", code], cwd=pilot.data.REPO_ROOT, env=env, capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr


def test_val_texts_reach_the_tokenizer_unchanged(laya, tmp_path, monkeypatch):
    """A message is the row's state string, unchanged (no lowercasing, folding or normalization), in split-file
    order, after the warm-up: what the timed callable receives is exactly the val rows' texts."""
    import torch
    threads = torch.get_num_threads()
    seen = []

    def spy(fn, texts, warmup, sync, clock=None):
        seen.append((list(texts), warmup))
        return [0.001] * len(texts)
    monkeypatch.setattr(latency, "preflight", lambda g: dict(PREFLIGHT))
    monkeypatch.setattr(plat, "time_messages", spy)
    splits, manifest, m, root = laya
    try:
        latency.main(run_args("laya", splits, manifest, m, root, tmp_path / "out.json"))
    finally:
        torch.set_num_threads(threads)
    assert seen == [([c["state"] for c in val_cases(splits)], 1)]
