"""Contracts of pilot/fewshot.py: the few-shot protocol frozen before any fine-tuned test number existed. Nothing is
trained here; the few-shot experiment executes the protocol unchanged.

- draw: deterministic per seed, nested (the k = 5 support set is the first 5 positives and first 20 negatives of each
  tag of the k = 10 draw), sized by the settings, the replay set disjoint from the k = 10 support set, every list sorted
  and duplicate-free; checked against an independent reading of SAMPLER_RULE written here, not taken from the code.
- the realised positive count of every held-out tag per k, and the digest of DIGEST_RULE.
- every pool smaller than a draw needs is refused, naming the tag and the counts; an unknown or repeated tag and a
  repeated uid are refused.
- pilot.fewshot imports the stdlib only.
- data-marked: the three draws on the verified committed selection_train.
- fewshot.py --support-sets, under a temp protocol pinned by monkeypatch around a block drawn on the
  synthetic selection_train: prints the protocol sha256 and each draw's digest and counts (no uid list, no text);
  refuses a missing, unpinned or one-byte-changed protocol before the split is read, a block of another source and a
  draw with one uid changed; no mode flag is an argparse error.
Offline except the `data` tests.
"""
import hashlib
import json
import os
import random
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, NoReturn

import pytest

import pilot.data
from conftest import SyntheticRaw, prepared
from intents import INTENTS
from pilot import fewshot as fs

pytestmark = pytest.mark.unit

HEAVY = ("torch", "numpy", "scipy", "sklearn", "laya", "ray", "transformers")
TAGS = ("savings", "loan", "overdraft", "appointment", "contactless", "direct_debit")


def synthetic_rows(n: int = 600, cut: tuple[str, int] | None = None) -> list[dict[str, Any]]:
    """n rows with full 48-tag labels: row i carries TAGS[i % 12] when i % 12 < 6 (50 positives per tag at n = 600),
    savings and loan together every 25th row (co-occurrence) and card every 3rd; uids bank_<fold>_<index>, in a
    shuffled order (the sampler must sort). cut=(tag, m) keeps only the first m positives of tag."""
    rows, seen = [], {}
    for i in range(n):
        e = {t: False for t in INTENTS}
        if i % 12 < 6:
            e[TAGS[i % 12]] = True
        if i % 25 == 0:
            e["savings"] = e["loan"] = True
        e["card"] = i % 3 == 0
        if cut is not None and e[cut[0]]:
            seen[cut[0]] = seen.get(cut[0], 0) + 1
            if seen[cut[0]] > cut[1]:
                e[cut[0]] = False
        rows.append({"state": "mesaj %d" % i, "uid": "bank_%d_%d" % (i // 30, i % 30), "questions": {}, "expected": e})
    random.Random(7).shuffle(rows)
    return rows


def by_rule(rows: list[dict[str, Any]], seed: int, tags: Sequence[str] = TAGS, k_values: Sequence[int] = (5, 10),
            neg_per_pos: int = 4, replay: int = 200) -> tuple[dict[str, list[str]], list[str]]:
    """SAMPLER_RULE read independently of pilot.fewshot: uids sorted; one random.Random(seed); per tag in order, a
    sample of max(k) positives, then of neg_per_pos x max(k) negatives; the k set is the union of the first k positives
    and first neg_per_pos x k negatives of every tag; replay is a sample from the sorted uids outside the max-k set."""
    uids = sorted(r["uid"] for r in rows)
    lab = {r["uid"]: r["expected"] for r in rows}
    rng, kmax, picks = random.Random(seed), max(k_values), {}
    for t in tags:
        pos = rng.sample([u for u in uids if lab[u][t]], kmax)
        neg = rng.sample([u for u in uids if not lab[u][t]], neg_per_pos * kmax)
        picks[t] = (pos, neg)
    support = {str(k): sorted({u for t in tags for u in picks[t][0][:k] + picks[t][1][:neg_per_pos * k]})
               for k in k_values}
    top = set(support[str(kmax)])
    rep = sorted(rng.sample([u for u in uids if u not in top], replay))
    return support, rep


def test_draw_is_deterministic() -> None:
    rows = synthetic_rows()
    a, b = fs.draw(rows, 0), fs.draw(rows, 0)
    assert a == b and fs.draw_digest(a) == fs.draw_digest(b)
    c = fs.draw(rows, 1)
    assert c["support"] != a["support"] and fs.draw_digest(c) != fs.draw_digest(a)
    assert fs.draw(list(reversed(rows)), 0) == a                  # the row order never matters: pools are sorted uids


def test_draw_is_nested() -> None:
    rows = synthetic_rows()
    lab = {r["uid"]: r["expected"] for r in rows}
    for seed in (0, 1, 2):
        d = fs.draw(rows, seed)
        assert d["schema"] == "fewshot-draw/1" and d["seed"] == seed and d["tags"] == list(TAGS)
        assert sorted(d["support"]) == ["10", "5"]
        s5, s10 = set(d["support"]["5"]), set(d["support"]["10"])
        assert s5 <= s10 and len(s10) <= 6 * 50
        for t in TAGS:
            assert sum(lab[u][t] for u in s10) >= 10 and sum(not lab[u][t] for u in s10) >= 40
            assert sum(lab[u][t] for u in s5) >= 5 and sum(not lab[u][t] for u in s5) >= 20
        assert len(d["replay"]) == 200 and not set(d["replay"]) & s10
        for xs in (d["support"]["5"], d["support"]["10"], d["replay"]):
            assert xs == sorted(set(xs))
        assert (d["support"], d["replay"]) == by_rule(rows, seed)       # the exact rule, read independently


def test_draw_realised_positives() -> None:
    rows = synthetic_rows()
    lab = {r["uid"]: r["expected"] for r in rows}
    d = fs.draw(rows, 0)
    assert sorted(d["positives"]) == ["10", "5"]
    for k in ("5", "10"):
        assert list(d["positives"][k]) == list(TAGS)
        for t in TAGS:
            n = sum(lab[u][t] for u in d["support"][k])
            assert d["positives"][k][t] == n and n >= int(k)
    assert json.loads(json.dumps(d)) == d                                 # string k keys: the JSON round-trips


def test_draw_other_settings() -> None:
    rows = synthetic_rows()
    d = fs.draw(rows, 3, tags=("card",), k_values=(1, 2), neg_per_pos=1, replay=2)
    assert (d["support"], d["replay"]) == by_rule(rows, 3, ("card",), (1, 2), 1, 2)
    assert d["tags"] == ["card"] and sorted(d["support"]) == ["1", "2"] and len(d["replay"]) == 2


def test_draw_digest_rule() -> None:
    d = fs.draw(synthetic_rows(), 2)
    body = {"schema": d["schema"], "seed": d["seed"], "tags": d["tags"], "support": d["support"], "replay": d["replay"]}
    want = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()
    assert fs.draw_digest(d) == want
    assert fs.draw_digest(dict(d, positives={})) == want                  # positives are derived, not digested
    moved = json.loads(json.dumps(d))
    moved["support"]["10"][0] = "bank_99_99"
    assert fs.draw_digest(moved) != want


def test_draw_positive_pool_too_small_refused() -> None:
    rows = synthetic_rows(cut=("loan", 3))
    with pytest.raises(SystemExit) as e:
        fs.draw(rows, 0)
    msg = str(e.value)
    assert "loan" in msg and "3" in msg and "10" in msg


def test_draw_negative_pool_too_small_refused() -> None:
    rows = synthetic_rows(n=60)
    for r in rows:
        r["expected"]["card"] = True
    for r in rows[:5]:
        r["expected"]["card"] = False
    with pytest.raises(SystemExit) as e:
        fs.draw(rows, 0, tags=("card",), k_values=(1, 2), neg_per_pos=4, replay=2)
    assert "card" in str(e.value) and "negative" in str(e.value) and "5" in str(e.value)


def test_draw_replay_pool_too_small_refused() -> None:
    rows = synthetic_rows(n=240)
    with pytest.raises(SystemExit) as e:
        fs.draw(rows, 0)
    assert "replay" in str(e.value) and "200" in str(e.value)


def test_draw_bad_tags_and_uids_refused() -> None:
    rows = synthetic_rows()
    with pytest.raises(SystemExit, match="not_a_tag"):
        fs.draw(rows, 0, tags=("savings", "not_a_tag"))
    with pytest.raises(SystemExit, match="loan"):
        fs.draw(rows, 0, tags=("loan", "savings", "loan"))
    with pytest.raises(SystemExit, match=rows[0]["uid"]):
        fs.draw(rows + [dict(rows[0])], 0)


def test_fewshot_imports_stdlib_only() -> None:
    code = "import sys, pilot.fewshot; bad = [m for m in %r if m in sys.modules]; assert not bad, bad" % (HEAVY,)
    env = {k: val for k, val in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "-c", code], cwd=pilot.data.REPO_ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def committed_rows(name: str, roles: tuple[str, ...]) -> list[dict[str, Any]]:
    path = os.path.join(os.path.dirname(pilot.data.DEFAULT_MANIFEST), "splits", name + ".jsonl")
    with open(path, "rb") as f:
        raw = f.read()
    pilot.data.verify_split_file(path, roles=roles, data=raw)
    return pilot.data.parse_jsonl(raw)


@pytest.mark.data
def test_committed_selection_train_draws() -> None:
    rows = committed_rows("selection_train", ("train",))
    assert len(rows) == 1450
    uids = {r["uid"] for r in rows}
    for seed in (0, 1, 2):
        a, b = fs.draw(rows, seed), fs.draw(rows, seed)
        assert a == b and fs.draw_digest(a) == fs.draw_digest(b)
        assert set(a["support"]["10"]) | set(a["replay"]) <= uids
        assert set(a["support"]["5"]) <= set(a["support"]["10"]) and len(a["replay"]) == 200
        assert (a["support"], a["replay"]) == by_rule(rows, seed)


# ---- eligibility, the protocol block and check_draws ----

TEST_POSITIVES = dict(zip(TAGS, (13, 11, 13, 13, 13, 12)))          # committed test.jsonl, labels only
FINAL_TRAIN_POSITIVES = dict(zip(TAGS, (56, 57, 56, 52, 50, 42)))   # committed final_train.jsonl, labels only
RULE_KEYS = ("base_runs", "masking", "new_question", "support_labels", "replay", "continued_training",
             "laya_item_rule", "primary_metric", "secondary_metrics", "compute_fallback", "gate", "test_admission",
             "phase_9_notes")


def test_settings_match_the_approved_matrix() -> None:
    from pilot import compute
    m = compute.MATRIX
    assert fs.TAGS == TAGS and len(fs.TAGS) == m["fewshot_tags"]
    assert fs.K_VALUES == tuple(m["fewshot_k"]) and len(fs.DRAW_SEEDS) == m["fewshot_draws"]
    assert fs.DRAW_SEEDS == (0, 1, 2)
    assert (fs.NEG_PER_POS, fs.REPLAY, fs.EPOCHS) == (m["fewshot_neg_per_pos"], m["fewshot_replay"],
                                                      m["fewshot_epochs"])
    assert len(fs.BASE_RUNS["seeds"]) == m["fewshot_base_runs"]


def test_base_runs_are_the_chosen_recipes() -> None:
    """The masked base runs use each model's committed chosen recipe (postfit records) and shared LRs."""
    from pilot import berturk, postfit
    for kind in ("laya", "berturk"):
        with open(os.path.join(pilot.data.REPO_ROOT, "results", "postfit", kind + ".json"), encoding="utf-8") as f:
            chosen = json.load(f)["chosen"]
        b = fs.BASE_RUNS[kind]
        assert (b["epochs"], b["stop_after_epoch"], b["encoder_lr"]) == (chosen["epochs_schedule"],
                                                                         chosen["stop_after_epoch"],
                                                                         chosen["encoder_lr"])
        assert b["head_lr"] == postfit.HEAD_LR and b["device"] == "mps"
    assert fs.BASE_RUNS["laya"]["neg"] == 8 and fs.BASE_RUNS["berturk"]["batch"] == berturk.BATCH
    assert fs.BASE_RUNS["berturk"]["mask_label"] == berturk.MASK_LABEL
    assert fs.BASE_RUNS["split"] == "selection_train" and list(fs.BASE_RUNS["seeds"]) == list(fs.DRAW_SEEDS)


def test_check_eligibility_passes_the_committed_counts() -> None:
    from pilot import pins
    counts = {"test": dict(TEST_POSITIVES), "final_train": dict(FINAL_TRAIN_POSITIVES)}
    got = fs.check_eligibility(TAGS, counts, pins.OOD_SCORING_TAGS)
    assert got == {"rule": fs.ELIGIBILITY, "counts": counts}
    assert fs.ELIGIBILITY["min_test_positives"] == 10 and fs.ELIGIBILITY["min_final_train_positives"] == 40


@pytest.mark.parametrize("split,tag,n,word", [("test", "loan", 9, "test"), ("final_train", "direct_debit", 39,
                                                                              "final_train")])
def test_check_eligibility_refuses_a_short_count(split: str, tag: str, n: int, word: str) -> None:
    from pilot import pins
    counts = {"test": dict(TEST_POSITIVES), "final_train": dict(FINAL_TRAIN_POSITIVES)}
    counts[split][tag] = n
    with pytest.raises(SystemExit) as e:
        fs.check_eligibility(TAGS, counts, pins.OOD_SCORING_TAGS)
    assert tag in str(e.value) and str(n) in str(e.value) and word in str(e.value)


def test_check_eligibility_refuses_a_generic_or_uncounted_tag() -> None:
    from pilot import pins
    counts = {"test": dict(TEST_POSITIVES, affirm=40), "final_train": dict(FINAL_TRAIN_POSITIVES, affirm=200)}
    with pytest.raises(SystemExit, match="affirm"):
        fs.check_eligibility(TAGS + ("affirm",), counts, pins.OOD_SCORING_TAGS)
    del counts["final_train"]["savings"]
    with pytest.raises(SystemExit, match="savings"):
        fs.check_eligibility(TAGS, counts, pins.OOD_SCORING_TAGS)


def synthetic_block(rows: list[dict[str, Any]], seeds: Sequence[int] = (0, 1)) -> dict[str, Any]:
    """A few-shot block around the synthetic rows, built the way a consumer would: draw, then draw_digest."""
    draws = []
    for i, s in enumerate(seeds):
        d = fs.draw(rows, s)
        draws.append({"draw": i, "seed": s, "base_seed": s, "support": d["support"], "replay": d["replay"],
                      "positives": d["positives"], "digest": fs.draw_digest(d)})
    uids = pilot.data.sha256_bytes("\n".join(r["uid"] for r in rows).encode("utf-8"))
    return {"tags": list(TAGS), "k_values": [5, 10], "draw_seeds": list(seeds), "neg_per_pos": 4, "replay": 200,
            "source": {"split": "selection_train", "sha256": "0" * 64, "uids_sha256": uids}, "draws": draws}


def test_check_draws_passes_its_own_rows() -> None:
    rows = synthetic_rows()
    assert fs.check_draws(synthetic_block(rows), rows) is None


def test_check_draws_refuses_a_changed_support_list() -> None:
    rows = synthetic_rows()
    b = synthetic_block(rows)
    lst = b["draws"][1]["support"]["10"]
    lst[0] = next(r["uid"] for r in sorted(rows, key=lambda r: r["uid"]) if r["uid"] not in lst)
    lst.sort()
    with pytest.raises(SystemExit, match="draw 1"):
        fs.check_draws(b, rows)


def test_check_draws_refuses_a_changed_digest() -> None:
    rows = synthetic_rows()
    b = synthetic_block(rows)
    b["draws"][0]["digest"] = "f" * 64
    with pytest.raises(SystemExit, match="draw 0.*digest"):
        fs.check_draws(b, rows)


def test_check_draws_refuses_rows_of_another_split() -> None:
    rows, other = synthetic_rows(), synthetic_rows(n=700)
    with pytest.raises(SystemExit, match="draw 0"):
        fs.check_draws(synthetic_block(rows), other)


def test_check_draws_refuses_a_misnumbered_or_missing_draw() -> None:
    rows = synthetic_rows()
    b = synthetic_block(rows)
    b["draws"][1]["base_seed"] = 0
    with pytest.raises(SystemExit, match="draw 1"):
        fs.check_draws(b, rows)
    b = synthetic_block(rows)
    b["draws"].pop()
    with pytest.raises(SystemExit, match="draw"):
        fs.check_draws(b, rows)


def test_protocol_block_refuses_another_manifest(tmp_path: Path, synthetic_raw: SyntheticRaw) -> None:
    _, manifest = prepared(tmp_path, synthetic_raw)
    with pytest.raises(SystemExit, match="committed"):
        fs.protocol_block(manifest)


@pytest.mark.data
def test_protocol_block_on_the_committed_data() -> None:
    b = fs.protocol_block()
    assert b == fs.protocol_block()                                     # deterministic
    assert json.loads(json.dumps(b)) == b
    assert b["tags"] == list(TAGS) and b["k_values"] == [5, 10] and b["draw_seeds"] == [0, 1, 2]
    assert (b["neg_per_pos"], b["replay"], b["epochs"]) == (4, 200, 10)
    assert b["eligibility"]["counts"] == {"test": TEST_POSITIVES, "final_train": FINAL_TRAIN_POSITIVES}
    assert b["eligibility"]["rule"] == fs.ELIGIBILITY and "labels only, no model output" in b["eligibility"]["note"]
    with open(pilot.data.DEFAULT_MANIFEST, encoding="utf-8") as f:
        entry = json.load(f)["splits"]["selection_train"]
    assert b["source"] == {"split": "selection_train", "sha256": entry["sha256"], "uids_sha256": entry["uids_sha256"]}
    assert b["sampler"] == {"schema": fs.DRAW_SCHEMA, "rule": fs.SAMPLER_RULE, "digest": fs.DIGEST_RULE}
    rows = committed_rows("selection_train", ("train",))
    assert len(b["draws"]) == 3
    for i, d in enumerate(b["draws"]):
        assert (d["draw"], d["seed"], d["base_seed"]) == (i, i, i)
        assert d["digest"] == fs.draw_digest(fs.draw(rows, i)) and len(d["replay"]) == 200
    assert sorted(b["rules"]) == sorted(RULE_KEYS)
    assert "pilot.metrics.AP_RULE" in b["rules"]["primary_metric"]["rule"]
    assert b["rules"]["compute_fallback"]["base_runs_per_model"] == 1
    assert fs.check_draws(b, rows) is None


# ---- fewshot.py --support-sets: the few-shot runner's draw check ----

def runner() -> ModuleType:
    """The root-level few-shot runner (fewshot.py), imported here so a missing runner fails inside the test."""
    import fewshot
    return fewshot


def frozen_protocol(tmp_path: Path, synthetic_raw: SyntheticRaw, monkeypatch: pytest.MonkeyPatch,
                    edit: Callable[[dict[str, Any]], None] | None = None) -> tuple[dict[str, Any], str]:
    """A REQUIRED_KEYS-complete protocol around a few-shot block drawn on the synthetic selection_train (tags card,
    k 1 and 2, draw seed 0, 1 negative per positive, replay 2), written to tmp_path and pinned: PROTOCOL_PATH,
    PROTOCOL_SHA256 and DEFAULT_MANIFEST are monkeypatched. edit(block) changes the block before it is written.
    Returns (block, pin)."""
    from pilot import protocol as pr
    _, manifest = prepared(tmp_path, synthetic_raw)
    with open(manifest, encoding="utf-8") as f:
        entry = json.load(f)["splits"]["selection_train"]
    with open(os.path.join(os.path.dirname(manifest), entry["path"]), "rb") as f:
        rows = pilot.data.parse_jsonl(f.read())
    tags, ks, neg, rep = ["card"], [1, 2], 1, 2
    d = fs.draw(rows, 0, tags, ks, neg, rep)
    block = {"tags": tags, "k_values": ks, "draw_seeds": [0], "neg_per_pos": neg, "replay": rep,
             "source": {"split": "selection_train", "sha256": entry["sha256"], "uids_sha256": entry["uids_sha256"]},
             "draws": [{"draw": 0, "seed": 0, "base_seed": 0, "support": d["support"], "replay": d["replay"],
                        "positives": d["positives"], "digest": fs.draw_digest(d)}]}
    if edit is not None:
        edit(block)
    protocol = {k: {} for k in pr.REQUIRED_KEYS}
    protocol.update(schema=pr.SCHEMA, fewshot=block)
    path = str(tmp_path / "protocol.json")
    pin = pr.write_protocol(protocol, path)
    monkeypatch.setattr(pr, "PROTOCOL_PATH", path)
    monkeypatch.setattr(pr, "PROTOCOL_SHA256", pin)
    monkeypatch.setattr(pilot.data, "DEFAULT_MANIFEST", manifest)
    return block, pin


def test_fewshot_cli_support_sets_prints_the_frozen_draws(tmp_path: Path, synthetic_raw: SyntheticRaw,
                                                          monkeypatch: pytest.MonkeyPatch,
                                                          capsys: pytest.CaptureFixture[str]) -> None:
    block, pin = frozen_protocol(tmp_path, synthetic_raw, monkeypatch)
    capsys.readouterr()                                         # drop prepare()'s progress lines
    out = runner().main(["--support-sets"])
    printed = capsys.readouterr().out
    assert json.loads(printed) == out
    assert out["protocol_sha256"] == pin and out["source"] == block["source"]
    (d,) = out["draws"]
    want = block["draws"][0]
    assert (d["draw"], d["seed"], d["base_seed"], d["digest"]) == (0, 0, 0, want["digest"])
    assert d["n_support"] == {k: len(v) for k, v in want["support"].items()} and set(d["n_support"]) == {"1", "2"}
    assert d["n_replay"] == 2 and d["positives"] == want["positives"]
    assert "Kartım" not in printed and "bank_" not in printed          # digests and counts only: no text, no uid list


@pytest.mark.parametrize("case", ["missing", "unpinned", "changed"])
def test_fewshot_cli_support_sets_refuses_without_the_frozen_protocol(tmp_path: Path, synthetic_raw: SyntheticRaw,
                                                                      monkeypatch: pytest.MonkeyPatch,
                                                                      case: str) -> None:
    from pilot import protocol as pr
    frozen_protocol(tmp_path, synthetic_raw, monkeypatch)
    if case == "missing":
        os.remove(pr.PROTOCOL_PATH)
    elif case == "unpinned":
        monkeypatch.setattr(pr, "PROTOCOL_SHA256", None)
    else:
        with open(pr.PROTOCOL_PATH, "rb") as f:
            raw = f.read()
        with open(pr.PROTOCOL_PATH, "wb") as f:
            f.write(raw.replace(b"protocol/1", b"protocol/2", 1))   # one byte changed
    cli = runner()

    def never(*args: Any, **kwargs: Any) -> NoReturn:
        raise AssertionError("the split was read before the protocol was checked")
    monkeypatch.setattr(cli, "parse_jsonl", never)
    monkeypatch.setattr(cli, "verify_split_file", never)
    with pytest.raises(SystemExit, match="protocol"):
        cli.main(["--support-sets"])


def test_fewshot_cli_support_sets_refuses_another_source(tmp_path: Path, synthetic_raw: SyntheticRaw,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    def edit(block: dict[str, Any]) -> None:
        block["source"]["sha256"] = "0" * 64
    frozen_protocol(tmp_path, synthetic_raw, monkeypatch, edit)
    with pytest.raises(SystemExit, match="source"):
        runner().main(["--support-sets"])


def test_fewshot_cli_support_sets_refuses_a_changed_draw(tmp_path: Path, synthetic_raw: SyntheticRaw,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    def edit(block: dict[str, Any]) -> None:
        lst = block["draws"][0]["support"]["2"]
        lst[0] = "bank_99_99"                                   # one uid changed
        lst.sort()
    frozen_protocol(tmp_path, synthetic_raw, monkeypatch, edit)
    with pytest.raises(SystemExit, match="draw 0"):
        runner().main(["--support-sets"])


def test_fewshot_cli_needs_a_mode(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as e:
        runner().main([])
    assert e.value.code == 2 and "--support-sets" in capsys.readouterr().err
