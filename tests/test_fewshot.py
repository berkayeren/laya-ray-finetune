"""Contracts of pilot/fewshot.py (REP-01, ROADMAP Phase 8 criterion 3): the few-shot protocol frozen before any
fine-tuned test number exists. Nothing is trained here; Phase 9 executes the protocol unchanged.

- draw: deterministic per seed, nested (the k = 5 support set is the first 5 positives and first 20 negatives of each
  tag of the k = 10 draw), sized by the settings, the replay set disjoint from the k = 10 support set, every list sorted
  and duplicate-free; checked against an independent reading of SAMPLER_RULE written here, not taken from the code.
- the realised positive count of every held-out tag per k, and the digest of DIGEST_RULE.
- every pool smaller than a draw needs is refused, naming the tag and the counts; an unknown or repeated tag and a
  repeated uid are refused.
- pilot.fewshot imports the stdlib only.
- data-marked: the three draws on the verified committed selection_train.
Offline except the `data` tests.
"""
import hashlib
import json
import os
import random
import subprocess
import sys

import pytest

import pilot.data
from intents import INTENTS
from pilot import fewshot as fs

pytestmark = pytest.mark.unit

HEAVY = ("torch", "numpy", "scipy", "sklearn", "laya", "ray", "transformers")
TAGS = ("savings", "loan", "overdraft", "appointment", "contactless", "direct_debit")


def synthetic_rows(n=600, cut=None):
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


def by_rule(rows, seed, tags=TAGS, k_values=(5, 10), neg_per_pos=4, replay=200):
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


def test_draw_is_deterministic():
    rows = synthetic_rows()
    a, b = fs.draw(rows, 0), fs.draw(rows, 0)
    assert a == b and fs.draw_digest(a) == fs.draw_digest(b)
    c = fs.draw(rows, 1)
    assert c["support"] != a["support"] and fs.draw_digest(c) != fs.draw_digest(a)
    assert fs.draw(list(reversed(rows)), 0) == a                  # the row order never matters: pools are sorted uids


def test_draw_is_nested():
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


def test_draw_realised_positives():
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


def test_draw_other_settings():
    rows = synthetic_rows()
    d = fs.draw(rows, 3, tags=("card",), k_values=(1, 2), neg_per_pos=1, replay=2)
    assert (d["support"], d["replay"]) == by_rule(rows, 3, ("card",), (1, 2), 1, 2)
    assert d["tags"] == ["card"] and sorted(d["support"]) == ["1", "2"] and len(d["replay"]) == 2


def test_draw_digest_rule():
    d = fs.draw(synthetic_rows(), 2)
    body = {"schema": d["schema"], "seed": d["seed"], "tags": d["tags"], "support": d["support"], "replay": d["replay"]}
    want = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()
    assert fs.draw_digest(d) == want
    assert fs.draw_digest(dict(d, positives={})) == want                  # positives are derived, not digested
    moved = json.loads(json.dumps(d))
    moved["support"]["10"][0] = "bank_99_99"
    assert fs.draw_digest(moved) != want


def test_draw_positive_pool_too_small_refused():
    rows = synthetic_rows(cut=("loan", 3))
    with pytest.raises(SystemExit) as e:
        fs.draw(rows, 0)
    msg = str(e.value)
    assert "loan" in msg and "3" in msg and "10" in msg


def test_draw_negative_pool_too_small_refused():
    rows = synthetic_rows(n=60)
    for r in rows:
        r["expected"]["card"] = True
    for r in rows[:5]:
        r["expected"]["card"] = False
    with pytest.raises(SystemExit) as e:
        fs.draw(rows, 0, tags=("card",), k_values=(1, 2), neg_per_pos=4, replay=2)
    assert "card" in str(e.value) and "negative" in str(e.value) and "5" in str(e.value)


def test_draw_replay_pool_too_small_refused():
    rows = synthetic_rows(n=240)
    with pytest.raises(SystemExit) as e:
        fs.draw(rows, 0)
    assert "replay" in str(e.value) and "200" in str(e.value)


def test_draw_bad_tags_and_uids_refused():
    rows = synthetic_rows()
    with pytest.raises(SystemExit, match="not_a_tag"):
        fs.draw(rows, 0, tags=("savings", "not_a_tag"))
    with pytest.raises(SystemExit, match="loan"):
        fs.draw(rows, 0, tags=("loan", "savings", "loan"))
    with pytest.raises(SystemExit, match=rows[0]["uid"]):
        fs.draw(rows + [dict(rows[0])], 0)


def test_fewshot_imports_stdlib_only():
    code = "import sys, pilot.fewshot; bad = [m for m in %r if m in sys.modules]; assert not bad, bad" % (HEAVY,)
    env = {k: val for k, val in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "-c", code], cwd=pilot.data.REPO_ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def committed_rows(name, roles):
    path = os.path.join(os.path.dirname(pilot.data.DEFAULT_MANIFEST), "splits", name + ".jsonl")
    with open(path, "rb") as f:
        raw = f.read()
    pilot.data.verify_split_file(path, roles=roles, data=raw)
    return pilot.data.parse_jsonl(raw)


@pytest.mark.data
def test_committed_selection_train_draws():
    rows = committed_rows("selection_train", ("train",))
    assert len(rows) == 1450
    uids = {r["uid"] for r in rows}
    for seed in (0, 1, 2):
        a, b = fs.draw(rows, seed), fs.draw(rows, seed)
        assert a == b and fs.draw_digest(a) == fs.draw_digest(b)
        assert set(a["support"]["10"]) | set(a["replay"]) <= uids
        assert set(a["support"]["5"]) <= set(a["support"]["10"]) and len(a["replay"]) == 200
        assert (a["support"], a["replay"]) == by_rule(rows, seed)
