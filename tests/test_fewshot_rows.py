"""Contracts of the few-shot row builders (FEW-01, FEW-02; 09-01): what every masked base and continued run trains on.

- masked_rows deletes every held-out key from expected (never sets it to False: laya builds a negative item for a False
  value), keeps every message and leaves the input rows unchanged.
- selection(block, draw, k) is the frozen support and replay uid sets of one (draw, k); an unknown draw, a k outside
  the frozen k_values and an overlapping support/replay pair are refused; the k = 5 set is nested in the k = 10 set.
- laya_rows builds a continued run's rows by the frozen laya_item_rule in one random.Random(neg seed) pass in file
  order (LAYA_ROWS_RULE, A2), checked against an independent reading of that rule written here; laya_item_count gives
  the same item count without an RNG; berturk_rows pairs support rows with no mask and replay rows with the held-out
  tags masked.
- every builder refuses an empty or unknown tag list and a selected uid that is missing from the rows or repeated;
  tags and uids compare as exact strings.
- check_source binds rows to the frozen source (sha256, uids_sha256, role train) and re-checks every frozen draw.
- fewshot.py --counts: require_frozen first (no split read without the frozen protocol), a missing manifest refused
  in one line (08-REVIEW IN-07), the printed JSON equal to fewshot.counts.
- data-marked: the exact counts on the committed selection_train and protocol.json (14,580 masked-base Laya items,
  456 Laya steps and 46 BERTurk steps per epoch; 4,298/6,475, 4,268/6,414, 4,328/6,492 continued Laya items).
Offline except the `data` tests.
"""
import json
import math
import os
import random

import pytest

import pilot.data
from fewshot_synthetic import synthetic_protocol
from intents import INTENTS
from pilot import fewshot as fs
from pilot import protocol as pr
from pilot import runs, spike

pytestmark = pytest.mark.unit

HELD = ("savings", "loan", "overdraft", "appointment", "contactless", "direct_debit")


def full_rows(n=40, seed=3):
    """n rows labelling all 48 tags: row i carries 1 + i % 4 random positives (held-out tags included now and then),
    uids u<i> in a shuffled file order."""
    rng, rows = random.Random(seed), []
    for i in range(n):
        pos = set(rng.sample(list(INTENTS), 1 + i % 4))
        if i % 5 == 0:
            pos.add(HELD[i % len(HELD)])
        rows.append({"state": "mesaj %d" % i, "uid": "u%02d" % i, "questions": {},
                     "expected": {t: t in pos for t in INTENTS}})
    rng.shuffle(rows)
    return rows


def sel_of(rows, support, replay):
    return {"support": frozenset(rows[i]["uid"] for i in support), "replay": frozenset(rows[i]["uid"] for i in replay)}


def by_rule(rows, sel, tags, seed, neg):
    """LAYA_ROWS_RULE read independently: one random.Random(seed); rows in file order; per selected row the old-tag
    positives, then a sample of min(neg, n) from the sorted old-tag negatives, then (support only) every held-out
    label."""
    rng, out, held = random.Random(seed), [], set(tags)
    for r in rows:
        if r["uid"] not in sel["support"] | sel["replay"]:
            continue
        e = r["expected"]
        keep = [t for t, v in e.items() if v and t not in held]
        neg_pool = sorted(t for t, v in e.items() if not v and t not in held)
        keep += rng.sample(neg_pool, min(neg, len(neg_pool)))
        if r["uid"] in sel["support"]:
            keep += list(tags)
        out.append({t: e[t] for t in keep})
    return out


# ---- masked_rows (FEW-01) ----

def test_masked_rows_deletes_the_held_out_keys_and_keeps_every_message():
    rows = full_rows()
    before = json.dumps(rows, sort_keys=True)
    out = fs.masked_rows(rows, HELD)
    assert json.dumps(rows, sort_keys=True) == before                     # input not mutated
    assert [r["uid"] for r in out] == [r["uid"] for r in rows]          # every message kept, in order
    for r, m in zip(rows, out):
        assert not set(HELD) & set(m["expected"])                       # deleted, never set to False
        assert m["expected"] == {t: v for t, v in r["expected"].items() if t not in HELD}
        assert len(m["expected"]) == 42 and m["state"] == r["state"]


@pytest.mark.parametrize("tags", [(), ["Savings"], ["savings", "savings"], ["not_a_tag"]])
def test_every_builder_refuses_an_empty_or_unknown_tag_list(tags):
    rows = full_rows()
    sel = sel_of(rows, [0, 1], [2, 3])
    for call in (lambda: fs.masked_rows(rows, tags), lambda: fs.laya_rows(rows, sel, tags, 0, 8),
                 lambda: fs.laya_item_count(rows, sel, tags, 8), lambda: fs.berturk_rows(rows, sel, tags)):
        with pytest.raises(SystemExit, match="tag"):
            call()


# ---- laya_rows, laya_item_count, berturk_rows (FEW-02, A2) ----

def test_laya_rows_follow_the_frozen_item_rule():
    rows = full_rows()
    sel = sel_of(rows, [0, 2, 4, 6], [1, 3, 5])
    out = fs.laya_rows(rows, sel, HELD, 0, 8)
    assert [r["expected"] for r in out] == by_rule(rows, sel, HELD, 0, 8)
    assert out == fs.laya_rows(rows, sel, HELD, 0, 8)                   # deterministic per seed
    order = [r["uid"] for r in rows if r["uid"] in sel["support"] | sel["replay"]]
    assert [r["uid"] for r in out] == order                             # file order, unselected rows skipped
    for r in out:
        e, src = r["expected"], next(x for x in rows if x["uid"] == r["uid"])["expected"]
        old_pos = [t for t, v in src.items() if v and t not in HELD]
        old_neg = [t for t in e if t not in HELD and not e[t]]
        assert all(e[t] for t in old_pos) and len(old_neg) == min(8, 42 - len(old_pos))
        if r["uid"] in sel["support"]:
            assert all(t in e and e[t] == src[t] for t in HELD)         # every held-out label, positive or negative
        else:
            assert not set(HELD) & set(e)                               # a replay row keeps no held-out key
    assert sum(len(r["expected"]) for r in out) == fs.laya_item_count(rows, sel, HELD, 8)
    assert "random.Random" in fs.LAYA_ROWS_RULE and "file order" in fs.LAYA_ROWS_RULE


def test_laya_rows_seed_changes_only_the_sampled_negatives():
    rows = full_rows()
    sel = sel_of(rows, [0, 2], [1, 3])
    a, b = fs.laya_rows(rows, sel, HELD, 0, 8), fs.laya_rows(rows, sel, HELD, 1, 8)
    assert [r["uid"] for r in a] == [r["uid"] for r in b] and a != b


def test_berturk_rows_mask_the_held_out_tags_on_replay_rows_only():
    rows = full_rows()
    sel = sel_of(rows, [0, 2], [1, 3, 5])
    out = fs.berturk_rows(rows, sel, HELD)
    order = [r for r in rows if r["uid"] in sel["support"] | sel["replay"]]
    assert [r for r, _ in out] == order                                 # the full rows, in file order
    for r, mask in out:
        assert mask == (() if r["uid"] in sel["support"] else tuple(sorted(HELD)))


@pytest.mark.parametrize("case", ["missing", "repeated", "case"])
def test_builders_refuse_a_selected_uid_missing_or_repeated(case):
    rows = full_rows()
    sel = sel_of(rows, [0, 1], [2, 3])
    if case == "missing":
        sel = dict(sel, replay=sel["replay"] | {"u99"})
    elif case == "case":
        sel = dict(sel, replay=sel["replay"] | {rows[5]["uid"].upper()})   # exact strings: U05 is not u05
    else:
        rows = rows + [dict(rows[0])]
    for call in (lambda: fs.laya_rows(rows, sel, HELD, 0, 8), lambda: fs.laya_item_count(rows, sel, HELD, 8),
                 lambda: fs.berturk_rows(rows, sel, HELD)):
        with pytest.raises(SystemExit, match="uid"):
            call()


def test_builders_refuse_an_overlapping_selection():
    rows = full_rows()
    sel = sel_of(rows, [0, 1], [1, 2])
    for call in (lambda: fs.laya_rows(rows, sel, HELD, 0, 8), lambda: fs.berturk_rows(rows, sel, HELD)):
        with pytest.raises(SystemExit, match="support"):
            call()


# ---- selection and check_source on the synthetic frozen protocol ----

def test_selection_is_the_frozen_lists_nested_by_k(tmp_path, synthetic_raw, monkeypatch):
    ns = synthetic_protocol(tmp_path, synthetic_raw, monkeypatch)
    d = ns.block["draws"][0]
    s1, s2 = fs.selection(ns.block, 0, 1), fs.selection(ns.block, 0, 2)
    assert s1 == {"support": frozenset(d["support"]["1"]), "replay": frozenset(d["replay"])}
    assert s1["support"] <= s2["support"] and s1["replay"] == s2["replay"]
    assert not s2["support"] & s2["replay"]


@pytest.mark.parametrize("draw,k", [(1, 1), (-1, 1), (0, 3), (0, "1"), (0, True), (True, 1)])
def test_selection_refuses_an_unknown_draw_or_k(tmp_path, synthetic_raw, monkeypatch, draw, k):
    ns = synthetic_protocol(tmp_path, synthetic_raw, monkeypatch)
    with pytest.raises(SystemExit, match="draw|k "):
        fs.selection(ns.block, draw, k)


def test_selection_refuses_an_overlapping_support_and_replay(tmp_path, synthetic_raw, monkeypatch):
    ns = synthetic_protocol(tmp_path, synthetic_raw, monkeypatch)
    block = ns.block
    block["draws"][0]["replay"] = sorted(set(block["draws"][0]["replay"]) | {block["draws"][0]["support"]["1"][0]})
    with pytest.raises(SystemExit, match="support"):
        fs.selection(block, 0, 1)


def test_check_source_binds_the_frozen_source(tmp_path, synthetic_raw, monkeypatch):
    ns = synthetic_protocol(tmp_path, synthetic_raw, monkeypatch)
    with open(ns.manifest, encoding="utf-8") as f:
        entry = json.load(f)["splits"]["selection_train"]
    assert fs.check_source(ns.block, entry, ns.rows) is None
    for key, value in (("sha256", "0" * 64), ("uids_sha256", "1" * 64), ("role", "val")):
        with pytest.raises(SystemExit, match=key if key != "role" else "role"):
            fs.check_source(ns.block, dict(entry, **{key: value}), ns.rows)
    with pytest.raises(SystemExit, match="draw 0|source"):
        fs.check_source(ns.block, entry, ns.rows[1:])                    # not the frozen rows
    other = dict(ns.block, source=dict(ns.block["source"], split="final_train"))
    with pytest.raises(SystemExit, match="selection_train"):
        fs.check_source(other, entry, ns.rows)


# ---- fewshot.py --counts ----

def runner():
    import fewshot
    return fewshot


def test_fewshot_counts_cli(tmp_path, synthetic_raw, monkeypatch, capsys):
    ns = synthetic_protocol(tmp_path, synthetic_raw, monkeypatch)
    capsys.readouterr()
    out = runner().main(["--counts"])
    printed = capsys.readouterr().out
    assert json.loads(printed) == out
    want = runner().counts(ns.block, ns.rows)
    want["protocol_sha256"] = ns.pin
    assert out == want
    assert out["base"]["messages"] == len(ns.rows)
    assert [(c["draw"], c["k"]) for c in out["continued"]] == [(0, 1), (0, 2)]
    sel = fs.selection(ns.block, 0, 2)
    c = out["continued"][1]
    assert (c["support_rows"], c["replay_rows"]) == (len(sel["support"]), len(sel["replay"]))
    assert c["berturk_messages"] == len(sel["support"]) + len(sel["replay"])
    assert c["berturk_steps_per_epoch"] == math.ceil(c["berturk_messages"] / 32)
    assert c["laya_steps_per_epoch"] == spike.optimizer_steps(c["laya_items"], 8, 4)
    assert out["laya_rows_rule"] == fs.LAYA_ROWS_RULE
    assert "Kartım" not in printed and "bank_" not in printed          # counts only: no text, no uid list


@pytest.mark.parametrize("case", ["missing", "unpinned", "changed"])
def test_fewshot_counts_refuses_without_the_frozen_protocol(tmp_path, synthetic_raw, monkeypatch, case):
    synthetic_protocol(tmp_path, synthetic_raw, monkeypatch)
    if case == "missing":
        os.remove(pr.PROTOCOL_PATH)
    elif case == "unpinned":
        monkeypatch.setattr(pr, "PROTOCOL_SHA256", None)
    else:
        with open(pr.PROTOCOL_PATH, "rb") as f:
            raw = f.read()
        with open(pr.PROTOCOL_PATH, "wb") as f:
            f.write(raw.replace(b"protocol/1", b"protocol/2", 1))
    cli = runner()

    def never(*args, **kwargs):
        raise AssertionError("the split was read before the protocol was checked")
    monkeypatch.setattr(cli, "parse_jsonl", never)
    monkeypatch.setattr(cli, "verify_split_file", never)
    with pytest.raises(SystemExit, match="protocol"):
        cli.main(["--counts"])


def test_fewshot_counts_refuses_a_missing_manifest(tmp_path, synthetic_raw, monkeypatch):
    synthetic_protocol(tmp_path, synthetic_raw, monkeypatch)
    path = str(tmp_path / "nowhere" / "splits.json")
    with pytest.raises(SystemExit) as e:
        runner().main(["--counts", "--manifest", path])
    msg = str(e.value)
    assert path in msg and "make data" in msg and "\n" not in msg


def test_fewshot_cli_names_both_modes(capsys):
    with pytest.raises(SystemExit) as e:
        runner().main([])
    err = capsys.readouterr().err
    assert e.value.code == 2 and "--support-sets" in err and "--counts" in err


# ---- the committed data and protocol.json ----

COUNTS = [(0, 5, 143, 4298, 343), (0, 10, 279, 6475, 479), (1, 5, 142, 4268, 342), (1, 10, 274, 6414, 474),
          (2, 5, 144, 4328, 344), (2, 10, 278, 6492, 478)]


def committed():
    proto, pin = pr.require_frozen()
    rows = runner().read_source(proto["fewshot"], pilot.data.DEFAULT_MANIFEST)
    return proto["fewshot"], pin, rows


@pytest.mark.data
def test_counts_on_the_committed_selection_train():
    block, _, rows = committed()
    tags = block["tags"]
    assert tuple(tags) == HELD and len(rows) == 1450
    masked = fs.masked_rows(rows, tags)
    assert runs.neg_items(masked, 8) == 14580 and spike.optimizer_steps(14580, 8, 4) == 456
    assert math.ceil(1450 / 32) == 46
    for draw, k, n_support, items, messages in COUNTS:
        sel = fs.selection(block, draw, k)
        assert (len(sel["support"]), len(sel["replay"])) == (n_support, 200)
        assert fs.laya_item_count(rows, sel, tags, 8) == items
        built = fs.laya_rows(rows, sel, tags, block["draws"][draw]["seed"], 8)
        assert sum(len(r["expected"]) for r in built) == items
        assert len(fs.berturk_rows(rows, sel, tags)) == messages
        assert math.ceil(messages / 32) == (11 if k == 5 else 15)
    for draw in range(3):
        assert fs.selection(block, draw, 5)["support"] <= fs.selection(block, draw, 10)["support"]


@pytest.mark.data
def test_fewshot_counts_cli_on_the_committed_protocol(capsys):
    block, pin, rows = committed()
    out = runner().main(["--counts"])
    assert json.loads(capsys.readouterr().out) == out and out["protocol_sha256"] == pin
    b = out["base"]
    assert (b["messages"], b["laya_items"], b["laya_steps_per_epoch"], b["berturk_steps_per_epoch"]) == \
        (1450, 14580, 456, 46)
    assert [(c["draw"], c["k"], c["support_rows"], c["laya_items"], c["berturk_messages"])
            for c in out["continued"]] == COUNTS
    assert all(c["replay_rows"] == 200 for c in out["continued"])
