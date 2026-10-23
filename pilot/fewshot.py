"""Frozen few-shot protocol (REP-01, ROADMAP Phase 8 criterion 3): settings, support-set sampler, eligibility check.

REP-01 requires the few-shot protocol (held-out tags, k values, draws, LR, epochs, metric and replay) "frozen in the
same file before any test scoring"; this module is that protocol as code and data, and 08-05 writes protocol_block()
into protocol.json's "fewshot" block. Nothing is trained here. Phase 9 executes the protocol unchanged and reads the
frozen support-set uid lists from protocol.json instead of regenerating them; check_draws only proves that a
regeneration still gives the frozen lists and digests.

What it fixes (08-RESEARCH A7-A13, taken under the user's standing delegation of 2026-10-21):
- TAGS: the 6 held-out tags of FEATURES TS-14, listed on 2026-10-13 before any model existed. Eligibility
  (banking-only, enough test and final_train positives) is a check of that list, never the selector.
- K_VALUES (5, 10), nested: the k = 5 support set is the first 5 positives and first 20 negatives of each tag drawn
  for k = 10. NEG_PER_POS 4 negatives per positive, from selection_train messages without the tag. Every support
  message keeps its full gold labels for all 48 tags; the support set is the union over tags.
- REPLAY 200 selection_train messages outside the draw's k = 10 support set (old-tag labels only).
- DRAW_SEEDS (0, 1, 2): draw d uses random.Random(d) over sorted uid pools (SAMPLER_RULE) and continues from the
  masked base run of seed d; each draw is bound by the sha256 of its canonical JSON (DIGEST_RULE).
- EPOCHS 10 of continued training at the base LRs.

Stdlib only at import time, like pilot/data.py: it never imports torch, numpy, scipy, laya or Ray. QUESTIONS is
never defined here (intents.py is its only definition).

usage:
    from pilot import fewshot
    d = fewshot.draw(rows, seed)              # rows: parsed, verified selection_train
    fewshot.draw_digest(d)
"""
import hashlib
import json
import random

from intents import INTENTS

TAGS = ("savings", "loan", "overdraft", "appointment", "contactless", "direct_debit")   # FEATURES TS-14, A7
K_VALUES = (5, 10)
DRAW_SEEDS = (0, 1, 2)
NEG_PER_POS = 4
REPLAY = 200
EPOCHS = 10
DRAW_SCHEMA = "fewshot-draw/1"
SAMPLER_RULE = ("rows sorted by uid; one random.Random(seed) per draw (CPython 3.12 random); for each tag in TAGS "
                "order: rng.sample of max(k) positives from the sorted uids carrying the tag, then rng.sample of "
                "NEG_PER_POS x max(k) negatives from the sorted uids without it; the k support set is the union over "
                "tags of the first k positives and the first NEG_PER_POS x k negatives (in sample order), so the "
                "k = 5 set is nested in the k = 10 set; replay = rng.sample of REPLAY from the sorted uids outside "
                "the max-k support set; every stored list is sorted; a pool smaller than a draw needs is refused, "
                "never shortened")
DIGEST_RULE = ("sha256 of the compact (separators ',' and ':'), key-sorted, UTF-8 JSON of {\"schema\", \"seed\", "
               "\"tags\", \"support\": {k: sorted uids}, \"replay\": sorted uids}; k keys are strings")


def tag_counts(rows, tags):
    """{tag: number of rows whose expected[tag] is true}, in `tags` order (gold labels only)."""
    return {t: sum(1 for r in rows if r["expected"].get(t, False)) for t in tags}


def _check_tags(tags):
    seen = set()
    for t in tags:
        if t not in INTENTS:
            raise SystemExit("few-shot tag %r is not an intents.py tag" % (t,))
        if t in seen:
            raise SystemExit("few-shot tag %s is listed twice" % t)
        seen.add(t)


def _sample(rng, pool, n, what):
    if len(pool) < n:
        raise SystemExit("few-shot draw needs %d %s but the pool holds %d; the protocol never shortens a draw"
                         % (n, what, len(pool)))
    return rng.sample(pool, n)


def draw(rows, seed, tags=TAGS, k_values=K_VALUES, neg_per_pos=NEG_PER_POS, replay=REPLAY):
    """One support-set draw by SAMPLER_RULE: {"schema", "seed", "tags", "support": {str(k): sorted uids},
    "replay": sorted uids, "positives": {str(k): {tag: support messages carrying it}}}. The realised positive count
    can exceed k through co-occurrence. SystemExit on an unknown or repeated tag, a repeated uid, or a positive,
    negative or replay pool smaller than the draw needs (naming the tag, the need and the pool)."""
    tags, k_values = list(tags), sorted(k_values)
    _check_tags(tags)
    if not k_values or len(set(k_values)) != len(k_values) or any(not isinstance(k, int) or k < 1 for k in k_values):
        raise SystemExit("few-shot k values must be distinct positive integers, got %r" % (k_values,))
    lab, uids = {}, []
    for r in rows:
        if r["uid"] in lab:
            raise SystemExit("few-shot rows repeat uid %s; draw from one verified split" % r["uid"])
        lab[r["uid"]] = r["expected"]
        uids.append(r["uid"])
    uids.sort()
    rng, kmax, picks = random.Random(seed), k_values[-1], {}
    for t in tags:
        pos = [u for u in uids if lab[u].get(t, False)]
        neg = [u for u in uids if not lab[u].get(t, False)]
        picks[t] = (_sample(rng, pos, kmax, "positives of %s" % t),
                    _sample(rng, neg, neg_per_pos * kmax, "negatives of %s (messages without it)" % t))
    support = {str(k): sorted({u for t in tags for u in picks[t][0][:k] + picks[t][1][:neg_per_pos * k]})
               for k in k_values}
    top = set(support[str(kmax)])
    rep = sorted(_sample(rng, [u for u in uids if u not in top], replay,
                         "replay messages outside the k = %d support set" % kmax))
    positives = {k: {t: sum(1 for u in s if lab[u].get(t, False)) for t in tags} for k, s in support.items()}
    return {"schema": DRAW_SCHEMA, "seed": seed, "tags": tags, "support": support, "replay": rep,
            "positives": positives}


def draw_digest(d):
    """DIGEST_RULE of a draw (its realised positive counts are derived, so not digested)."""
    body = {"schema": d["schema"], "seed": d["seed"], "tags": d["tags"], "support": d["support"],
            "replay": d["replay"]}
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()
