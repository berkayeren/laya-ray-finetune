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

protocol_block() reads selection_train, final_train and test from verified bytes of the committed data/splits.json
(test and final_train for gold label counts only, no model output), checks the tags' eligibility, draws the three
support sets and returns the complete block: settings, eligibility, source, sampler, draws and the rule texts
(RULES: base runs, masking, the new question, support labels, replay, continued training, the Laya item rule, the
primary and secondary metrics, the compute fallback, the gate, test admission and the Phase 9 notes).

usage:
    from pilot import fewshot
    d = fewshot.draw(rows, seed)              # rows: parsed, verified selection_train
    fewshot.draw_digest(d)
    block = fewshot.protocol_block()          # the "fewshot" block of protocol.json (08-05)
    fewshot.check_draws(block, rows)          # SystemExit unless a regeneration gives the frozen lists and digests
"""
import hashlib
import json
import os
import random

from intents import INTENTS
from pilot import pins
from pilot.data import (DEFAULT_MANIFEST, REPO_ROOT, manifest_is_committed, parse_jsonl, sha256_bytes,
                        verify_split_file)

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
# a check of the TS-14 list, never its selector (A7): every held-out tag must have a banking-only gold on HOTELS and
# enough positives for a stable test AP (test) and for the draws (final_train holds selection_train's folds)
ELIGIBILITY = {"banking_only": "every held-out tag is in pilot.pins.OOD_SCORING_TAGS", "min_test_positives": 10,
               "min_final_train_positives": 40}
# A8: the label-masked base runs, on selection_train at each model's chosen recipe (results/postfit/<kind>.json:
# Laya epoch 3 of 4, BERTurk epoch 29 of 30), one per draw seed; literals so the module stays stdlib only, and
# tests/test_fewshot.py checks them against the committed records, pilot.postfit and pilot.berturk
BASE_RUNS = {
    "split": "selection_train", "seeds": [0, 1, 2],
    "laya": {"trainer": "ray_train.py", "epochs": 4, "stop_after_epoch": 3, "neg": 8, "encoder_lr": 2e-05,
             "head_lr": 0.0001, "device": "mps", "workers": 1, "micro_batch": 8, "grad_accum": 4, "max_len": 128},
    "berturk": {"trainer": "berturk_train.py", "epochs": 30, "stop_after_epoch": 29, "encoder_lr": 2e-05,
                "head_lr": 0.0001, "batch": 32, "max_len": 128, "device": "mps", "dtype": "float32",
                "mask_label": -100.0},
}
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
    can exceed k through co-occurrence and because another tag's negatives may carry the tag. SystemExit on an unknown or repeated tag, a repeated uid, or a positive,
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


RULES = {
    "base_runs": dict(BASE_RUNS, rule=(
        "one label-masked base run per model and seed on selection_train (folds 0-13, so val stays held out for the "
        "bases' θ) at the model's chosen recipe: Laya ray_train.py --epochs 4 --stop-after-epoch 3, neg 8, encoder LR "
        "2e-05, head LR 1e-4, mps, one worker; BERTurk berturk_train.py --epochs 30 --stop-after-epoch 29, LR 2e-05, "
        "head LR 1e-4, batch 32, mps fp32; seeds 0, 1, 2, base seed d paired with draw d; the same recipe as the main "
        "experiment, so the comparison isolates the masking")),
    "masking": ("every selection_train message is kept; the six held-out tags are removed from each row's expected "
                "before neg8 sampling for Laya (laya.train.items_from_rows skips a question with no expected entry, "
                "so the base sees no item of them) and set to pilot.berturk.MASK_LABEL -100 for BERTurk (masked BCE: "
                "zero gradient); the masked base sees no label of the six tags (Phase 9 proves it: base items per "
                "held-out question = 0, unmasked BERTurk labels in those columns = 0)"),
    "new_question": {
        "laya": ("the masked tag's existing intents.py question, QUESTIONS[tag] ({\"type\": \"noul\", "
                 "\"instructions\": INTENTS[tag]}), never reworded: questions_sha256 is frozen in data/splits.json "
                 "and a wording change refreezes every split; the base never sees it because items_from_rows skips "
                 "a question without an expected entry"),
        "berturk": ("the base's own classifier row for the tag, trained on no label (masked BCE gives it zero "
                    "gradient; weight decay only); no weight imprinting (XCMP-04 is v2)"),
    },
    "support_labels": ("every support message carries its full gold labels for all 48 tags; the support set of a k "
                       "is the union over the six tags of their k positives and 4k negatives; the realised positive "
                       "count of each new tag per k is reported: it can exceed k, through co-occurrence and because a "
                       "negative drawn for one tag may carry another held-out tag (negatives lack only their own tag)"),
    "replay": ("200 selection_train messages outside the draw's k = 10 support set (the same replay set for k = 5 "
               "and k = 10), with old-tag labels only: the six new tags are masked as in the base runs"),
    "continued_training": {"epochs": EPOCHS, "encoder_lr": 2e-05, "head_lr": 0.0001, "rule": (
        "10 epochs from the masked base of seed = draw seed at the base LRs (encoder 2e-05, head 1e-4), on the "
        "support set plus the replay set, each trainer's own schedule over the 10 epochs (Laya cosine and sigma "
        "anneal, BERTurk linear with 10% warm-up), training seed = draw seed, no early stopping, no selection on "
        "val or test; epoch parity, not step parity, disclosed: BERTurk takes about 160 optimizer steps against "
        "Laya's about 2,000")},
    "laya_item_rule": ("support rows: every positive, every label of the six new tags (positive or negative) and 8 "
                       "random old-tag negatives; replay rows: neg8 over the 42 old tags; plain neg8 would drop most "
                       "new-tag negatives; the asymmetry with BERTurk's full labels is disclosed as in the main "
                       "experiment (neg8 against all 48 labels)"),
    "primary_metric": {"k": 10, "rule": (
        "per new tag, average precision on test by pilot.metrics.AP_RULE on P = sigmoid(logit) (T-invariant: raw "
        "T = 1); macro-AP = mean over the six tags; the aggregate is the mean over the three draws; criterion (d) "
        "reads the mean over draws of (Laya - BERTurk) macro-AP at k = 10; k = 5 is reported")},
    "secondary_metrics": [
        "k = 0 new-tag macro-AP of each masked base (Laya answers the new question zero-shot; BERTurk's untrained row)",
        "new-tag micro-F1 and 42-old-tag micro-F1 at the masked base's val θ (the val rule at T = 1)",
        "forgetting: old-tag micro-F1 of the continued model minus the masked base's",
        "a paired message-level bootstrap CI of the new-tag macro-AP delta (descriptive only)",
        "incremental wall clock per continued-training run",
    ],
    "compute_fallback": {"base_runs_per_model": 1, "seed": 0, "rule": (
        "1 masked base run per model (seed 0), all three draws continued from it"), "trigger": (
        "the Phase 9 projection from the results/compute rates exceeds 10 h, or a primary base run still fails "
        "after the LAYA-03 ladder"), "disclosure": "the results state that base-model variance is not captured"},
    "gate": ("the few-shot runner calls pilot.protocol.require_frozen() before anything else and records "
             "protocol_sha256 in every Phase 9 record"),
    "test_admission": ("a few-shot checkpoint reaches test only through the score.py protocol gate, once Phase 9 "
                       "binds its run record to this protocol_sha256 and a draw digest; Phase 8's allow-list holds "
                       "the six final seeds only"),
    "phase_9_notes": [
        "ray_train.py has no tag masking: add --mask-tags (an IDENTITY key dropped when None), applied to expected "
        "before spike.neg8_rows",
        "berturk_train.py refuses continued training from a fine-tuned head (optimizer_groups requires a newly "
        "initialised head): add a continued-training mode",
        "support sets are not registered splits: both trainers need a protocol-derived support-set source over the "
        "verified selection_train bytes; data/splits.json is never refrozen",
    ],
}


def check_eligibility(tags, counts, ood_tags, rule=ELIGIBILITY):
    """{"rule", "counts": {"test": {tag: n}, "final_train": {tag: n}}} when every tag is banking-only (in `ood_tags`)
    and has at least the rule's test and final_train positives; SystemExit naming the tag otherwise. A check of the
    pre-results list, never a selector."""
    for t in tags:
        if t not in ood_tags:
            raise SystemExit("few-shot tag %s is not banking-only (pilot.pins.OOD_SCORING_TAGS); the held-out tags "
                             "are the TS-14 list and eligibility only checks it" % t)
        for split, key in (("test", "min_test_positives"), ("final_train", "min_final_train_positives")):
            n = counts.get(split, {}).get(t)
            if n is None:
                raise SystemExit("few-shot tag %s has no %s positive count" % (t, split))
            if n < rule[key]:
                raise SystemExit("few-shot tag %s has %d %s positives, fewer than the %d eligibility needs"
                                 % (t, n, split, rule[key]))
    return {"rule": dict(rule), "counts": {s: {t: counts[s][t] for t in tags} for s in ("test", "final_train")}}


def check_draws(block, rows):
    """None when every draw of `block` regenerates from `rows` (the verified source split, in file order) with the
    block's settings and seeds to exactly its uid lists, realised positives and digest, and the rows are the block's
    source; SystemExit naming the draw and what differs otherwise."""
    for key in ("tags", "k_values", "draw_seeds", "neg_per_pos", "replay", "source", "draws"):
        if key not in block:
            raise SystemExit("the few-shot block has no %s" % key)
    seeds, draws = block["draw_seeds"], block["draws"]
    if len(draws) != len(seeds):
        raise SystemExit("the few-shot block holds %d draws for %d draw seeds" % (len(draws), len(seeds)))
    for i, (s, e) in enumerate(zip(seeds, draws)):
        name = "draw %d (seed %s)" % (i, s)
        if (e.get("draw"), e.get("seed"), e.get("base_seed")) != (i, s, s):
            raise SystemExit("%s is numbered draw %r, seed %r, base seed %r; draw i uses the i-th draw seed and "
                             "continues from the base run of that seed" % (name, e.get("draw"), e.get("seed"),
                                                                           e.get("base_seed")))
        try:
            d = draw(rows, s, block["tags"], block["k_values"], block["neg_per_pos"], block["replay"])
        except SystemExit as x:
            raise SystemExit("%s: %s" % (name, x))
        for key in ("support", "replay", "positives"):
            if e.get(key) != d[key]:
                raise SystemExit("%s: its %s differs from a regeneration on these rows (a changed list, rows of "
                                 "another split, or a changed sampler); Phase 9 trains on the frozen lists only"
                                 % (name, key))
        want = draw_digest(d)
        if e.get("digest") != want:
            raise SystemExit("%s: digest %s != %s regenerated (DIGEST_RULE)" % (name, e.get("digest"), want))
    have = sha256_bytes("\n".join(r["uid"] for r in rows).encode("utf-8"))
    if have != block["source"].get("uids_sha256"):
        raise SystemExit("the rows are not the block's source split %s (uids_sha256 %s != %s); pass the verified "
                         "split's rows in file order" % (block["source"].get("split"), have,
                                                         block["source"].get("uids_sha256")))


# ---- Phase 9: the rows every few-shot run trains on (FEW-01, FEW-02; 09-01) ----

SOURCE_SPLIT = "selection_train"
# A2 (09-RESEARCH): the frozen laya_item_rule names no RNG; this is the reading Phase 9 takes, disclosed in its
# records. A module text, never part of RULES, so protocol.json's fewshot block is unchanged.
LAYA_ROWS_RULE = ("a continued run's Laya rows: one random.Random(neg seed) pass over the support and replay rows in "
                  "selection_train file order (neg seed = training seed = draw seed); per row the old-tag positives, "
                  "then rng.sample(sorted old-tag negatives, min(neg, n)) (pilot.spike.neg8_rows's shape), then, for "
                  "a support row only, every label of the held-out tags (positive or negative); a replay row keeps no "
                  "held-out key")


def _held(tags):
    """The held-out tags as a list after _check_tags; SystemExit on an empty list (never a no-op mask)."""
    if isinstance(tags, str):
        raise SystemExit("few-shot tags must be a list of intents.py tags, got the string %r" % (tags,))
    tags = list(tags or ())
    if not tags:
        raise SystemExit("the few-shot held-out tag list is empty; pass the frozen protocol's tags")
    _check_tags(tags)
    return tags


def masked_rows(rows, tags):
    """New rows with every held-out tag's key deleted from expected (FEW-01): laya builds a negative item for a False
    value, so a key is deleted, never set to False. Every row is kept, in order; the input rows are not mutated. Apply
    before any negative sampling."""
    held = set(_held(tags))
    return [dict(r, expected={t: v for t, v in r["expected"].items() if t not in held}) for r in rows]


def selection(block, draw, k):
    """{"support": frozenset of uids, "replay": frozenset of uids} of the frozen draw `draw` at `k` (D-02: the frozen
    lists, never regenerated); SystemExit on a draw index outside the block's draws, a k outside its k_values, a
    repeated uid, or a support set that intersects the replay set."""
    draws = block.get("draws") or []
    if isinstance(draw, bool) or not isinstance(draw, int) or not 0 <= draw < len(draws):
        raise SystemExit("few-shot draw %r is not one of the frozen draws 0..%d; Phase 9 runs the frozen protocol "
                         "unchanged" % (draw, len(draws) - 1))
    ks = block.get("k_values") or []
    if isinstance(k, bool) or not isinstance(k, int) or k not in ks:
        raise SystemExit("few-shot k %r is not one of the frozen k values %s; Phase 9 runs the frozen protocol "
                         "unchanged" % (k, ks))
    d = draws[draw]
    if d.get("draw") != draw:
        raise SystemExit("the frozen draw at index %d is numbered %r" % (draw, d.get("draw")))
    sup, rep = (d.get("support") or {}).get(str(k)), d.get("replay")
    if not isinstance(sup, list) or not isinstance(rep, list):
        raise SystemExit("frozen draw %d holds no support list for k %d or no replay list" % (draw, k))
    support, replay = frozenset(sup), frozenset(rep)
    if len(support) != len(sup) or len(replay) != len(rep):
        raise SystemExit("frozen draw %d repeats a uid in its k = %d support or replay list" % (draw, k))
    _check_disjoint({"support": support, "replay": replay}, "frozen draw %d at k = %d" % (draw, k))
    return {"support": support, "replay": replay}


def _check_disjoint(sel, name):
    both = sel["support"] & sel["replay"]
    if both:
        raise SystemExit("%s: %d uids are in both the support and the replay set (first %s); the replay set lies "
                         "outside the support set" % (name, len(both), sorted(both)[0]))


def _selected(rows, sel):
    """[(row, is_support)] of the selected rows in file order; SystemExit on an overlapping selection, or a selected
    uid that is missing from the rows or appears twice (never a shortened row set). uids compare as exact strings."""
    _check_disjoint(sel, "the few-shot selection")
    want = sel["support"] | sel["replay"]
    seen, out = set(), []
    for r in rows:
        u = r["uid"]
        if u not in want:
            continue
        if u in seen:
            raise SystemExit("selected uid %s appears twice in the rows; build from one verified selection_train"
                             % u)
        seen.add(u)
        out.append((r, u in sel["support"]))
    missing = want - seen
    if missing:
        raise SystemExit("%d selected uids are not in the rows (first %s); the frozen lists come from the verified "
                         "selection_train bytes" % (len(missing), sorted(missing)[0]))
    return out


def _check_neg(neg):
    if isinstance(neg, bool) or not isinstance(neg, int) or neg < 1:
        raise SystemExit("few-shot neg must be a positive count (the frozen neg 8), got %r" % (neg,))


def laya_rows(rows, sel, tags, seed, neg):
    """The rows of a continued Laya run by the frozen laya_item_rule, read as LAYA_ROWS_RULE (A2): support rows keep
    every positive, every label of `tags` and min(neg, n) random old-tag negatives; replay rows keep no held-out key,
    their old-tag positives and min(neg, n) random old-tag negatives; one random.Random(seed) pass in file order; rows
    outside `sel` are skipped. SystemExit on a bad tag list, a selected uid missing or repeated, or a support row
    without a held-out label."""
    tags = _held(tags)
    _check_neg(neg)
    held, rng, out = set(tags), random.Random(seed), []
    for r, support in _selected(rows, sel):
        e = r["expected"]
        keep = [t for t, v in e.items() if v and t not in held]
        old_neg = sorted(t for t, v in e.items() if not v and t not in held)
        keep += rng.sample(old_neg, min(neg, len(old_neg)))
        if support:
            lack = [t for t in tags if t not in e]
            if lack:
                raise SystemExit("support row %s has no label for held-out tag %s; every support message carries its "
                                 "full gold labels" % (r["uid"], lack[0]))
            keep += tags
        out.append(dict(r, expected={t: e[t] for t in keep}))
    return out


def laya_item_count(rows, sel, tags, neg):
    """The item count laya_rows implies, without an RNG: per support row old_pos + len(tags) + min(neg, old_labels -
    old_pos), per replay row old_pos + min(neg, old_labels - old_pos). Same refusals as laya_rows."""
    tags = _held(tags)
    _check_neg(neg)
    held, n = set(tags), 0
    for r, support in _selected(rows, sel):
        e = r["expected"]
        old = [t for t in e if t not in held]
        pos = sum(1 for t in old if e[t])
        n += pos + min(neg, len(old) - pos) + (len(tags) if support else 0)
    return n


def berturk_rows(rows, sel, tags):
    """[(row, mask tags)] of a continued BERTurk run in file order: a support row with no mask (its full 48 labels),
    a replay row with the sorted held-out tags masked (pilot.berturk.label_vector sets them to MASK_LABEL). Same
    refusals as laya_rows."""
    mask = tuple(sorted(_held(tags)))
    return [(r, () if support else mask) for r, support in _selected(rows, sel)]


def check_source(block, entry, rows):
    """None when the manifest entry `entry` is the block's frozen source (role train, sha256 and uids_sha256 equal to
    block source's, source split selection_train) and every frozen draw regenerates from `rows` (check_draws, D-02);
    SystemExit naming what differs otherwise."""
    src = block.get("source") or {}
    if src.get("split") != SOURCE_SPLIT:
        raise SystemExit("the frozen few-shot source is %r, not %s; the protocol draws from %s only"
                         % (src.get("split"), SOURCE_SPLIT, SOURCE_SPLIT))
    if entry.get("role") != "train":
        raise SystemExit("the few-shot source entry has role %r, not train; pass the manifest's %s entry"
                         % (entry.get("role"), SOURCE_SPLIT))
    for key in ("sha256", "uids_sha256"):
        if entry.get(key) != src.get(key):
            raise SystemExit("the source entry has %s %s, the frozen few-shot source %s; the draws are bound to the "
                             "split the protocol froze (run `make data`)" % (key, entry.get(key), src.get(key)))
    check_draws(block, rows)


# ---- Phase 9: protocol binding by refusal, never by transcription (D-01, D-03, D-06, D-07, D-08; 09-01) ----

STAGES = ("base", "continued")
MODES = ("primary", "fallback")
FALLBACK_HOURS = 10                          # the frozen compute_fallback trigger's "exceeds 10 h" (a test checks it)
DECISION_SCHEMA = "fewshot-projection/1"
# the D-08 decision record (09-04 writes it before any base run); read at call time, so a test can monkeypatch it
DECISION_PATH = os.path.join(REPO_ROOT, "results", "fewshot", "projection.json")
# canonical keys each trainer's base must hold at the frozen rules.base_runs.<kind> value (Laya micro_batch and
# grad_accum through the matched LAYA-03 ladder configuration)
RECIPE_KEYS = {"laya": ("epochs", "stop_after_epoch", "neg", "encoder_lr", "head_lr", "device", "workers", "max_len",
                        "micro_batch", "grad_accum"),
               "berturk": ("epochs", "stop_after_epoch", "encoder_lr", "head_lr", "batch", "max_len", "device")}
# keys a masked base may hold apart from the chosen selection run's canonical config; each is bound on its own
BASE_FREE = ("seed", "neg_seed", "stop_after_epoch", "mask_tags", "fewshot", "freeze_embeddings", "freeze_encoder",
             "micro_batch", "grad_accum")
# keys a continued run may hold apart from its base's canonical config; each is bound on its own
CONTINUED_FREE = ("base_model_sha256", "base_config_sha256", "base_files_sha256", "epochs", "stop_after_epoch",
                  "mask_tags", "fewshot", "seed", "neg_seed")
_MISSING = object()
_FIX = "Phase 9 runs the frozen protocol unchanged: pass the frozen value"


def _json(v):
    return json.dumps(v, sort_keys=True, ensure_ascii=False)


def _shown(v):
    return "<absent>" if v is _MISSING else repr(v)


def _same(have, want):
    return have is not _MISSING and want is not _MISSING and _json(have) == _json(want)


def _block(protocol):
    block = protocol.get("fewshot") if isinstance(protocol, dict) else None
    if not isinstance(block, dict) or not isinstance(block.get("rules"), dict):
        raise SystemExit("the protocol holds no few-shot block with rules; restore protocol.json with `git checkout -- "
                         "protocol.json`")
    return block


def _int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def binding(pin, stage, draw=None, k=None, digest=None, base_run_id=None):
    """The `fewshot` value of a Phase 9 run's canonical config: {"protocol_sha256", "stage": "base"} for a masked
    base, plus draw, k, digest and base_run_id for a continued run; SystemExit on a stage outside STAGES, a missing
    continued field, or a draw field on a base."""
    if stage not in STAGES:
        raise SystemExit("few-shot stage %r is not one of %s" % (stage, ", ".join(STAGES)))
    extra = {"draw": draw, "k": k, "digest": digest, "base_run_id": base_run_id}
    if stage == "base":
        given = sorted(key for key, v in extra.items() if v is not None)
        if given:
            raise SystemExit("a masked base binding carries no %s; only a continued run names a draw" % given[0])
        return {"protocol_sha256": pin, "stage": "base"}
    for key, v in extra.items():
        if v is None:
            raise SystemExit("a continued few-shot binding needs its %s" % key)
    return dict({"protocol_sha256": pin, "stage": "continued"}, **extra)


def ladder_configs(rule):
    """{LAYA-03 rung label: (freeze_embeddings, freeze_encoder, micro_batch, grad_accum)} relative to the frozen
    rules.base_runs.laya `rule` (A5): none, embeddings_frozen, embeddings_frozen+smaller_batch (half the micro-batch,
    twice the accumulation: the same effective batch) and encoder_frozen; the labels are pilot.runs.RUNG_NEXT's."""
    mb, ga = rule["micro_batch"], rule["grad_accum"]
    return {"none": (False, False, mb, ga), "embeddings_frozen": (True, False, mb, ga),
            "embeddings_frozen+smaller_batch": (True, False, mb // 2, ga * 2), "encoder_frozen": (False, True, mb, ga)}


def chosen_canonical(protocol, kind, root=REPO_ROOT):
    """The chosen selection run's canonical config of `kind`, read from the postfit record the protocol pins
    (systems.<kind>.postfit path, relative to `root` unless absolute, and sha256); SystemExit on a missing system,
    a missing file or bytes that do not hash to the pin."""
    pf = (((protocol.get("systems") or {}).get(kind) or {}).get("postfit") or {}) if isinstance(protocol, dict) else {}
    path, want = pf.get("path"), pf.get("sha256")
    if not path or not want:
        raise SystemExit("the protocol pins no postfit record for system %r (systems.%s.postfit path and sha256)"
                         % (kind, kind))
    full = path if os.path.isabs(path) else os.path.join(root, path)
    if not os.path.isfile(full):
        raise SystemExit("%s not found; the protocol binds the %s base runs to that postfit record (`git checkout -- "
                         "%s`)" % (full, kind, path))
    with open(full, "rb") as f:
        raw = f.read()
    have = sha256_bytes(raw)
    if have != want:
        raise SystemExit("%s has sha256 %s, the protocol pins %s; the chosen recipe is read from the pinned record "
                         "only (`git checkout -- %s`)" % (path, have, want, path))
    canon = (json.loads(raw.decode("utf-8")).get("chosen") or {}).get("canonical")
    if not isinstance(canon, dict):
        raise SystemExit("%s holds no chosen.canonical config" % path)
    return canon


def _check_mode(mode):
    if mode not in MODES:
        raise SystemExit("few-shot mode %r is not one of %s (the D-08 decision record names it)"
                         % (mode, ", ".join(MODES)))


def allowed_base_seeds(block, mode):
    """The base seeds a masked base may train under `mode`: rules.base_runs.seeds (primary) or the compute
    fallback's seed alone (fallback)."""
    _check_mode(mode)
    rules = block["rules"]
    if mode == "primary":
        return list(rules["base_runs"]["seeds"])
    return [rules["compute_fallback"]["seed"]]


def base_seed_of(block, draw, mode):
    """The base seed draw `draw` continues from: its own frozen base_seed under primary, the compute fallback's seed
    under fallback (D-07, D-08). The only place the draw-to-base pairing is decided."""
    _check_mode(mode)
    draws = block.get("draws") or []
    if not _int(draw) or not 0 <= draw < len(draws):
        raise SystemExit("few-shot draw %r is not one of the frozen draws 0..%d" % (draw, len(draws) - 1))
    if mode == "primary":
        return draws[draw]["base_seed"]
    return block["rules"]["compute_fallback"]["seed"]


def read_mode(pin, path=None):
    """'primary' or 'fallback' from the D-08 decision record (DECISION_PATH, read at call time, or `path`); SystemExit
    on a missing or invalid record, another schema, a record of another protocol sha256, a mode outside MODES, or a
    mode that disagrees with its trigger fields (fallback exactly when projection_fired or ladder_failure)."""
    path = DECISION_PATH if path is None else path
    if not os.path.isfile(path):
        raise SystemExit("%s not found; run `python fewshot.py --projection --out results/fewshot/projection.json` "
                         "first: D-08 records the projection before any base run" % path)
    try:
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
    except ValueError as e:
        raise SystemExit("%s is not valid JSON (%s); rebuild it with `python fewshot.py --projection`" % (path, e))
    if not isinstance(rec, dict) or rec.get("schema") != DECISION_SCHEMA:
        raise SystemExit("%s: schema %r, expected %s" % (path, rec.get("schema") if isinstance(rec, dict) else None,
                                                         DECISION_SCHEMA))
    if rec.get("protocol_sha256") != pin:
        raise SystemExit("%s was recorded under protocol sha256 %s, not the frozen %s; the decision binds one protocol"
                         % (path, rec.get("protocol_sha256"), pin))
    mode = rec["decision"].get("mode") if isinstance(rec.get("decision"), dict) else None
    if mode not in MODES:
        raise SystemExit("%s: decision mode %r is not one of %s" % (path, mode, ", ".join(MODES)))
    trig = rec.get("trigger")
    flags = (trig.get("projection_fired"), trig.get("ladder_failure")) if isinstance(trig, dict) else (None, None)
    if not all(isinstance(v, bool) for v in flags):
        raise SystemExit("%s: the trigger needs boolean projection_fired and ladder_failure, got %r" % (path, trig))
    if (mode == "fallback") != any(flags):
        raise SystemExit("%s: decision mode %s disagrees with its trigger (projection_fired %s, ladder_failure %s); "
                         "the frozen trigger decides the mode" % (path, mode, flags[0], flags[1]))
    return mode


def _differ(canon, want, free, what):
    """SystemExit naming the first key in sorted order (outside `free`) whose JSON value differs, a key missing on
    one side shown as <absent>."""
    for key in sorted((set(canon) | set(want)) - set(free)):
        have, ref = canon.get(key, _MISSING), want.get(key, _MISSING)
        if not _same(have, ref):
            raise SystemExit("%s is %s here but %s in %s; %s" % (key, _shown(have), _shown(ref), what, _FIX))


def _equal(key, have, want, what):
    if not _same(have, want):
        raise SystemExit("%s is %s here but %s in %s; %s" % (key, _shown(have), _shown(want), what, _FIX))


def check_base(protocol, pin, kind, canon, chosen, mode):
    """The LAYA-03 rung label ("none" for BERTurk) of a masked base command whose canonical config `canon` binds to
    the frozen protocol; SystemExit naming the first difference otherwise (D-01, D-03, A5). The split and data_sha256
    are the frozen source's; the seed one of allowed_base_seeds(mode), Laya's neg_seed the seed; mask_tags the sorted
    held-out tags; fewshot binding(pin, "base"); every RECIPE_KEYS value the frozen rules.base_runs.<kind> (Laya's
    freeze flags, micro-batch and grad-accum exactly one ladder configuration); every other key outside BASE_FREE
    equal to the chosen selection run's `chosen` canonical config, the first differing key in sorted order named."""
    block = _block(protocol)
    if kind not in RECIPE_KEYS:
        raise SystemExit("few-shot system kind %r is not one of %s" % (kind, ", ".join(sorted(RECIPE_KEYS))))
    _check_mode(mode)
    rules = block["rules"]["base_runs"]
    rule, src = rules[kind], block.get("source") or {}
    _equal("split", canon.get("split", _MISSING), rules["split"], "the frozen rules.base_runs")
    _equal("data_sha256", canon.get("data_sha256", _MISSING), src.get("sha256"), "the frozen few-shot source")
    seed, allowed = canon.get("seed"), allowed_base_seeds(block, mode)
    if not _int(seed) or seed not in allowed:
        raise SystemExit("seed %r is not one of the base seeds %s of the %s protocol; %s"
                         % (seed, allowed, mode, _FIX))
    if kind == "laya" and not _same(canon.get("neg_seed", _MISSING), seed):
        raise SystemExit("neg_seed is %s here but the seed is %r; a masked base draws its negatives with neg_seed == "
                         "seed (omit --neg-seed)" % (_shown(canon.get("neg_seed", _MISSING)), seed))
    _equal("mask_tags", canon.get("mask_tags", _MISSING), sorted(block["tags"]), "the frozen held-out tags (sorted)")
    _equal("fewshot", canon.get("fewshot", _MISSING), binding(pin, "base"), "this protocol's base binding")
    rung = "none"
    if kind == "laya":
        have = [canon.get(key, _MISSING) for key in ("freeze_embeddings", "freeze_encoder", "micro_batch",
                                                     "grad_accum")]
        ladder = ladder_configs(rule)
        match = [label for label, cfg in ladder.items() if _same(have, list(cfg))]
        if not match:
            raise SystemExit("freeze_embeddings, freeze_encoder, micro_batch, grad_accum are %s here, not one LAYA-03 "
                             "ladder configuration (%s); a base walks the ladder by its flags only"
                             % (", ".join(_shown(v) for v in have),
                                "; ".join("%s %s" % (label, list(cfg)) for label, cfg in ladder.items())))
        rung = match[0]
    for key in RECIPE_KEYS[kind]:
        if key not in ("micro_batch", "grad_accum"):
            _equal(key, canon.get(key, _MISSING), rule.get(key, _MISSING), "the frozen rules.base_runs.%s" % kind)
    _differ(canon, chosen, BASE_FREE, "the chosen %s selection run" % kind)
    return rung


def check_continued(protocol, pin, kind, canon, base_canon, base_run_id, draw, k, mode):
    """None when a continued command's canonical config `canon` binds to the frozen protocol and to its masked base
    `base_canon` (run id `base_run_id`); SystemExit naming the first difference otherwise (D-01, D-06, D-07). The
    (draw, k) is a frozen selection; seed (and Laya's neg_seed) the draw seed; fewshot the continued binding of this
    draw's digest and base; epochs and LRs the frozen continued_training's; no stop_after_epoch, mask_tags or
    postfit_sha256; the base a masked base of this protocol (its own binding, base_seed_of(block, draw, mode), the
    sorted held-out tags); every other key outside CONTINUED_FREE equal to the base's, so the base's rung carries
    over."""
    block = _block(protocol)
    if kind not in RECIPE_KEYS:
        raise SystemExit("few-shot system kind %r is not one of %s" % (kind, ", ".join(sorted(RECIPE_KEYS))))
    selection(block, draw, k)
    d = block["draws"][draw]
    seed = canon.get("seed", _MISSING)
    if not _int(seed) or seed != d["seed"]:
        raise SystemExit("seed is %s here but draw %d's seed is %r; the training seed is the draw seed; %s"
                         % (_shown(seed), draw, d["seed"], _FIX))
    if kind == "laya":
        _equal("neg_seed", canon.get("neg_seed", _MISSING), d["seed"], "draw %d (neg_seed = draw seed)" % draw)
    _equal("fewshot", canon.get("fewshot", _MISSING), binding(pin, "continued", draw, k, d["digest"], base_run_id),
           "this protocol's continued binding of draw %d (digest %s), k %d and base %s"
           % (draw, d["digest"], k, base_run_id))
    ct = block["rules"]["continued_training"]
    for key in ("epochs", "encoder_lr", "head_lr"):
        _equal(key, canon.get(key, _MISSING), ct[key], "the frozen continued_training")
    for key in ("stop_after_epoch", "mask_tags", "postfit_sha256"):
        if canon.get(key) is not None:
            raise SystemExit("%s is %r here; a continued run trains every one of its %d epochs on the frozen support "
                             "and replay rows, with no %s; %s" % (key, canon[key], ct["epochs"], key, _FIX))
    if not _same(base_canon.get("fewshot", _MISSING), binding(pin, "base")):
        raise SystemExit("the base %s carries fewshot binding %s, not a masked base of this protocol (%s); continue "
                         "from a bound masked base" % (base_run_id, _shown(base_canon.get("fewshot", _MISSING)),
                                                        _json(binding(pin, "base"))))
    want = base_seed_of(block, draw, mode)
    if base_canon.get("seed") != want or not _int(base_canon.get("seed")):
        raise SystemExit("the base %s has seed %r, but draw %d continues from the base of seed %r under the %s "
                         "protocol (base_seed_of); %s" % (base_run_id, base_canon.get("seed"), draw, want, mode, _FIX))
    if not _same(base_canon.get("mask_tags", _MISSING), sorted(block["tags"])):
        raise SystemExit("the base %s masks %s, not the frozen held-out tags %s"
                         % (base_run_id, _shown(base_canon.get("mask_tags", _MISSING)), sorted(block["tags"])))
    _differ(canon, base_canon, CONTINUED_FREE, "its base %s" % base_run_id)


def _read_split(manifest_path, m, name, roles):
    entry = m["splits"][name]
    path = os.path.join(os.path.dirname(os.path.realpath(manifest_path)), entry["path"])
    if not os.path.isfile(path):
        raise SystemExit("%s not found; run `make data` to write the registered splits" % path)
    with open(path, "rb") as f:
        raw = f.read()
    verify_split_file(path, manifest_path=manifest_path, roles=roles, data=raw)
    return entry, parse_jsonl(raw)


def protocol_block(manifest_path=DEFAULT_MANIFEST):
    """The complete few-shot block 08-05 freezes into protocol.json, from verified bytes of the committed manifest:
    settings, eligibility (gold label counts of test and final_train), source, sampler, the three draws with their
    uid lists, realised positives and digests, and RULES. SystemExit on another manifest, a missing or tampered
    split, an ineligible tag or a short pool."""
    if not manifest_is_committed(manifest_path):
        raise SystemExit("%s is not the committed data/splits.json; the few-shot protocol is frozen from the "
                         "committed splits only (drop --manifest, or run `make data`)" % manifest_path)
    with open(manifest_path, encoding="utf-8") as f:
        m = json.load(f)
    entry, rows = _read_split(manifest_path, m, "selection_train", ("train",))
    _, final_rows = _read_split(manifest_path, m, "final_train", ("train",))
    _, test_rows = _read_split(manifest_path, m, "test", ("test",))
    elig = check_eligibility(TAGS, {"test": tag_counts(test_rows, TAGS), "final_train": tag_counts(final_rows, TAGS)},
                             pins.OOD_SCORING_TAGS)
    elig["note"] = ("gold labels only, no model output: positives per held-out tag in the committed test and "
                    "final_train splits")
    draws = []
    for i, s in enumerate(DRAW_SEEDS):
        d = draw(rows, s)
        draws.append({"draw": i, "seed": s, "base_seed": s, "support": d["support"], "replay": d["replay"],
                      "positives": d["positives"], "digest": draw_digest(d)})
    block = {"tags": list(TAGS), "k_values": list(K_VALUES), "draw_seeds": list(DRAW_SEEDS),
             "neg_per_pos": NEG_PER_POS, "replay": REPLAY, "epochs": EPOCHS, "eligibility": elig,
             "source": {"split": "selection_train", "sha256": entry["sha256"], "uids_sha256": entry["uids_sha256"]},
             "sampler": {"schema": DRAW_SCHEMA, "rule": SAMPLER_RULE, "digest": DIGEST_RULE}, "draws": draws,
             "rules": json.loads(json.dumps(RULES))}
    check_draws(block, rows)
    return block
