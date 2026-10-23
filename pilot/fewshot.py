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
from pilot.data import DEFAULT_MANIFEST, manifest_is_committed, parse_jsonl, sha256_bytes, verify_split_file

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
