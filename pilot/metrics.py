"""Torch-free metric functions on [n, 48] arrays, shared by report.py and later phases (EVAL-02).

Every system (majority, label prior, zero-shot and fine-tuned Laya, BERTurk) goes through the same two functions, so
nothing here may name a system (D-10). Values are computed in float64 and never rounded (D-01): rounding is for
display only.

Semantics (ported from the legacy evaluate.metrics):
- decision rule: a tag is predicted when P >= theta (inclusive); theta is one float or one value per tag
- micro precision/recall/F1 from the summed tp/fp/fn; 0/0 gives 0.0
- macro-F1 averages per-tag F1 over tags with gold support > 0 or at least one prediction; a predicted tag with zero
  support counts F1 = 0, a tag with neither is left out (n_tags_in_macro records how many were averaged); None when
  no tag qualifies
- exact match: the predicted set equals the gold set, so an empty prediction on a no-intent row is a hit
- top-1: on rows with at least one gold tag only (n_top1), the argmax tag is gold; argmax ties go to the earlier
  INTENTS tag; None when no row has a gold tag

usage: from pilot import metrics; metrics.metrics(Y, metrics.probabilities(logits, T), 0.5)
"""
import numpy as np

from intents import INTENTS

TAGS = list(INTENTS)
N_RANKED = 5                                                # tags listed in best_tags / worst_tags


def probabilities(logits, temperature):
    """sigmoid(logits / T) in float64, written as 0.5 * (1 + tanh(x / 2)) so no |logit| overflows."""
    try:
        t = float(temperature)
    except (TypeError, ValueError):
        raise SystemExit("temperature %r is not a number; expected a finite T > 0" % (temperature,))
    if not np.isfinite(t) or t <= 0:
        raise SystemExit("temperature %r is not a finite T > 0" % (temperature,))
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(logits, dtype=np.float64) / t))


def _prf(tp, fp, fn):
    """(precision, recall, F1) of integer counts; every 0/0 is 0.0."""
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * tp / (2 * tp + fp + fn) if tp else 0.0          # == 2pr / (p + r), exact for integer counts
    return float(p), float(r), float(f)


def metrics(Y, P, theta):
    """The metric block of one system at one threshold: Y bool [n, 48] gold, P float [n, 48] probabilities."""
    if len(Y) == 0:
        raise SystemExit("no messages to score")
    Y = np.asarray(Y, dtype=bool)
    P = np.asarray(P, dtype=np.float64)
    if Y.ndim != 2 or Y.shape[1] != len(TAGS) or P.shape != Y.shape:
        raise SystemExit("gold %s and probabilities %s must both be [n, %d]" % (Y.shape, P.shape, len(TAGS)))
    th = np.asarray(theta, dtype=np.float64)
    if th.shape not in ((), (len(TAGS),)) or not np.isfinite(th).all():
        raise SystemExit("theta must be one finite float or %d finite per-tag values, got shape %s"
                         % (len(TAGS), th.shape))
    n = Y.shape[0]
    pred = P >= th
    tp = (Y & pred).sum(axis=0)
    fp = (~Y & pred).sum(axis=0)
    fn = (Y & ~pred).sum(axis=0)
    support, n_pred = Y.sum(axis=0), pred.sum(axis=0)

    per_tag = {}
    for j, t in enumerate(TAGS):
        p, r, f = _prf(int(tp[j]), int(fp[j]), int(fn[j]))
        per_tag[t] = {"support": int(support[j]), "n_pred": int(n_pred[j]), "tp": int(tp[j]), "fp": int(fp[j]),
                      "fn": int(fn[j]), "precision": p, "recall": r, "f1": f}
    micro_p, micro_r, micro_f = _prf(int(tp.sum()), int(fp.sum()), int(fn.sum()))
    in_macro = [t for t in TAGS if per_tag[t]["support"] > 0 or per_tag[t]["n_pred"] > 0]
    macro = sum(per_tag[t]["f1"] for t in in_macro) / len(in_macro) if in_macro else None

    tagged = Y.any(axis=1)
    n_top1 = int(tagged.sum())
    hit = Y[np.arange(n), P.argmax(axis=1)]                 # argmax keeps the earliest INTENTS tag on ties
    top1 = float(hit[tagged].mean()) if n_top1 else None

    ranked = [t for t in TAGS if per_tag[t]["support"] > 0]   # stable sorts: ties stay in INTENTS order
    best = sorted(ranked, key=lambda t: -per_tag[t]["f1"])[:N_RANKED]
    worst = sorted(ranked, key=lambda t: per_tag[t]["f1"])[:N_RANKED]
    return {"n": int(n), "n_top1": n_top1, "n_no_intent": int(n - n_top1),
            "micro_precision": micro_p, "micro_recall": micro_r, "micro_f1": micro_f,
            "macro_f1": macro, "n_tags_in_macro": len(in_macro),
            "exact_match": float((pred == Y).all(axis=1).mean()),
            "top1_hit": top1,
            "mean_tags_predicted": float(pred.sum(axis=1).mean()), "mean_gold_tags": float(Y.sum(axis=1).mean()),
            "per_tag": per_tag,
            "best_tags": {t: per_tag[t]["f1"] for t in best}, "worst_tags": {t: per_tag[t]["f1"] for t in worst}}
