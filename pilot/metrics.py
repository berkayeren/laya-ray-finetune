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

Val-selected thresholds (EVAL-02, D-12), one rule for every system, chosen on val only and never on test:
- GRID: 0.05..0.95 in steps of 0.01 (91 points), built as integers 5..95 divided by 100 so each point equals its
  decimal literal; no rounding call
- global θ: maximises val micro-F1 over GRID
- per-tag θ: only for tags with at least MIN_VAL_POSITIVES (10) val positives, maximising that tag's val F1 over the
  same grid; every other tag uses the global θ (theta_vector)
- ties (global and per-tag alike): the θ closest to 0.5; between two equally close points, the lower θ; when every
  point scores 0 that is 0.5. The per-tag objective and this tie rule are research assumption A2, confirmed by the
  user on 2026-10-14 (the alternative, breaking per-tag ties toward the global θ, was rejected)
- θ is selected on the probabilities a system reports, i.e. after its temperature (the caller passes them)

Paired bootstrap (EVAL-07), the one interval procedure for every comparison:
- seed_average: each system's per-seed probabilities sigmoid(logit / T_s) are averaged per message and tag BEFORE
  thresholding (probabilities, never metrics, are averaged); one seed is the identity
- paired_bootstrap: messages are resampled with replacement through ONE index draw (rng.integers(0, n, size=(B, n))
  from a numpy Generator seeded with `seed`) shared by both systems, so a system compared with itself (or an
  identical copy) has every replicate delta exactly 0.0 and a [0.0, 0.0] interval
- the draw becomes count weights W [B, n] and every replicate's metrics come from W @ indicator matrices
  (replicate_metrics), with metrics()' semantics; BOOT_METRICS are micro-F1, macro-F1, exact match and top-1
- the interval is the 95% percentile interval of the replicate deltas A - B (numpy linear interpolation);
  thresholds are inputs, never selected inside the bootstrap

usage: from pilot import metrics; metrics.metrics(Y, metrics.probabilities(logits, T), 0.5)
       sel = metrics.select_theta(Y_val, P_val); metrics.metrics(Y, P, metrics.theta_vector(sel))
       metrics.paired_bootstrap(Y, metrics.seed_average([PA0, PA1]), thA, PB, thB, B=10000, seed=0)
"""
import numpy as np

from intents import INTENTS

TAGS = list(INTENTS)
N_RANKED = 5                                                # tags listed in best_tags / worst_tags
GRID = np.arange(5, 96) / 100.0                             # 0.05..0.95, 91 points, each equal to its decimal literal
MIN_VAL_POSITIVES = 10                                      # val positives a tag needs for its own θ
TIE_RULE = "closest to 0.5; between two equally close points, the lower θ"
BOOT_METRICS = ("micro_f1", "macro_f1", "exact_match", "top1_hit")
BOOT_METHOD = "percentile (numpy linear interpolation)"


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


def _arrays(Y, P, empty):
    """(Y bool, P float64) checked to be the same non-empty [n, 48]; `empty` is the refusal for zero rows."""
    if len(Y) == 0:
        raise SystemExit(empty)
    Y = np.asarray(Y, dtype=bool)
    P = np.asarray(P, dtype=np.float64)
    if Y.ndim != 2 or Y.shape[1] != len(TAGS) or P.shape != Y.shape:
        raise SystemExit("gold %s and probabilities %s must both be [n, %d]" % (Y.shape, P.shape, len(TAGS)))
    return Y, P


def metrics(Y, P, theta):
    """The metric block of one system at one threshold: Y bool [n, 48] gold, P float [n, 48] probabilities."""
    Y, P = _arrays(Y, P, "no messages to score")
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


def pick(scores):
    """The GRID point with the best score; ties go to the point closest to 0.5, then to the lower θ (TIE_RULE)."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != GRID.shape:
        raise SystemExit("%d scores for a %d-point θ grid" % (scores.size, GRID.size))
    cand = np.flatnonzero(scores == scores.max())
    return float(GRID[cand[np.argmin(np.abs(GRID[cand] - 0.5))]])   # argmin keeps the first (lower) on a tie


def _f1_counts(tp, fp, fn):
    """F1 = 2tp / (2tp + fp + fn) elementwise; 0 where tp is 0 (so 0/0 is 0)."""
    den = 2 * tp + fp + fn
    return np.where(tp > 0, 2 * tp / np.maximum(den, 1), 0.0)


def select_theta(Y_val, P_val):
    """The val-selected θ (D-12): global θ on val micro-F1, per-tag θ on val F1 for tags with >= 10 val positives.

    P_val are the probabilities the system reports (after its temperature). Only val may be passed in: the caller
    applies the result to test with theta_vector, never the other way round.
    """
    Y, P = _arrays(Y_val, P_val, "no val messages to select θ on; score the full val split")
    pred = P[None] >= GRID[:, None, None]                   # [91, n, 48], the inclusive decision rule
    tp = (pred & Y[None]).sum(axis=1)                       # [91, 48] counts per grid point and tag
    fp = (pred & ~Y[None]).sum(axis=1)
    fn = (~pred & Y[None]).sum(axis=1)
    support = Y.sum(axis=0)
    eligible = [t for j, t in enumerate(TAGS) if support[j] >= MIN_VAL_POSITIVES]
    per_tag_f1 = _f1_counts(tp, fp, fn)
    return {"global": pick(_f1_counts(tp.sum(axis=1), fp.sum(axis=1), fn.sum(axis=1))),
            "per_tag": {t: pick(per_tag_f1[:, TAGS.index(t)]) for t in eligible},
            "eligible_tags": eligible, "min_val_positives": MIN_VAL_POSITIVES,
            "grid": {"start": float(GRID[0]), "stop": float(GRID[-1]), "step": 0.01, "n": int(GRID.size)},
            "objective": {"global": "val micro-F1", "per_tag": "val per-tag F1"},
            "tie_break": TIE_RULE}


def theta_vector(sel):
    """float64 [48] θ in INTENTS order: a tag's own θ where select_theta gave one, the global θ elsewhere."""
    return np.array([sel["per_tag"].get(t, sel["global"]) for t in TAGS], dtype=np.float64)


def seed_average(prob_list):
    """EVAL-07: the per-message, per-tag mean of a system's per-seed probabilities (one seed -> itself).

    Probabilities are averaged before any threshold, never metrics: a tag at 0.4 and 0.7 averages to 0.55 and is
    predicted at θ 0.5 although one seed alone would not predict it.
    """
    arrays = [np.asarray(p, dtype=np.float64) for p in prob_list]
    if not arrays:
        raise SystemExit("no seed dumps to average")
    if any(p.shape != arrays[0].shape for p in arrays):
        raise SystemExit("seed probabilities have different shapes %s; every seed must score the same messages"
                         % sorted({p.shape for p in arrays}))
    return np.mean(np.stack(arrays), axis=0)


def bootstrap_weights(n, B, seed):
    """(idx int [B, n], W float64 [B, n]): one draw of n row indices with replacement per replicate, and W[b, i] the
    number of times row i is in replicate b. Built without a Python loop over replicates."""
    if int(n) < 1:
        raise SystemExit("no messages to bootstrap (n = %d)" % int(n))
    if int(B) < 1:
        raise SystemExit("the bootstrap needs B >= 1 replicates, got %d" % int(B))
    n, B = int(n), int(B)
    idx = np.random.default_rng(seed).integers(0, n, size=(B, n))
    W = np.bincount((np.arange(B)[:, None] * n + idx).ravel(), minlength=B * n).reshape(B, n)
    return idx, W.astype(np.float64)


def _ratio(num, den):
    """num / den elementwise; 0.0 where den is 0."""
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)


def replicate_metrics(W, Y, pred, P):
    """{name: float64 [B]} of BOOT_METRICS for every replicate row of W, with the semantics of metrics():
    micro-F1, exact match and top-1 score 0.0 on a 0 denominator; macro-F1 averages per-tag F1 over tags with
    replicate support > 0 or a prediction, and is NaN for a replicate with no such tag."""
    W = np.asarray(W, dtype=np.float64)
    Y, pred = np.asarray(Y, dtype=bool), np.asarray(pred, dtype=bool)
    n = Y.shape[0]
    tp = W @ (Y & pred).astype(np.float64)                  # [B, 48] weighted counts (integers held exactly)
    fp = W @ (~Y & pred).astype(np.float64)
    fn = W @ (Y & ~pred).astype(np.float64)
    tps, fps, fns = tp.sum(axis=1), fp.sum(axis=1), fn.sum(axis=1)
    micro = _ratio(2 * tps, 2 * tps + fps + fns)
    f1 = _ratio(2 * tp, 2 * tp + fp + fn)
    in_macro = ((W @ Y.astype(np.float64)) > 0) | ((W @ pred.astype(np.float64)) > 0)
    k = in_macro.sum(axis=1)
    macro = np.where(k > 0, (f1 * in_macro).sum(axis=1) / np.maximum(k, 1), np.nan)
    exact = (W @ (pred == Y).all(axis=1).astype(np.float64)) / n
    tagged = Y.any(axis=1)
    hit = Y[np.arange(n), np.asarray(P).argmax(axis=1)] & tagged   # argmax ties -> earlier INTENTS tag
    top1 = _ratio(W @ hit.astype(np.float64), W @ tagged.astype(np.float64))
    return {"micro_f1": micro, "macro_f1": macro, "exact_match": exact, "top1_hit": top1}


def _percentiles(level):
    """The two percentile points of a central interval at `level` (0.95 -> [2.5, 97.5])."""
    level = float(level)
    if not 0.0 < level < 1.0:
        raise SystemExit("interval level must be in (0, 1), got %r" % level)
    return [round(50.0 * (1.0 - level), 12), round(50.0 * (1.0 + level), 12)]


def paired_bootstrap(Y, PA, thA, PB, thB, B=10000, seed=0, level=0.95):
    """EVAL-07 paired message-level bootstrap of A - B on one shared index draw.

    PA / PB are each system's (seed-averaged) probabilities on the same messages in the same order, thA / thB its
    thresholds (one float or 48 values; chosen by the caller, never here). Each metric reports A and B on the full
    data, delta = A - B, and the percentile interval of the replicate deltas; replicates where the delta is undefined
    (macro-F1 with no tag in scope) are counted in n_undefined and left out of the percentile.
    """
    Y, PA = _arrays(Y, PA, "no messages to bootstrap")
    _, PB = _arrays(Y, PB, "no messages to bootstrap")
    q = _percentiles(level)
    full_a, full_b = metrics(Y, PA, thA), metrics(Y, PB, thB)       # also validates both thresholds
    _, W = bootstrap_weights(Y.shape[0], B, seed)                   # ONE draw shared by both systems (pairing)
    ra = replicate_metrics(W, Y, PA >= np.asarray(thA, dtype=np.float64), PA)
    rb = replicate_metrics(W, Y, PB >= np.asarray(thB, dtype=np.float64), PB)
    out = {}
    for name in BOOT_METRICS:
        a, b = full_a[name], full_b[name]
        d = ra[name] - rb[name]
        undefined = np.isnan(d)
        if undefined.all():
            ci, width = None, None
        else:
            ci = [float(v) for v in np.percentile(d[~undefined], q)]
            width = ci[1] - ci[0]
        out[name] = {"a": a, "b": b, "delta": None if a is None or b is None else a - b,
                     "ci": ci, "width": width, "n_undefined": int(undefined.sum())}
    return {"B": int(B), "seed": seed, "level": level, "method": BOOT_METHOD, "n": int(Y.shape[0]),
            "metrics": out}
