"""The few-shot results record (Phase 9, FEW-02, D-10, D-12): the frozen primary and secondary metrics of the few-shot
experiment, computed from the dumps the committed runs manifest lists, numpy only, no model.

What it reads, and nothing else:
- the frozen protocol (pilot.protocol.require_frozen, by the caller) and its few-shot block: the held-out tags, k values,
  draws (digests, realised positives), rules.primary_metric.k and the bootstrap block (B, seed, level)
- results/fewshot/runs.json through pilot.fewshot.read_runs (check_runs under the recorded D-08 mode, the decision
  record bound by its sha256)
- per listed entry its dumps under results/dumps/<dump_run_id>/ through pilot.dumps.load_dump (the only reader) and
  read_gold: a masked base's val dump (ungated: no protocol_sha256, no fewshot_runs_sha256; its θ source, D-05) and every
  entry's test dump (gated: protocol_sha256 the pin, fewshot_runs_sha256 the manifest's sha256). Each sidecar must be of
  the entry's system kind, training run id and seed; every test dump must score one uid order (the draws and models are
  paired message by message) and be a full dump, and each split must be the frozen protocol's (sha256); a masked
  base's val dump must be of its test dump's source (SAME_SOURCE_KEYS: kind, checkpoint identity, shipped T, scorer
  device, dtype and amp; 09-REVIEW WR-02), as report.py requires of any val θ source

What it computes (D-12; protocol.json rules.primary_metric and rules.secondary_metrics):
- primary: per new tag the average precision by pilot.metrics.AP_RULE on P = sigmoid(logit) at raw T = 1 on test (AP is
  T-invariant), macro over the held-out tags (pilot.metrics.new_tag_ap), per draw Laya - BERTurk at k = the frozen
  primary k; draws = the per-draw deltas and mean_delta = float(np.mean(np.asarray(draws, dtype=np.float64))), exactly
  the value pilot.verdict.criterion_d checks
- ap: per model the per-tag AP and macro-AP of every masked base ("s<seed>", k = 0) and continued run ("d<draw>-k<k>")
- by_k: the primary block's per-draw table at every k: "0" (each draw's masked base, pilot.fewshot.base_seed_of) and
  every frozen k (k = 5 is the reported secondary)
- theta (A1, THETA_RULE): each masked base's θ, pilot.metrics.select_theta on its own val dump at T = 1 over all 48
  columns; f1: new-tag (held-out columns) and old-tag (the other 42) micro-F1 = 2TP / (2TP + FP + FN) of the summed
  per-tag counts at that θ (0.0 when the denominator is 0), for every base and every continued run from it, and
  forgetting = the continued run's old-tag micro-F1 minus its base's
- theta_trained (09-REVIEW CR-01, POST_HOC_LABEL: post hoc, not frozen, descriptive; not a decision input): the same
  θ and f1 blocks with each masked base's θ chosen on its trained columns only (trained_theta, THETA_TRAINED_RULE),
  and the cause it discloses (THETA_CAUSE: a masked BERTurk base's untrained held-out rows sit near P = 0.5 on val
  and push its 48-column global θ up). The frozen reading (theta, f1) is unchanged; the primary block, by_k, the
  bootstrap and so criterion (d) read neither
- bootstrap (A3, BOOT_PAIRING): per k a descriptive paired percentile interval of the mean-over-draws macro-AP delta on
  one pilot.metrics.bootstrap_weights(n, B, seed) draw with the protocol's B, seed and level; a replicate in which a
  held-out tag has no positive is counted in n_undefined and left out, never scored as 0 (on the full split such a tag
  is refused by pilot.metrics.new_tag_ap)
- wall_clock_s: the incremental wall clock of every continued run, copied from the runs manifest (WALL_RULE); positives:
  the realised positives per tag, draw and k, copied from the protocol's frozen draws

render_markdown(record) is the README Phase 9 block (points with 2 decimals, AP and F1 with 3; rounding for display
only); fewshot.py --render PATH prints it for a record of the frozen protocol.

Values are float64 and never rounded (rounding happens only in render_markdown). The record holds no timestamp, no uid
list, no message text and no local path (repo-relative paths, ids and digests), and is written as key-sorted JSON, so it
rebuilds byte for byte from the same dumps, whatever order the runs manifest lists its entries in. Nothing in it is
selected on test: θ comes only from each masked base's own val dump, AP needs no θ or T.

Numpy and stdlib only: never torch, scipy, sklearn, laya or Ray (sklearn's average_precision_score is a test oracle).

usage:
    from pilot import fewshot_report as fr
    rec = fr.build_record(protocol, pin)                # results/fewshot/runs.json and the dumps it lists
    fr.write_record(rec, fr.RECORD_PATH)
    print(fr.render_markdown(rec))
    python fewshot.py --report --out results/fewshot/fewshot.json      # make fewshot-reports
    python fewshot.py --render results/fewshot/fewshot.json            # the README Phase 9 block
"""
import json
import os

import numpy as np

import pilot.data
from pilot import dumps, fewshot, metrics
from pilot.data import REPO_ROOT, repo_relpath, sha256_bytes, write_atomic

RESULTS_SCHEMA = "fewshot-results/1"
RECORD_PATH = os.path.join(REPO_ROOT, "results", "fewshot", "fewshot.json")
PRIMARY_METRIC = "new_tag_macro_ap"
KINDS = fewshot.KINDS                                       # ("laya", "berturk"): every delta is Laya - BERTurk
THETA_RULE = ("A1: each masked base's θ is pilot.metrics.select_theta on its own val dump at T = 1 over all 48 columns "
              "(the val rule unchanged: a global θ on val micro-F1, a tag's own θ only with >= %d val positives); the "
              "held-out tags have fewer val positives, so they use the global θ; the base's θ is applied unchanged to "
              "the base and to every continued run from it (pilot.fewshot.base_seed_of); nothing is chosen on test"
              % metrics.MIN_VAL_POSITIVES)
POST_HOC_LABEL = "post hoc, not frozen, descriptive; not a decision input"
THETA_TRAINED_RULE = ("each masked base's θ is pilot.metrics.select_theta on its own val dump at T = 1 over its trained "
                      "columns only: the held-out columns are excluded (gold False and P 0, so they add no TP, FP or "
                      "FN), with the same grid, tie rule and per-tag eligibility; the base's θ is applied unchanged to "
                      "the base and to every continued run from it, and the held-out tags use its global θ; nothing "
                      "is chosen on test. It was added after the results were seen (09-REVIEW CR-01): the frozen "
                      "reading (theta and f1, θ over all 48 columns, A1) stays the record's secondary numbers, and the "
                      "primary metric and criterion (d) read neither")
THETA_CAUSE = ("BERTurk's new-tag micro-F1 can be zero at the frozen reading because a masked BERTurk base's held-out "
               "classifier rows get no gradient (masked BCE), so they score near sigmoid(0) = 0.5 on val; "
               "a global θ below them adds false positives across every held-out column, so the 48-column val rule "
               "pushes the BERTurk base's global θ up, and the new-tag scores its continued runs reach on test can stay "
               "below it; the θ over the trained columns is set by the trained columns alone")
BOOT_PAIRING = ("A3: one pilot.metrics.bootstrap_weights(n, B, seed) draw of test messages, shared by every model and "
                "draw; each replicate's statistic is the mean over draws of (Laya - BERTurk) new-tag macro-AP on the "
                "resampled messages; a replicate in which a held-out tag has no positive is undefined (AP is undefined "
                "there), counted in n_undefined and left out of the percentile interval; descriptive only, never gated")
NOTES = [
    "epoch parity, not step parity: every continued run trains the frozen number of epochs on the same support and "
    "replay rows, so BERTurk (batch 32) takes far fewer optimizer steps than Laya (one item per question)",
    "Laya trains on neg8 items (pilot.fewshot.LAYA_ROWS_RULE) against BERTurk's full 48 labels per row, as in the main "
    "experiment",
    "a continued Laya run starts from the masked base checkpoint as saved (fp16-rounded weights, the same checkpoint "
    "that is scored as k = 0); BERTurk checkpoints are fp32",
    "AP is threshold-free and T-invariant: the primary metric uses raw T = 1 and no θ; only the secondary F1 numbers "
    "use the masked base's val θ",
    "the support-set and replay uid lists live in protocol.json only; this record names each draw by its digest",
]


def _manifest(manifest_path):
    return pilot.data.DEFAULT_MANIFEST if manifest_path is None else manifest_path


def _root(dump_root):
    return dumps.DUMP_ROOT if dump_root is None else dump_root


def run_key(e):
    """The record key of a runs-manifest entry: "s<seed>" for a masked base, "d<draw>-k<k>" for a continued run."""
    return "s%d" % e["seed"] if e["stage"] == "base" else "d%d-k%d" % (e["draw"], e["k"])


def _what(e, split):
    return "%s/%s (%s %s %s)" % (e["dump_run_id"], split, e["kind"], e["stage"], run_key(e))


def load(entry, split, pin, runs_sha256, dump_root=None, manifest_path=None, uids_ref=None):
    """(Y bool [n, 48], P float64 [n, 48] at T = 1, uids, sidecar, dump_sha256) of one listed entry's dump, verified
    by pilot.dumps.load_dump and bound to the entry (system kind, training run id, seed) and to the gate: a test dump
    carries protocol_sha256 == `pin` and fewshot_runs_sha256 == `runs_sha256`, a val dump (a masked base's θ source)
    carries neither. With `uids_ref` the dump must score exactly those uids in that order. SystemExit naming the dump
    and what is wrong otherwise."""
    manifest = _manifest(manifest_path)
    if split not in ("val", "test") or (split == "val" and entry["stage"] != "base"):
        raise SystemExit("%s: the few-shot record reads a masked base's val dump and every entry's test dump only"
                         % _what(entry, split))
    logits, uids, side, sentry = dumps.load_dump(os.path.join(_root(dump_root), entry["dump_run_id"]), split,
                                                 manifest)
    what = _what(entry, split)
    system, training = side.get("system") or {}, side.get("training") or {}
    if system.get("kind") != entry["kind"]:
        raise SystemExit("%s: the dump is of system kind %r, the runs manifest lists a %s run" % (what,
                                                                                               system.get("kind"),
                                                                                               entry["kind"]))
    if training.get("run_id") != entry["train_run_id"]:
        raise SystemExit("%s: the dump's training run is %r, the runs manifest lists %s; score the listed checkpoint"
                         % (what, training.get("run_id"), entry["train_run_id"]))
    if side.get("seed") != entry["seed"]:
        raise SystemExit("%s: the dump records seed %r, the runs manifest lists seed %r" % (what, side.get("seed"),
                                                                                           entry["seed"]))
    if split == "test":
        for key, want in (("protocol_sha256", pin), ("fewshot_runs_sha256", runs_sha256)):
            if side.get(key) != want:
                raise SystemExit("%s: %s %s, expected %s; a few-shot test dump is scored once through score.py's "
                                 "protocol gate under this protocol and runs manifest" % (what, key, side.get(key),
                                                                                         want))
    elif "protocol_sha256" in side or "fewshot_runs_sha256" in side:
        raise SystemExit("%s: the val dump carries protocol_sha256 or fewshot_runs_sha256; a masked base's θ source is "
                         "its own ungated val dump (D-05)" % what)
    if uids_ref is not None and uids != uids_ref:
        raise SystemExit("%s: the dump scores the %s messages in another uid order than the other few-shot dumps; "
                         "every model and draw is paired message by message (rescore the full split)" % (what, split))
    if side.get("n_subset") is not None:
        raise SystemExit("%s: a --n subset dump (n_subset %r); the few-shot record reads full dumps only"
                         % (what, side.get("n_subset")))
    Y = dumps.read_gold(sentry, manifest, uids)
    return Y, metrics.probabilities(logits, 1.0), uids, side, side["dump_sha256"]


# the sidecar keys on which a masked base's val dump (its θ source) and its test dump must agree (09-REVIEW WR-02):
# report.SAME_SOURCE_KEYS without the checkpoint path, which a local checkpoint_identity binds (report.PATH_KEY; a test
# keeps the two equal)
SAME_SOURCE_KEYS = (("system", "kind"), ("system", "checkpoint_identity"), ("temperature", "shipped"),
                    ("scorer", "device"), ("scorer", "dtype"), ("scorer", "amp_enabled"))


def same_source(entry, val_side, test_side):
    """None when a masked base's val sidecar and test sidecar agree on SAME_SOURCE_KEYS (a missing key reads as None
    on both sides); SystemExit naming the base and the first key that differs otherwise. The val dump is ungated, so it
    could be rescored elsewhere (another device or scorer); its θ is applied only to a test dump of the same source,
    as report.py applies a val θ (report.same_source)."""
    for path in SAME_SOURCE_KEYS:
        a, b = val_side, test_side
        for k in path:
            a = a.get(k) if isinstance(a, dict) else None
            b = b.get(k) if isinstance(b, dict) else None
        if a != b:
            raise SystemExit("%s: the masked base's val dump (its θ source) and its test dump differ in %s (%r vs %r); "
                             "its val θ applies only to a test dump of the same source, so rescore the val dump with "
                             "the test dump's checkpoint, device and precision" % (_what(entry, "val"), ".".join(path),
                                                                                    a, b))


def _sorted_entries(runs):
    return sorted(runs["entries"], key=fewshot.entry_key)


def load_all(protocol, pin, runs, runs_sha256, dump_root=None, manifest_path=None):
    """{"Y": {split: gold}, "P": {(split, kind, key): P}, "dumps": {dump_run_id: {split: dump_sha256}}, "split":
    {split: manifest entry}} of every dump the record reads, in entry_key order (so the result does not depend on the
    order the manifest lists its entries in); every test dump scores one uid order, every val dump one too, each
    split is the frozen protocol's (sha256), and each masked base's val and test dumps are of one source
    (same_source, 09-REVIEW WR-02)."""
    manifest = _manifest(manifest_path)
    out = {"Y": {}, "P": {}, "dumps": {}, "split": {}}
    ref = {}
    for e in _sorted_entries(runs):
        sides = {}
        for split in (("val", "test") if e["stage"] == "base" else ("test",)):
            Y, P, uids, side, sha = load(e, split, pin, runs_sha256, dump_root, manifest, ref.get(split))
            sides[split] = side
            if split not in ref:
                want = ((protocol.get("splits") or {}).get(split) or {}).get("sha256")
                if side["split"]["sha256"] != want:
                    raise SystemExit("%s: the dump's %s split has sha256 %s, the frozen protocol's %s; score the "
                                     "protocol's split" % (_what(e, split), split, side["split"]["sha256"], want))
                ref[split], out["Y"][split] = uids, Y
                out["split"][split] = dict(side["split"])
            out["P"][(split, e["kind"], run_key(e))] = P
            out["dumps"].setdefault(e["dump_run_id"], {})[split] = sha
        if e["stage"] == "base":
            same_source(e, sides["val"], sides["test"])
    return out


def _ap(data, block, kind, key):
    r = metrics.new_tag_ap(data["Y"]["test"], data["P"][("test", kind, key)], block["tags"])
    return {"per_tag": r["per_tag"], "macro": r["macro"]}


def mean_delta(draws):
    """The aggregate criterion (d) reads: float(np.mean) of the per-draw deltas in float64 (pilot.verdict._fewshot)."""
    return float(np.mean(np.asarray(draws, dtype=np.float64)))


def k_block(block, data, k, mode):
    """{"per_draw": [{"draw", "laya", "berturk", "delta"}], "draws", "mean_delta"} of the new-tag macro-AP at `k`
    (k = 0: each draw's masked base, base_seed_of(draw, mode)), delta = Laya - BERTurk, in draw order."""
    per = []
    for d in range(len(block["draws"])):
        key = "s%d" % fewshot.base_seed_of(block, d, mode) if k == 0 else "d%d-k%d" % (d, k)
        laya, bert = (_ap(data, block, kind, key)["macro"] for kind in KINDS)
        per.append({"draw": d, "laya": laya, "berturk": bert, "delta": laya - bert})
    draws = [p["delta"] for p in per]
    return {"per_draw": per, "draws": draws, "mean_delta": mean_delta(draws)}


def _primary_k(block):
    k = (block["rules"].get("primary_metric") or {}).get("k")
    if k not in block["k_values"]:
        raise SystemExit("the frozen primary metric's k %r is not one of the frozen k values %s; restore protocol.json "
                         "(`git checkout -- protocol.json`)" % (k, block["k_values"]))
    return k


def _primary(block, data, mode):
    k = _primary_k(block)
    return dict({"k": k, "metric": PRIMARY_METRIC}, **k_block(block, data, k, mode))


def primary(protocol, pin, runs, runs_sha256, dump_root=None, manifest_path=None):
    """The primary block {"k", "metric", "per_draw", "draws", "mean_delta"} (see the module docstring) from the dumps
    the runs manifest `runs` lists."""
    block = fewshot._block(protocol)
    return _primary(block, load_all(protocol, pin, runs, runs_sha256, dump_root, manifest_path), runs["mode"])


def subset_micro_f1(per_tag, tags):
    """Micro-F1 2TP / (2TP + FP + FN) of the tp/fp/fn counts of pilot.metrics.metrics()'s per_tag, summed over `tags`;
    0.0 when the denominator is 0."""
    tp = sum(per_tag[t]["tp"] for t in tags)
    fp = sum(per_tag[t]["fp"] for t in tags)
    fn = sum(per_tag[t]["fn"] for t in tags)
    den = 2 * tp + fp + fn
    return 2 * tp / den if den else 0.0


def _f1(data, kind, key, sel, held, old):
    m = metrics.metrics(data["Y"]["test"], data["P"][("test", kind, key)], metrics.theta_vector(sel))["per_tag"]
    return {"new": subset_micro_f1(m, held), "old": subset_micro_f1(m, old)}


def trained_theta(Y_val, P_val, held):
    """THETA_TRAINED_RULE (post hoc, descriptive): pilot.metrics.select_theta of a masked base's val dump with the
    held-out columns excluded (gold False, P 0.0: no TP, FP or FN at any grid θ), so the global θ is the val micro-F1
    optimum of the trained columns and a held-out tag (no val positive left) gets no θ of its own."""
    cols = [metrics.TAGS.index(t) for t in held]
    Y, P = np.array(Y_val, dtype=bool), np.array(P_val, dtype=np.float64)
    Y[:, cols], P[:, cols] = False, 0.0
    return metrics.select_theta(Y, P)


def _theta_row(sel):
    return {"global": sel["global"], "per_tag": dict(sel["per_tag"]), "n_per_tag": len(sel["per_tag"])}


def _f1_block(data, block, entries, mode, sels, held, old):
    """{kind: {"base": {key: {new, old}}, "continued": {key: {new, old, forgetting, base_seed}}}} of the micro-F1 at
    each masked base's θ in `sels` ({(kind, seed): select_theta result}), forgetting = old - the base's old."""
    f1 = {kind: {"base": {}, "continued": {}} for kind in KINDS}
    for e in entries:
        if e["stage"] == "base":
            f1[e["kind"]]["base"][run_key(e)] = _f1(data, e["kind"], run_key(e), sels[(e["kind"], e["seed"])], held,
                                                    old)
    for e in entries:
        if e["stage"] == "continued":
            b = fewshot.base_seed_of(block, e["draw"], mode)
            row = _f1(data, e["kind"], run_key(e), sels[(e["kind"], b)], held, old)
            row.update(forgetting=row["old"] - f1[e["kind"]]["base"]["s%d" % b]["old"], base_seed=b)
            f1[e["kind"]]["continued"][run_key(e)] = row
    return f1


def ap_delta_bootstrap(Y, PA_by_draw, PB_by_draw, tags, B, seed, level):
    """BOOT_PAIRING: {"delta", "ci", "B", "seed", "level", "method", "n_undefined", "pairing"} of the mean over draws of
    (A - B) new-tag macro-AP. Y is the gold [n, 48], PA_by_draw / PB_by_draw the per-draw probabilities of A (Laya) and
    B (BERTurk) on the same messages in the same order. delta is the full-data mean (mean_delta); every defined
    replicate is the same statistic on the rows idx[b] of one bootstrap_weights(n, B, seed) draw (each AP exactly
    pilot.metrics.average_precision of the resampled column, the macro new_tag_ap's mean); ci is the percentile interval
    at `level` over the defined replicates (None when none is defined)."""
    Y = np.asarray(Y, dtype=bool)
    if len(PA_by_draw) != len(PB_by_draw) or not PA_by_draw:
        raise SystemExit("the bootstrap needs one A and one B probability array per draw, got %d and %d"
                         % (len(PA_by_draw), len(PB_by_draw)))
    tags = list(tags)
    q = metrics._percentiles(level)
    observed = [metrics.new_tag_ap(Y, a, tags)["macro"] - metrics.new_tag_ap(Y, b, tags)["macro"]
                for a, b in zip(PA_by_draw, PB_by_draw)]       # validates every array; refuses a tag with no positive
    cols = [metrics.TAGS.index(t) for t in tags]
    Ys = Y[:, cols]
    pairs = [(np.asarray(a, dtype=np.float64)[:, cols], np.asarray(b, dtype=np.float64)[:, cols])
             for a, b in zip(PA_by_draw, PB_by_draw)]
    idx, _ = metrics.bootstrap_weights(Y.shape[0], B, seed)    # ONE draw shared by every model and draw (pairing)

    def macro(y, s):
        return float(sum(metrics.average_precision(y[:, i], s[:, i]) for i in range(len(cols))) / len(cols))

    reps, undefined = [], 0
    for r in idx:
        Yr = Ys[r]
        if not Yr.any(axis=0).all():                         # a held-out tag without a positive: AP undefined
            undefined += 1
            continue
        reps.append(mean_delta([macro(Yr, a[r]) - macro(Yr, b[r]) for a, b in pairs]))
    ci = [float(v) for v in np.percentile(np.asarray(reps, dtype=np.float64), q)] if reps else None
    return {"delta": mean_delta(observed), "ci": ci, "B": int(B), "seed": seed, "level": level,
            "method": metrics.BOOT_METHOD, "n_undefined": int(undefined), "pairing": BOOT_PAIRING}


def _boot_settings(protocol):
    boot = protocol.get("bootstrap") if isinstance(protocol, dict) else None
    if not isinstance(boot, dict) or not all(key in boot for key in ("B", "seed", "level")):
        raise SystemExit("the frozen protocol holds no bootstrap block with B, seed and level; restore protocol.json "
                         "(`git checkout -- protocol.json`)")
    return boot["B"], boot["seed"], boot["level"]


def _check(protocol, pin, runs):
    """The recorded mode, after check_runs of `runs` in entry_key order under read_mode(pin) (any listing order)."""
    mode = fewshot.read_mode(pin)
    fewshot.check_runs(dict(runs, entries=_sorted_entries(runs)), protocol, pin, mode)
    return mode


def record_of(protocol, pin, runs, runs_sha256, runs_path, dump_root=None, manifest_path=None):
    """The results record (RESULTS_SCHEMA) of a runs manifest `runs` whose bytes hash to `runs_sha256`, read from
    `runs_path`, and the dumps it lists; the entries may come in any order."""
    block = fewshot._block(protocol)
    mode = _check(protocol, pin, runs)
    with open(fewshot.DECISION_PATH, "rb") as f:
        dec_sha = sha256_bytes(f.read())
    if runs["decision"]["sha256"] != dec_sha:
        raise SystemExit("the runs manifest names the D-08 decision record sha256 %s, but %s hashes to %s; rebuild the "
                         "runs manifest" % (runs["decision"]["sha256"], repo_relpath(fewshot.DECISION_PATH), dec_sha))
    data = load_all(protocol, pin, runs, runs_sha256, dump_root, manifest_path)
    fallback = block["rules"]["compute_fallback"]
    B, seed, level = _boot_settings(protocol)
    entries = _sorted_entries(runs)
    held = list(block["tags"])
    old = [t for t in metrics.TAGS if t not in set(held)]
    ap = {kind: {"base": {}, "continued": {}} for kind in KINDS}
    theta, sels = {kind: {} for kind in KINDS}, {}
    theta_tr, sels_tr = {kind: {} for kind in KINDS}, {}       # the post hoc variant (THETA_TRAINED_RULE)
    wall = {kind: {} for kind in KINDS}
    for e in entries:
        ap[e["kind"]][e["stage"]][run_key(e)] = _ap(data, block, e["kind"], run_key(e))
        if e["stage"] == "base":
            Yv, Pv = data["Y"]["val"], data["P"][("val", e["kind"], run_key(e))]
            sels[(e["kind"], e["seed"])] = sel = metrics.select_theta(Yv, Pv)          # A1, the frozen reading
            theta[e["kind"]][run_key(e)] = _theta_row(sel)
            sels_tr[(e["kind"], e["seed"])] = sel = trained_theta(Yv, Pv, held)
            theta_tr[e["kind"]][run_key(e)] = _theta_row(sel)
        else:
            wall[e["kind"]][run_key(e)] = e["wall_clock_s"]
    f1 = _f1_block(data, block, entries, mode, sels, held, old)
    f1_tr = _f1_block(data, block, entries, mode, sels_tr, held, old)
    by_k, boot = {}, {}
    for k in [0] + sorted(block["k_values"]):
        by_k[str(k)] = k_block(block, data, k, mode)
        keys = ["s%d" % fewshot.base_seed_of(block, d, mode) if k == 0 else "d%d-k%d" % (d, k)
                for d in range(len(block["draws"]))]
        boot[str(k)] = ap_delta_bootstrap(data["Y"]["test"], [data["P"][("test", "laya", key)] for key in keys],
                                          [data["P"][("test", "berturk", key)] for key in keys], held, B, seed, level)
    k_primary = _primary_k(block)
    return {
        "schema": RESULTS_SCHEMA, "protocol_sha256": pin,
        "runs": {"path": repo_relpath(runs_path), "sha256": runs_sha256},
        "decision": {"path": repo_relpath(fewshot.DECISION_PATH), "sha256": dec_sha, "mode": mode,
                     "disclosure": fallback["disclosure"] if mode == "fallback" else None},
        "tags": list(block["tags"]), "k_values": list(block["k_values"]),
        "draws": [{"draw": d["draw"], "seed": d["seed"], "base_seed": d["base_seed"],
                   "base_seed_used": fewshot.base_seed_of(block, i, mode), "digest": d["digest"]}
                  for i, d in enumerate(block["draws"])],
        "split": {s: {"sha256": data["split"][s]["sha256"], "uids_sha256": data["split"][s]["uids_sha256"],
                      "n": int(data["Y"][s].shape[0]),
                      "positives": {t: int(data["Y"][s][:, metrics.TAGS.index(t)].sum()) for t in block["tags"]}}
                  for s in ("test", "val")},
        "primary": dict({"k": k_primary, "metric": PRIMARY_METRIC}, **by_k[str(k_primary)]),
        "by_k": by_k, "ap": ap, "theta": theta, "f1": f1, "bootstrap": boot,
        "theta_trained": {"label": POST_HOC_LABEL, "rule": THETA_TRAINED_RULE, "cause": THETA_CAUSE,
                          "theta": theta_tr, "f1": f1_tr},
        "wall_clock_s": wall, "wall_rule": runs["wall_rule"],
        "positives": {"d%d" % d["draw"]: d["positives"] for d in block["draws"]},
        "dumps": data["dumps"],
        "rules": {"ap": metrics.AP_RULE, "primary": block["rules"]["primary_metric"]["rule"], "theta": THETA_RULE,
                  "laya_rows": fewshot.LAYA_ROWS_RULE, "bootstrap": BOOT_PAIRING, "wall": fewshot.WALL_RULE},
        "notes": list(NOTES),
    }


def build_record(protocol, pin, runs_path=None, dump_root=None, manifest_path=None):
    """The results record of the committed runs manifest (pilot.fewshot.RUNS_PATH at call time, or `runs_path`) and
    the dumps it lists; SystemExit when there is no runs manifest or any check fails."""
    path = fewshot.RUNS_PATH if runs_path is None else runs_path
    runs, sha = fewshot.read_runs(protocol, pin, path)
    if runs is None:
        raise SystemExit("%s not found: no runs manifest, so no few-shot results; write it with `python fewshot.py "
                         "--runs-manifest CKPT... --out results/fewshot/runs.json` and score its checkpoints first"
                         % repo_relpath(path))
    return record_of(protocol, pin, runs, sha, path, dump_root, manifest_path)


def record_bytes(record):
    """Key-sorted JSON, indent 2, UTF-8, one trailing newline (no timestamp: a rebuild is byte-identical)."""
    return (json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_record(record, path):
    """Write the record atomically (a rebuild overwrites it); returns `path`."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    write_atomic(path, record_bytes(record))
    return path


def read_record(path):
    """(record, sha256 of the file bytes) of a few-shot results record; SystemExit on a missing file, invalid JSON or
    another schema."""
    if not os.path.isfile(path):
        raise SystemExit("%s not found; build it with `make fewshot-reports`" % path)
    with open(path, "rb") as f:
        raw = f.read()
    try:
        rec = json.loads(raw.decode("utf-8"))
    except ValueError as e:
        raise SystemExit("%s is not valid JSON (%s); rebuild it with `make fewshot-reports`" % (path, e))
    if not isinstance(rec, dict) or rec.get("schema") != RESULTS_SCHEMA:
        raise SystemExit("%s: schema %r, expected %s" % (path, rec.get("schema") if isinstance(rec, dict) else None,
                                                         RESULTS_SCHEMA))
    return rec, sha256_bytes(raw)


# ---- the README Phase 9 block (rounding for display only) ----

def _md(header, rows):
    return ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"] + \
        ["| " + " | ".join(r) + " |" for r in rows]


def _ap_text(x):
    return "n/a" if x is None else "%.3f" % x


def _points(x):
    return "n/a" if x is None else "%+.2f points" % (x * 100)


def _ci(ci):
    return "n/a (every replicate undefined)" if ci is None else "[%+.2f, %+.2f] points" % (ci[0] * 100, ci[1] * 100)


def _k_label(k):
    return "k = 0 (masked base)" if k == "0" else "k = %s" % k


def _f1_table(f1):
    """The micro-F1 and forgetting table of an f1 block ({kind: {"base", "continued"}})."""
    rows = []
    for kind, name in (("laya", "Laya"), ("berturk", "BERTurk")):
        for key in sorted(f1[kind]["base"], key=lambda x: int(x[1:])):
            f = f1[kind]["base"][key]
            rows.append((name, "base seed %s, k = 0" % key[1:], "%.3f" % f["new"], "%.3f" % f["old"], ""))
        for key in sorted(f1[kind]["continued"], key=lambda x: tuple(int(v[1:]) for v in x.split("-"))):
            f = f1[kind]["continued"][key]
            d, k = (int(v[1:]) for v in key.split("-"))
            rows.append((name, "draw %d, k = %d" % (d, k), "%.3f" % f["new"], "%.3f" % f["old"],
                         _points(f["forgetting"])))
    return _md(("Model", "Run", "New-tag micro-F1", "Old-tag micro-F1", "Old-tag change vs base (negative = "
                                                                       "forgetting)"), rows)


def render_markdown(record):
    """The README Phase 9 block of a few-shot results record (deterministic; points with 2 decimals, AP and F1 with
    3)."""
    r = record
    mode, dec = r["decision"]["mode"], r["decision"]
    p = r["primary"]
    lines = ["### Few-shot new-intent results (Phase 9)", "",
             "Protocol `%s`, runs manifest `%s`, decision record `%s`, mode %s. Held-out tags: %s; k values %s; "
             "draws %s." % (r["protocol_sha256"][:12], r["runs"]["sha256"][:12], dec["sha256"][:12], mode,
                            ", ".join(r["tags"]), ", ".join("%d" % k for k in r["k_values"]),
                            ", ".join("%d (digest `%s`, base seed %d)" % (d["draw"], d["digest"][:12],
                                                                          d["base_seed_used"]) for d in r["draws"]))]
    if dec.get("disclosure"):
        lines += ["", "Compute fallback: %s." % dec["disclosure"].rstrip(".")]
    lines += ["", "Test split: %d messages; positives per held-out tag: %s." % (
        r["split"]["test"]["n"], ", ".join("%s %d" % (t, r["split"]["test"]["positives"][t]) for t in r["tags"]))]

    lines += ["", "#### The primary metric: new-tag macro-AP at k = %d (test, raw T = 1)" % p["k"], ""]
    rows = [("%d" % d["draw"], _ap_text(d["laya"]), _ap_text(d["berturk"]), _points(d["delta"]))
            for d in p["per_draw"]]
    rows.append(("mean over draws", "", "", _points(p["mean_delta"])))
    lines += _md(("Draw", "Laya", "BERTurk", "Laya - BERTurk"), rows)
    lines += ["", "Criterion (d)'s few-shot path reads the mean over draws of Laya - BERTurk at k = %d, unrounded."
              % p["k"]]

    lines += ["", "#### New-tag macro-AP by k (mean over draws)", ""]
    rows = []
    for k in sorted(r["by_k"], key=int):
        b = r["by_k"][k]
        rows.append((_k_label(k), _ap_text(float(np.mean([d["laya"] for d in b["per_draw"]]))),
                     _ap_text(float(np.mean([d["berturk"] for d in b["per_draw"]]))), _points(b["mean_delta"])))
    lines += _md(("k", "Laya", "BERTurk", "Laya - BERTurk"), rows)

    lines += ["", "#### Per-tag AP at k = %d (mean over draws)" % p["k"], ""]
    keys = ["d%d-k%d" % (d["draw"], p["k"]) for d in r["draws"]]
    rows = [(t, _ap_text(float(np.mean([r["ap"]["laya"]["continued"][key]["per_tag"][t] for key in keys]))),
             _ap_text(float(np.mean([r["ap"]["berturk"]["continued"][key]["per_tag"][t] for key in keys]))))
            for t in r["tags"]]
    lines += _md(("Tag", "Laya", "BERTurk"), rows)

    lines += ["", "#### Forgetting: micro-F1 at each masked base's val θ (test)", ""]
    lines += _f1_table(r["f1"])

    t = r["theta_trained"]
    lines += ["", "#### Post hoc, not frozen: micro-F1 at each masked base's val θ on its trained columns (test)", "",
              "**%s%s.** Rule: %s. Cause: %s." % (t["label"][0].upper(), t["label"][1:], t["rule"].rstrip("."),
                                                  t["cause"].rstrip(".")), ""]
    rows = []
    for kind, name in (("laya", "Laya"), ("berturk", "BERTurk")):
        for key in sorted(r["theta"][kind], key=lambda x: int(x[1:])):
            a, b = r["theta"][kind][key], t["theta"][kind][key]
            rows.append((name, key[1:], "%.2f" % a["global"], "%.2f" % b["global"],
                         "yes" if a == b else "no"))
    lines += _md(("Model", "Base seed", "Global θ, all 48 columns (frozen reading)",
                  "Global θ, trained columns (post hoc)", "Same θ under both rules"), rows)
    lines += [""] + _f1_table(t["f1"])

    lines += ["", "#### Bootstrap of the mean-over-draws delta (descriptive, not gated)", ""]
    rows = []
    for k in sorted(r["bootstrap"], key=int):
        b = r["bootstrap"][k]
        rows.append((_k_label(k), _points(b["delta"]), _ci(b["ci"]), "%d" % b["n_undefined"], "%d" % b["B"]))
    level = "%g%% CI" % (next(iter(r["bootstrap"].values()))["level"] * 100) if r["bootstrap"] else "CI"
    lines += _md(("k", "Laya - BERTurk", level, "Undefined replicates", "B"), rows)

    lines += ["", "#### Wall clock of each continued run (seconds, reported, not gated)", ""]
    rows = []
    for kind, name in (("laya", "Laya"), ("berturk", "BERTurk")):
        for key in sorted(r["wall_clock_s"][kind], key=lambda x: tuple(int(v[1:]) for v in x.split("-"))):
            d, k = (int(v[1:]) for v in key.split("-"))
            rows.append((name, "%d" % d, "%d" % k, "%.1f" % r["wall_clock_s"][kind][key]))
    lines += _md(("Model", "Draw", "k", "Seconds"), rows)

    lines += ["", "#### Realised positives per held-out tag in each support set", ""]
    rows = [("%s" % key[1:], k, *("%d" % r["positives"][key][k][t] for t in r["tags"]))
            for key in sorted(r["positives"], key=lambda x: int(x[1:]))
            for k in sorted(r["positives"][key], key=int)]
    lines += _md(("Draw", "k") + tuple(r["tags"]), rows)

    lines += ["", "#### Notes", ""]
    lines += ["- %s." % n.rstrip(".") for n in (r["rules"]["theta"], r["rules"]["bootstrap"])]
    lines += ["- %s%s." % (n[0].upper(), n[1:].rstrip(".")) for n in r["notes"]]
    return "\n".join(lines)
