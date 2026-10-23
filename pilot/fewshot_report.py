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
  paired message by message) and be a full dump, and each split must be the frozen protocol's (sha256)

What it computes (D-12; protocol.json rules.primary_metric and rules.secondary_metrics):
- primary: per new tag the average precision by pilot.metrics.AP_RULE on P = sigmoid(logit) at raw T = 1 on test (AP is
  T-invariant), macro over the held-out tags (pilot.metrics.new_tag_ap), per draw Laya - BERTurk at k = the frozen
  primary k; draws = the per-draw deltas and mean_delta = float(np.mean(np.asarray(draws, dtype=np.float64))), exactly
  the value pilot.verdict.criterion_d checks
- ap: per model the per-tag AP and macro-AP of every masked base ("s<seed>", k = 0) and continued run ("d<draw>-k<k>")

Values are float64 and never rounded (rounding happens only in render_markdown). The record holds no timestamp, no uid
list, no message text and no local path (repo-relative paths, ids and digests), and is written as key-sorted JSON, so it
rebuilds byte for byte from the same dumps, whatever order the runs manifest lists its entries in. Nothing in it is
selected on test: θ comes only from each masked base's own val dump, AP needs no θ or T.

Numpy and stdlib only: never torch, scipy, sklearn, laya or Ray (sklearn's average_precision_score is a test oracle).

usage:
    from pilot import fewshot_report as fr
    rec = fr.build_record(protocol, pin)                # results/fewshot/runs.json and the dumps it lists
    fr.write_record(rec, fr.RECORD_PATH)
    python fewshot.py --report --out results/fewshot/fewshot.json      # make fewshot-reports
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


def _sorted_entries(runs):
    return sorted(runs["entries"], key=fewshot.entry_key)


def load_all(protocol, pin, runs, runs_sha256, dump_root=None, manifest_path=None):
    """{"Y": {split: gold}, "P": {(split, kind, key): P}, "dumps": {dump_run_id: {split: dump_sha256}}, "split":
    {split: manifest entry}} of every dump the record reads, in entry_key order (so the result does not depend on the
    order the manifest lists its entries in); every test dump scores one uid order, every val dump one too, and each
    split is the frozen protocol's (sha256)."""
    manifest = _manifest(manifest_path)
    out = {"Y": {}, "P": {}, "dumps": {}, "split": {}}
    ref = {}
    for e in _sorted_entries(runs):
        for split in (("val", "test") if e["stage"] == "base" else ("test",)):
            Y, P, uids, side, sha = load(e, split, pin, runs_sha256, dump_root, manifest, ref.get(split))
            if split not in ref:
                want = ((protocol.get("splits") or {}).get(split) or {}).get("sha256")
                if side["split"]["sha256"] != want:
                    raise SystemExit("%s: the dump's %s split has sha256 %s, the frozen protocol's %s; score the "
                                     "protocol's split" % (_what(e, split), split, side["split"]["sha256"], want))
                ref[split], out["Y"][split] = uids, Y
                out["split"][split] = dict(side["split"])
            out["P"][(split, e["kind"], run_key(e))] = P
            out["dumps"].setdefault(e["dump_run_id"], {})[split] = sha
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
    entries = _sorted_entries(runs)
    ap = {kind: {"base": {}, "continued": {}} for kind in KINDS}
    for e in entries:
        ap[e["kind"]][e["stage"]][run_key(e)] = _ap(data, block, e["kind"], run_key(e))
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
        "primary": _primary(block, data, mode),
        "ap": ap,
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
