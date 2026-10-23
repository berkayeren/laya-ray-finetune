"""Frozen decision protocol (REP-01, ROADMAP Phase 8 criterion 1): build, bytes, pinned reader and gate helpers.

REP-01 requires the decision rule, with a numeric threshold behind every criterion, and the few-shot protocol to be
written down and frozen before any fine-tuned model is scored on test. protocol.json at the repo root is that record
(A17). It is built once by build_protocol from the committed inputs and the code constants of pilot.verdict,
pilot.metrics and pilot.fewshot, written by the freeze commit (plan 08-10) and never rebuilt after it: from then on
the file is the source of truth. Its identity is the sha256 of its exact bytes (protocol_bytes: indent 2,
ensure_ascii False, one trailing newline, UTF-8), pinned in PROTOCOL_SHA256 by the same commit; the pin lives here,
never in the file it hashes. Every consumer (score.py's test and HOTELS gate, report.py --verdict, the few-shot
runner) calls require_frozen, which reads the file once, hashes those bytes, compares them to the pin and parses the
same bytes, so a re-serialised but equal file is refused too. require_frozen reads PROTOCOL_PATH and PROTOCOL_SHA256
at call time, so no consumer captures a stale pin.

gated_split(protocol, entry) names the gated split (test or hotels_probe) of a manifest entry whose sha256 and
uids_sha256 are the protocol's; final_seed_allowed(protocol, kind, dump_run_id) is true for exactly the three final
seeds of a system's committed final manifest.

Decisions taken as 08-RESEARCH recommended, under the user's standing delegation (2026-10-21): A5 (a seed-averaged
system uses its postfit record's theta.shipped at the record T), A14 (final seeds are scored on MPS, fp32,
--workers 0: the device the postfit T was fitted on), A15 (zero-shot Laya on HOTELS is an ungated reference row,
scored on cpu with Ray workers 2 x 4 CPUs like its committed val dump), A17 (protocol.json at the repo root, the pin
here) and A18 (every gating number uses the headline variant; raw is reported).

Stdlib only at import time: build_protocol and main import the numpy-using modules (pilot.metrics, pilot.dumps,
pilot.verdict) inside the function, so score.py's gate never pays for them.

usage:
    python -m pilot.protocol --build --out protocol.json    # write the protocol's bytes (refuses an existing --out)
    python -m pilot.protocol --check                         # require_frozen(): print the pinned sha256
"""
import argparse
import json
import os
import subprocess

from pilot.data import (DEFAULT_MANIFEST, REPO_ROOT, manifest_is_committed, sha256_bytes, sha256_file,
                        write_atomic)

SCHEMA = "protocol/1"
PROTOCOL_PATH = os.path.join(REPO_ROOT, "protocol.json")
# the sha256 of protocol.json's exact bytes; None until the freeze commit (plan 08-10) sets it
PROTOCOL_SHA256 = None
TITLE = "Laya vs BERTurk on Turkish banking intents: the frozen decision protocol (REP-01)"
UNITS = "fractions in [0, 1]; points = fraction x 100"
FROZEN = {
    "when": "before any fine-tuned model is scored on test or HOTELS (REP-01, ROADMAP Phase 8 criterion 1)",
    "why": ("every threshold, the seed-averaged bootstrap procedure and the few-shot protocol are fixed before a "
            "fine-tuned test number exists, so nothing can be tuned to the result"),
    "identity": ("the sha256 of this file's exact bytes, pinned in pilot.protocol.PROTOCOL_SHA256 by the freeze "
                 "commit; every consumer calls pilot.protocol.require_frozen and refuses a missing or changed file"),
}
GATED = ("test", "hotels_probe")
SPLIT_ORDER = ("test", "hotels_probe", "val", "selection_train", "final_train")
SPLIT_KEYS = ("role", "sha256", "uids_sha256", "n_messages", "n_no_intent")
KINDS = ("laya", "berturk")
FINAL_SEEDS = [0, 1, 2]
POSTFIT_DIR = "results/postfit"
SCORING = {
    "splits": list(GATED),
    "device": "mps",
    "workers": 0,
    "dtype": "torch.float32",
    "amp_enabled": False,
    "dump_root": "results/dumps",
    "once": ("each final seed is scored once per split into results/dumps/<dump_run_id>/<split>; an existing dump is "
             "never replaced"),
    "refused": ["--force", "--n", "an --out-root other than results/dumps",
                "any checkpoint outside systems.*.final_seeds",
                "a device or worker count other than this block's",
                "a manifest other than the committed data/splits.json"],
    "why": "mps, fp32, in-process: the scorer of the val dumps the postfit T and θ were fitted on (A14)",
}
INPUT_PATHS = (
    "data/splits.json",
    "results/postfit/laya.json",
    "results/postfit/berturk.json",
    "results/postfit/laya-final.json",
    "results/postfit/berturk-final.json",
    "results/postfit/berturk-reference.json",
    "results/dumps/prior/val.json",
    "results/dumps/prior/test.json",
    "results/reports/majority.json",
    "results/reports/prior.json",
    "results/reports/laya-zeroshot-7b928d82.json",
    "results/latency/laya-mps.json",
    "results/latency/laya-cpu.json",
    "results/latency/berturk-mps.json",
    "results/latency/berturk-cpu.json",
)
ZEROSHOT = "laya-zeroshot-7b928d82"
REFERENCES = {
    "majority": {"run_id": "majority", "report": "results/reports/majority.json", "gated": False,
                 "hotels_probe": None},
    "prior": {"run_id": "prior", "report": "results/reports/prior.json", "gated": False, "hotels_probe": None},
    ZEROSHOT: {"run_id": ZEROSHOT, "report": "results/reports/%s.json" % ZEROSHOT, "gated": False,
               "hotels_probe": {"device": "cpu", "workers": 2, "cpus_per_worker": 4, "gated": False,
                                "once": True}},
}
REFERENCES_RULE = ("reference rows are reported, never gated; their θ is D-12 on their own committed val dump; "
                   "majority and prior are not scored on hotels_probe; zero-shot Laya is scored once on hotels_probe "
                   "as an ungated reference OOD row, on the scorer of its committed val dump (cpu, Ray workers 2 x 4 "
                   "CPUs; A15)")
VARIANT = {
    "headline": ("each final seed's shipped T (= its postfit record's T) with the record's theta.shipped (global plus "
                 "per-tag, pilot.metrics.theta_vector); every gating number uses it (A18)"),
    "raw": "T = 1 with the record's theta.raw; reported only, never gating",
    "references": "majority, prior and zero-shot Laya keep D-12 on their own val dump (pilot.metrics.select_theta)",
}
SEED_AVERAGING = {
    "rule": ("per seed s, P_s = sigmoid(logit_s / T_s) with T_s the record T; P = the mean over the 3 seeds per "
             "message and tag (pilot.metrics.seed_average); θ = the record's theta.shipped (global plus per-tag, "
             "theta_vector) applied to P; probabilities are averaged before any threshold, never metrics (EVAL-07, "
             "A5)"),
    "raw": "the same with T_s = 1 and the record's theta.raw; reported only",
    "theta_source": ("the postfit record's θ, chosen on val by one selection seed; a final seed has no clean val "
                     "(final_train holds the val folds), so θ is never re-selected on the final seeds or on test"),
}
BOOTSTRAP_PROCEDURE = ("EVAL-07: a paired message-level bootstrap of Laya - BERTurk over the seed-averaged "
                       "probabilities of each system; one index draw (pilot.metrics.bootstrap_weights(n, B, seed)) is "
                       "shared by both systems and by every metric; each replicate's delta is computed at each "
                       "system's fixed θ; the interval is the percentile interval of the replicate deltas; replicates "
                       "where a delta is undefined are counted and left out")


# ---- bytes, writer and the pinned reader ----

def protocol_bytes(protocol):
    """The exact bytes of a protocol: indent 2, ensure_ascii False, one trailing newline, UTF-8 (REP-01 encoding)."""
    return (json.dumps(protocol, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def write_protocol(protocol, path):
    """Write protocol_bytes(protocol) to `path` atomically; an existing path is refused (the protocol is frozen
    once). Returns the sha256 of the written bytes."""
    if os.path.lexists(path):
        raise SystemExit("%s exists; the protocol is frozen once and never rewritten (write it to a new path)" % path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    raw = protocol_bytes(protocol)
    write_atomic(path, raw)
    return sha256_bytes(raw)


def require_frozen(path=None, pin=None):
    """(protocol dict, sha256) of the frozen protocol; SystemExit naming the fix unless the file exists, the pin is
    set, the file's bytes hash to the pin, and those same bytes parse to a protocol/1 object. With no argument it
    reads pilot.protocol.PROTOCOL_PATH and PROTOCOL_SHA256 at call time."""
    path = PROTOCOL_PATH if path is None else path
    pin = PROTOCOL_SHA256 if pin is None else pin
    if not os.path.isfile(path):
        raise SystemExit("%s not found; the decision protocol must be frozen (plan 08-10) before any fine-tuned model "
                         "is scored on test or HOTELS, the verdict is computed or the few-shot runner starts" % path)
    if pin is None:
        raise SystemExit("the decision protocol is not frozen: pilot.protocol.PROTOCOL_SHA256 is unset; the freeze "
                         "commit (plan 08-10) pins the sha256 of %s" % path)
    with open(path, "rb") as f:
        raw = f.read()                                      # hash and parse the same bytes (no re-read window)
    have = sha256_bytes(raw)
    if have != pin:
        raise SystemExit("%s has sha256 %s, the frozen protocol is %s; the protocol may not change after the freeze "
                         "(restore it with `git checkout -- protocol.json`)" % (path, have, pin))
    try:
        protocol = json.loads(raw.decode("utf-8"))
    except ValueError as e:
        raise SystemExit("%s is not valid JSON (%s); restore it with `git checkout -- protocol.json`" % (path, e))
    if not isinstance(protocol, dict):
        raise SystemExit("%s is not a JSON object; restore it with `git checkout -- protocol.json`" % path)
    if protocol.get("schema") != SCHEMA:
        raise SystemExit("%s: schema %r, expected %s; restore it with `git checkout -- protocol.json`"
                         % (path, protocol.get("schema"), SCHEMA))
    return protocol, have


# ---- the gate helpers (score.py, fewshot.py, report.py) ----

def gated_split(protocol, entry):
    """"test" or "hotels_probe" when the manifest `entry` has both the sha256 and the uids_sha256 of that split in
    the protocol, else None."""
    splits = protocol.get("splits") or {}
    for name in GATED:
        s = splits.get(name) or {}
        if s.get("sha256") and entry.get("sha256") == s["sha256"] and entry.get("uids_sha256") == s.get("uids_sha256"):
            return name
    return None


def final_seeds(protocol, kind):
    """The protocol's final seeds of system `kind`: [{seed, train_run_id, checkpoint_name, dump_run_id}]."""
    system = (protocol.get("systems") or {}).get(kind)
    if not isinstance(system, dict) or not isinstance(system.get("final_seeds"), list):
        raise SystemExit("the protocol has no final seeds for system %r; systems: %s"
                         % (kind, sorted(protocol.get("systems") or {})))
    return system["final_seeds"]


def final_seed_allowed(protocol, kind, dump_run_id):
    """True when `dump_run_id` is one of the final seeds of system `kind` in the protocol (an unknown kind: False)."""
    if kind not in (protocol.get("systems") or {}):
        return False
    return any(s.get("dump_run_id") == dump_run_id for s in final_seeds(protocol, kind))


# ---- the blocks ----

def _read_json(root, rel):
    """(parsed JSON, sha256 of the same bytes) of a committed file under root; SystemExit naming the git fix."""
    p = os.path.join(root, rel)
    try:
        with open(p, "rb") as f:
            raw = f.read()
        return json.loads(raw.decode("utf-8")), sha256_bytes(raw)
    except (OSError, ValueError) as e:
        raise SystemExit("%s cannot be read (%s); it is a committed input of the protocol, restore it with git"
                         % (rel, e))


def _committed(manifest_path):
    if not manifest_is_committed(manifest_path):
        raise SystemExit("%s is not the committed data/splits.json; the protocol is built from the committed splits "
                         "only (drop --manifest, or run `make data`)" % manifest_path)
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def splits_block(manifest_path=DEFAULT_MANIFEST):
    """The protocol's splits (test, hotels_probe with its ood_scoring_tags, val, selection_train, final_train): path,
    role, sha256, uids_sha256 and counts of the committed manifest; another manifest is refused."""
    m = _committed(manifest_path)
    out = {}
    for name in SPLIT_ORDER:
        e = m["splits"][name]
        out[name] = dict({"path": "data/" + e["path"]}, **{k: e[k] for k in SPLIT_KEYS})
        if name == "hotels_probe":
            out[name]["ood_scoring_tags"] = list(m["ood_scoring_tags"])
    return out


def _theta_summary(theta):
    return {"global": theta["global"], "n_per_tag": len(theta["per_tag"])}


def _system(root, kind):
    rec_rel, final_rel = "%s/%s.json" % (POSTFIT_DIR, kind), "%s/%s-final.json" % (POSTFIT_DIR, kind)
    record, rec_sha = _read_json(root, rec_rel)
    final, final_sha = _read_json(root, final_rel)
    if record.get("system_kind") != kind:
        raise SystemExit("%s is a postfit record of system %r, not %r; restore it with git"
                         % (rec_rel, record.get("system_kind"), kind))
    bound = final.get("postfit") or {}
    if final.get("schema") != "postfit-final/1" or bound.get("sha256") != rec_sha or bound.get("path") != rec_rel:
        raise SystemExit("%s is not bound to %s (postfit %r, the record's sha256 %s); restore both with git"
                         % (final_rel, rec_rel, bound, rec_sha))
    seeds = final.get("seeds") or []
    if [s.get("seed") for s in seeds] != FINAL_SEEDS or record["final"].get("seeds") != FINAL_SEEDS:
        raise SystemExit("%s lists seeds %s, the protocol needs exactly %s in order"
                         % (final_rel, [s.get("seed") for s in seeds], FINAL_SEEDS))
    c = record["chosen"]
    if (c.get("scorer") or {}).get("device") != SCORING["device"]:
        raise SystemExit("%s: the chosen candidate was scored on %r, the protocol scores final seeds on %s (the "
                         "device the postfit T was fitted on, A14)" % (rec_rel, (c.get("scorer") or {}).get("device"),
                                                                       SCORING["device"]))
    fs = [{"seed": s["seed"], "train_run_id": s["train_run_id"], "checkpoint_name": s["checkpoint_name"],
           "dump_run_id": s["dump_run_id"]} for s in seeds]
    ids = [s["dump_run_id"] for s in fs]
    if len(set(ids)) != len(ids):
        raise SystemExit("%s repeats a dump run id: %s" % (final_rel, ids))
    return {"kind": kind,
            "final_manifest": {"path": final_rel, "sha256": final_sha},
            "postfit": {"path": rec_rel, "sha256": rec_sha},
            "chosen": {"dump_run_id": c["dump_run_id"], "epoch": c["epoch"], "epochs_schedule": c["epochs_schedule"],
                       "temperature": c["temperature"],
                       "theta": {"shipped": _theta_summary(c["theta"]["shipped"]),
                                 "raw": _theta_summary(c["theta"]["raw"])},
                       "encoder_lr": c["encoder_lr"], "head_lr": c["canonical"]["head_lr"]},
            "seeds": list(FINAL_SEEDS),
            "final_seeds": fs,
            "dump_run_ids": ids}


def systems_block(root=REPO_ROOT):
    """The protocol's systems (laya, berturk) from the committed final manifests and postfit records: each final
    manifest refused unless bound to its record's sha256 and holding exactly seeds 0, 1, 2."""
    return {kind: _system(root, kind) for kind in KINDS}


def inputs_block(root, systems):
    """{repo-relative path: sha256} of every committed file the verdict reads, plus each chosen candidate's val
    sidecar; a missing input is refused."""
    rels = list(INPUT_PATHS) + ["results/dumps/%s/val.json" % systems[k]["chosen"]["dump_run_id"] for k in KINDS]
    out = {}
    for rel in rels:
        p = os.path.join(root, rel)
        if not os.path.isfile(p):
            raise SystemExit("%s not found; it is a committed input of the protocol, restore it with git" % rel)
        out[rel] = sha256_file(p)
    return out


def git_code(root=REPO_ROOT):
    """{"git_head", "dirty"} of the checkout the protocol is built in (dirty: a tracked file differs from HEAD)."""
    def git(*args):
        return subprocess.run(("git",) + args, cwd=root, capture_output=True, text=True, check=True).stdout
    try:
        return {"git_head": git("rev-parse", "HEAD").strip(),
                "dirty": bool(git("status", "--porcelain", "--untracked-files=no").strip())}
    except (OSError, subprocess.CalledProcessError) as e:
        raise SystemExit("git is needed to record the code state of the protocol (%s)" % e)


def build_protocol(manifest_path=DEFAULT_MANIFEST, root=REPO_ROOT, code=None):
    """The complete protocol/1 dict from the committed inputs; deterministic for a given `code` (else git_code)."""
    splits = splits_block(manifest_path)
    systems = systems_block(root)
    return {
        "schema": SCHEMA,
        "title": TITLE,
        "units": UNITS,
        "frozen": dict(FROZEN),
        "inputs": inputs_block(root, systems),
        "splits": splits,
        "systems": systems,
        "references": {"rule": REFERENCES_RULE, "rows": json.loads(json.dumps(REFERENCES))},
        "scoring": json.loads(json.dumps(SCORING)),
        "variant": dict(VARIANT),
        "seed_averaging": dict(SEED_AVERAGING),
        "bootstrap": bootstrap_block(),
        "code": dict(code) if code is not None else git_code(root),
    }


def bootstrap_block():
    from pilot import metrics
    return {"B": 10000, "seed": 0, "level": 0.95, "coverage": 80, "unit": "message", "method": metrics.BOOT_METHOD,
            "procedure": BOOTSTRAP_PROCEDURE, "extra": metrics.BOOT_EXTRA_RULE}


# ---- CLI ----

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build", action="store_true", help="build the protocol from the committed inputs to --out")
    mode.add_argument("--check", action="store_true", help="require_frozen(): print the pinned sha256")
    ap.add_argument("--out", help="--build: the path to write (refused when it exists)")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST, help="--build: must be the committed data/splits.json")
    a = ap.parse_args(argv)
    if a.build:
        if not a.out:
            ap.error("--build needs --out PATH")
        if os.path.lexists(a.out):
            raise SystemExit("%s exists; the protocol is frozen once and never rewritten (write it to a new path)"
                             % a.out)
        sha = write_protocol(build_protocol(a.manifest), a.out)
        print("protocol -> %s sha256 %s" % (a.out, sha))
        return sha
    if a.out:
        ap.error("--out is a --build option")
    _, sha = require_frozen()
    print("protocol.json is frozen: sha256 %s" % sha)
    return sha


if __name__ == "__main__":
    main()
