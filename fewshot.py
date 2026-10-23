"""The few-shot runner of the frozen protocol (REP-01; ROADMAP Phase 8 criterion 3, Phase 9 criterion 1).

protocol.json's "fewshot" block (pilot.fewshot.protocol_block, frozen by plan 08-10) is the whole few-shot protocol:
the held-out tags, k values, draw seeds, negatives per positive, replay size, the source split and the three frozen
support-set draws with their uid lists and digests. Phase 9 trains and scores on those frozen lists only, unchanged.

Phase 8 ships one mode, --support-sets, the runner's entry point (08-RESEARCH Open Question 3). Phase 9 adds the base,
incremental and scoring modes to this entry point; every mode calls pilot.protocol.require_frozen() before anything
else and records the protocol_sha256 it ran under, so the runner refuses a missing, unpinned or changed protocol.json
before it reads a split or loads a model (ROADMAP Phase 8 criterion 1).

--support-sets, in this order:
1. pilot.protocol.require_frozen(): protocol.json exists, PROTOCOL_SHA256 is set, its bytes hash to the pin and parse
   to a protocol/1 object; SystemExit otherwise, before any split is read.
2. the manifest's selection_train bytes, read once and verified (pilot.data.verify_split_file, role train, the exact
   bytes parsed); the entry's sha256 and uids_sha256 must be the block's source, so the rows are the frozen source.
3. pilot.fewshot.check_draws(block, rows): every draw regenerates to exactly its frozen uid lists, realised
   positives and digest; SystemExit naming the draw otherwise.
4. prints, as JSON, the protocol sha256, the source, and per draw its number, seed, base seed, digest, support sizes
   per k, replay size and realised positives; never a uid list or message text.

--counts (09-01), in this order: require_frozen() first; the verified source rows (read_source); pilot.fewshot.
check_source (the frozen source and every frozen draw re-checked over those bytes, D-02); then counts(block, rows),
printed as JSON: the masked-base and per-(draw, k) row, item and optimizer-step counts of both trainers, computed from
the frozen uid lists only (the D-08 projection consumes them). Counts only, never a uid list or message text.

--projection --out PATH (09-04, D-08), in this order: require_frozen() first; a fresh run refuses an existing PATH
(the decision is recorded once); the verified source and its counts as --counts; results/compute/
compute_checkpoint.json read once (its sha256 binds the record); pilot.compute.fewshot_projection on the committed
rates, the counts and the protocol's val and test sizes; pilot.compute.fewshot_decision applies the frozen
compute_fallback trigger exactly (fallback when the projection exceeds pilot.fewshot.FALLBACK_HOURS); the record
(schema fewshot-projection/1: protocol_sha256, compute, counts, projection, trigger, decision, notes; no timestamp) is
written atomically as key-sorted JSON (indent 2, one trailing newline) and printed. With --ladder-failure TEXT it
rewrites an existing primary record of this protocol into the fallback record naming the failure (the second frozen
trigger, a primary base run that still fails after the LAYA-03 ladder); it refuses a missing PATH, a fallback record
and a record of another protocol. Both trainers read the mode through pilot.fewshot.read_mode.

--inspect CKPT... (09-05, D-01, D-04): require_frozen() first, the mode from the D-08 decision record; then each
checkpoint on its own (entry_of): its kind from its files (rl_agent_config.json with training.ray_train: Laya;
berturk_train.json: BERTurk), its identity and dump run id exactly as score.py computes them (pilot.weights.identity and
pilot.dumps.run_id; pilot.berturk.identity and dump_run_id), its record read from the bytes that identity hashed, a run
id that is its canonical config's, a finished run (Laya: final.json of that run and one epoch row per trained epoch;
BERTurk: the record's epoch and rows), its binding re-checked under the recorded mode (a base: pilot.fewshot.check_base
against the protocol-pinned chosen canonical, the Laya rung the record's; a continued run: check_continued against the
base canonical and run id its record carries), and its masking proof (pilot.fewshot.proof_problems and check_entry). It
prints one runs-manifest entry per checkpoint (pilot.fewshot.ENTRY_KEYS: ids, digests, counts, wall clock by
WALL_RULE and losses; never a uid list or message text).

--runs-manifest CKPT... --out PATH (09-05, D-04, D-07, D-09): require_frozen() first; an existing PATH is refused (the
allow-list is written once); every checkpoint through entry_of; every continued entry's base among the inputs (the
same train run id), its canonical base digests (base_model_sha256, base_config_sha256, base_files_sha256) that base
checkpoint's identity and its record's base canonical that base's own; a missing slot (pilot.fewshot.slots) is refused
naming it; then pilot.fewshot.check_runs, and the manifest (schema fewshot-runs/1: protocol_sha256, mode, the decision
record's path and sha256, the sorted tags, WALL_RULE, the entries sorted by (kind, stage, seed, draw, k)) is written
atomically as key-sorted JSON with one trailing newline. score.py's protocol gate admits exactly its dump run ids, on
test only, once (pilot.fewshot.admitted).

usage:
    python fewshot.py --support-sets                     # refused until protocol.json is frozen (plan 08-10)
    python fewshot.py --support-sets --manifest OTHER_DIR/splits.json
    python fewshot.py --counts                           # rows, items and steps of every Phase 9 training run
    python fewshot.py --projection --out results/fewshot/projection.json          # the D-08 record, written once
    python fewshot.py --projection --ladder-failure TEXT --out results/fewshot/projection.json   # primary -> fallback
    python fewshot.py --inspect ckpt-fewshot-base-s0 ckpt-fewshot-d0-k10   # one allow-list entry per checkpoint
    python fewshot.py --runs-manifest ckpt-fewshot-* ckpt-berturk-fewshot-* --out results/fewshot/runs.json   # once
"""
import argparse
import json
import math
import os

import pilot.data
from pilot import berturk, compute, dumps, fewshot, protocol, runs, spike
from pilot.data import REPO_ROOT, parse_jsonl, repo_relpath, sha256_bytes, verify_split_file, write_atomic

SOURCE_SPLIT = fewshot.SOURCE_SPLIT
# the committed Phase 4 compute record whose measured rates the D-08 projection uses (bound by its sha256)
COMPUTE_PATH = os.path.join(REPO_ROOT, "results", "compute", "compute_checkpoint.json")


def read_source(block, manifest_path):
    """The verified rows of the block's source split: the manifest's selection_train bytes, read once, verified and
    bound to the block's source sha256 and uids_sha256; SystemExit naming the source otherwise."""
    return read_source_entry(block, manifest_path)[1]


def read_source_entry(block, manifest_path):
    """(manifest entry, verified rows) of the block's source split, as read_source; a missing manifest is refused in
    one line (08-REVIEW IN-07)."""
    src = block.get("source") or {}
    if src.get("split") != SOURCE_SPLIT:
        raise SystemExit("the frozen few-shot source is %r, not %s; the protocol draws from %s only"
                         % (src.get("split"), SOURCE_SPLIT, SOURCE_SPLIT))
    if not os.path.isfile(manifest_path):
        raise SystemExit("%s not found; run `make data` to write the committed manifest and splits, or pass "
                         "--manifest PATH" % manifest_path)
    with open(manifest_path, encoding="utf-8") as f:
        m = json.load(f)
    entry = (m.get("splits") or {}).get(SOURCE_SPLIT)
    if not entry:
        raise SystemExit("%s registers no %s; run `make data`" % (manifest_path, SOURCE_SPLIT))
    path = os.path.join(os.path.dirname(os.path.realpath(manifest_path)), entry["path"])
    if not os.path.isfile(path):
        raise SystemExit("%s not found; run `make data` to write the registered splits" % path)
    with open(path, "rb") as f:
        raw = f.read()
    verify_split_file(path, manifest_path=manifest_path, roles=("train",), data=raw)
    for key in ("sha256", "uids_sha256"):
        if entry.get(key) != src.get(key):
            raise SystemExit("%s has %s %s, the frozen few-shot source is %s; the draws are bound to the source split "
                             "the protocol froze (run `make data`, or pass the manifest it was frozen from)"
                             % (path, key, entry.get(key), src.get(key)))
    return entry, parse_jsonl(raw)


def frozen_protocol():
    """(protocol, fewshot block, pin) of require_frozen(), the first call of every mode: nothing is read without it."""
    proto, pin = protocol.require_frozen()
    block = proto["fewshot"]
    if not isinstance(block, dict):
        raise SystemExit("protocol.json's fewshot block is not an object; restore it with `git checkout -- "
                         "protocol.json`")
    return proto, block, pin


def frozen_block():
    """(fewshot block, pin) of require_frozen(), the first call of every mode: nothing is read without it."""
    _, block, pin = frozen_protocol()
    return block, pin


def support_sets(manifest_path):
    """--support-sets: the frozen draws, re-checked against the verified source, as a printable summary."""
    block, pin = frozen_block()                                 # first: nothing is read without the frozen protocol
    rows = read_source(block, manifest_path)
    fewshot.check_draws(block, rows)
    out = {"protocol_sha256": pin, "source": block["source"],
           "draws": [{"draw": d["draw"], "seed": d["seed"], "base_seed": d["base_seed"], "digest": d["digest"],
                      "n_support": {k: len(v) for k, v in d["support"].items()}, "n_replay": len(d["replay"]),
                      "positives": d["positives"]} for d in block["draws"]]}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


def counts(block, rows):
    """The rows, items and optimizer steps of every Phase 9 training run, from the frozen uid lists over `rows` (the
    verified source): the masked base (every message, the held-out keys deleted before neg sampling; Laya items by
    pilot.runs.neg_items, steps by pilot.spike.optimizer_steps at the frozen micro-batch and grad-accum; BERTurk
    steps ceil(n / batch)) and per (draw, k), in draw then k order, the support and replay rows, the Laya items of
    the frozen item rule (pilot.fewshot.laya_item_count) and both trainers' steps per epoch. protocol_sha256 is left
    for the caller to fill."""
    tags, base = block["tags"], block["rules"]["base_runs"]
    laya, bert = base["laya"], base["berturk"]
    mb, ga, neg, batch = laya["micro_batch"], laya["grad_accum"], laya["neg"], bert["batch"]
    items = runs.neg_items(fewshot.masked_rows(rows, tags), neg)
    out = {"protocol_sha256": None, "source": block["source"],
           "base": {"messages": len(rows), "laya_items": items,
                    "laya_steps_per_epoch": spike.optimizer_steps(items, mb, ga),
                    "berturk_steps_per_epoch": math.ceil(len(rows) / batch)},
           "continued": [], "laya_rows_rule": fewshot.LAYA_ROWS_RULE}
    for i in range(len(block["draws"])):
        for k in block["k_values"]:
            sel = fewshot.selection(block, i, k)
            n_items = fewshot.laya_item_count(rows, sel, tags, neg)
            n_bert = len(fewshot.berturk_rows(rows, sel, tags))
            out["continued"].append({"draw": i, "k": k, "support_rows": len(sel["support"]),
                                     "replay_rows": len(sel["replay"]), "laya_items": n_items,
                                     "laya_steps_per_epoch": spike.optimizer_steps(n_items, mb, ga),
                                     "berturk_messages": n_bert,
                                     "berturk_steps_per_epoch": math.ceil(n_bert / batch)})
    return out


def counts_mode(manifest_path):
    """--counts: require_frozen first, the verified source, check_source, then counts(), printed as JSON."""
    block, pin = frozen_block()                                 # first: nothing is read without the frozen protocol
    entry, rows = read_source_entry(block, manifest_path)
    fewshot.check_source(block, entry, rows)
    out = counts(block, rows)
    out["protocol_sha256"] = pin
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


def record_bytes(record):
    """The decision record's bytes: key-sorted JSON, indent 2, UTF-8, one trailing newline (no timestamp, so a rebuild
    is byte-identical)."""
    return (json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def read_compute(path=COMPUTE_PATH):
    """(measured block, {"path", "sha256"}) of the committed compute record, from one read of its bytes; SystemExit on
    a missing file or another schema."""
    if not os.path.isfile(path):
        raise SystemExit("%s not found; the D-08 projection uses the committed Phase 4 rates (`git checkout -- "
                         "results/compute/`)" % path)
    with open(path, "rb") as f:
        raw = f.read()
    report = json.loads(raw.decode("utf-8"))
    if report.get("schema") != compute.SCHEMA or not isinstance(report.get("measured"), dict):
        raise SystemExit("%s: schema %r with no measured block, expected %s" % (path, report.get("schema"),
                                                                              compute.SCHEMA))
    return report["measured"], {"path": repo_relpath(path), "sha256": sha256_bytes(raw)}


def scored_sizes(proto):
    """{"val": n, "test": n} of the frozen protocol's splits (the messages every base and continued model is scored
    on); SystemExit when the protocol does not record them."""
    splits = proto.get("splits") or {}
    out = {}
    for name in ("val", "test"):
        n = (splits.get(name) or {}).get("n_messages")
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise SystemExit("the frozen protocol records no %s n_messages; restore protocol.json with `git checkout "
                             "-- protocol.json`" % name)
        out[name] = n
    return out


def write_record(out, record):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    write_atomic(out, record_bytes(record))


def projection_mode(manifest_path, out, ladder_failure=None):
    """--projection: the D-08 record, written once (see the module docstring); with `ladder_failure` the rewrite of an
    existing primary record of this protocol into the fallback record."""
    proto, block, pin = frozen_protocol()                       # first: nothing is read without the frozen protocol
    if ladder_failure is not None:
        if not os.path.isfile(out):
            raise SystemExit("%s not found; a ladder failure turns the recorded primary decision into the fallback, "
                             "so record it first with `python fewshot.py --projection --out %s`" % (out, out))
        mode = fewshot.read_mode(pin, out)          # schema, this protocol's sha256, a mode that agrees with its trigger
        if mode != "primary":
            raise SystemExit("%s already records the fallback; the decision is recorded once and a ladder failure "
                             "only turns a primary record into the fallback" % out)
        with open(out, encoding="utf-8") as f:
            record = json.load(f)
        if not isinstance(record.get("projection"), dict) or not isinstance(record["projection"].get("total_s"),
                                                                            (int, float)):
            raise SystemExit("%s holds no projection with a total_s; a ladder failure rewrites the record `python "
                             "fewshot.py --projection` wrote (`git checkout -- %s`)" % (out, repo_relpath(out)))
        record.update(compute.fewshot_decision(record["projection"], block, ladder_failure))
    else:
        if os.path.lexists(out):
            raise SystemExit("%s exists; the D-08 decision is recorded once, before any base run (a primary base run "
                             "that still fails after the LAYA-03 ladder turns it into the fallback: pass "
                             "--ladder-failure TEXT)" % out)
        entry, rows = read_source_entry(block, manifest_path)
        fewshot.check_source(block, entry, rows)
        c = counts(block, rows)
        c["protocol_sha256"] = pin
        measured, ref = read_compute()
        projection = compute.fewshot_projection(measured, c, block, scored_sizes(proto))
        record = dict({"schema": fewshot.DECISION_SCHEMA, "protocol_sha256": pin, "compute": ref, "counts": c,
                       "projection": projection}, **compute.fewshot_decision(projection, block))
        record["notes"] = compute.fewshot_notes(measured, projection) + [
            "Counts: the masked-base Laya items are pilot.runs.neg_items of selection_train with the held-out keys "
            "deleted; each continued run's Laya items are pilot.fewshot.laya_item_count of its frozen draw (the "
            "frozen laya_item_rule) and its BERTurk messages the support plus replay rows (fewshot.py --counts)",
            "Scored messages: the frozen protocol's val and test splits (%d and %d messages)"
            % (projection["n_scored"]["val"], projection["n_scored"]["test"])]
    write_record(out, record)
    print(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True))
    return record


# ---- 09-05: the allow-list of few-shot checkpoints (D-01, D-04, D-09) ----

def _read_bound(path, sha, what):
    """The bytes of `path`, refused unless they hash to `sha` (the digest the checkpoint identity recorded)."""
    with open(path, "rb") as f:
        raw = f.read()
    if sha256_bytes(raw) != sha:
        raise SystemExit("%s changed after it was hashed (%s); rerun the command" % (what, os.path.basename(path)))
    return raw


def checkpoint_kind(ckpt):
    """'laya' or 'berturk' from the files of a checkpoint dir; SystemExit for a missing dir, both or neither."""
    if not os.path.isdir(ckpt):
        raise SystemExit("%s is not a checkpoint dir; pass a ray_train.py or berturk_train.py few-shot checkpoint"
                         % ckpt)
    laya = os.path.isfile(os.path.join(ckpt, "rl_agent_config.json"))
    bert = os.path.isfile(os.path.join(ckpt, berturk.RECORD))
    if laya == bert:
        raise SystemExit("%s holds %s; a few-shot checkpoint is a ray_train.py checkpoint (rl_agent_config.json) or a "
                         "berturk_train.py one (%s)" % (ckpt, "both" if laya else "neither", berturk.RECORD))
    return "laya" if laya else "berturk"


def _laya_run(ckpt):
    """(identity, dump run id, run id, canonical, record, wall clock, losses) of a finished ray_train.py checkpoint,
    read from the config bytes its identity hashed; SystemExit on a missing run record, a run id that is not its
    canonical's, or an unfinished run (no final.json of this run, or not one epoch row per trained epoch)."""
    from pilot.weights import identity
    ident = identity(ckpt)
    rid = dumps.run_id("laya", ident)
    cfg = json.loads(_read_bound(os.path.join(ckpt, "rl_agent_config.json"), ident["config_sha256"],
                                 "rl_agent_config.json").decode("utf-8"))
    tr = cfg.get("training") if isinstance(cfg, dict) else None
    rt = tr.get("ray_train") if isinstance(tr, dict) else None
    if not isinstance(rt, dict) or not isinstance(rt.get("canonical"), dict) or not isinstance(rt.get("record"),
                                                                                                dict):
        raise SystemExit("%s holds no ray_train.py run record (training.ray_train run_id, canonical, record); a "
                         "few-shot checkpoint is the --out of a ray_train.py --fewshot-* run" % ckpt)
    run, canon, rec = rt.get("run_id"), rt["canonical"], rt["record"]
    if run != runs.run_id(canon):
        raise SystemExit("%s names run %r, but its canonical config is run %s; pass an unedited checkpoint"
                         % (ckpt, run, runs.run_id(canon)))
    final = os.path.join(ckpt, "final.json")
    try:
        with open(final, encoding="utf-8") as f:
            mark = json.load(f)
    except (OSError, ValueError):
        mark = None
    stop = canon.get("stop_after_epoch") or canon.get("epochs")
    rows = rec.get("epochs") or []
    if not isinstance(mark, dict) or mark.get("run_id") != run or tr.get("epoch") != stop or len(rows) != stop:
        done = "of this run" if isinstance(mark, dict) and mark.get("run_id") == run else "missing"
        raise SystemExit("%s is not a finished run of %s (final.json %s, epoch %r and %d epoch rows of %r); pass the "
                         "--out of a ray_train.py run that finished" % (ckpt, run, done, tr.get("epoch"), len(rows),
                                                                        stop))
    terms = [row.get(key) for row in rows for key in ("train_s", "save_s")] + [rec.get("load_s"),
                                                                              rec.get("final_save_s")]
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in terms):
        raise SystemExit("%s: the run record lacks a train_s, save_s, load_s or final_save_s (WALL_RULE); pass the "
                         "--out of a ray_train.py run that finished" % ckpt)
    return ident, rid, run, canon, rec, float(sum(terms)), [row.get("loss") for row in rows]


def _berturk_run(ckpt):
    """(identity, dump run id, run id, canonical, record, wall clock, losses) of a finished berturk_train.py
    checkpoint, from the berturk_train.json bytes its identity hashed (pilot.berturk.read_record); SystemExit on a
    missing file, a run id that is not its canonical's, or an unfinished run."""
    berturk.require_files(ckpt, "few-shot checkpoint")
    ident = berturk.identity(ckpt)
    rid = berturk.dump_run_id(ident)
    doc = berturk.read_record(ckpt, ident)
    run, canon, rec = doc["run_id"], doc["canonical"], doc.get("record")
    if run != berturk.run_id(canon):
        raise SystemExit("%s names run %r, but its canonical config is run %s; pass an unedited checkpoint"
                         % (ckpt, run, berturk.run_id(canon)))
    stop = canon.get("stop_after_epoch") or canon.get("epochs")
    rows = (rec.get("epochs") or []) if isinstance(rec, dict) else []
    if not isinstance(rec, dict) or doc.get("epoch") != stop or len(rows) != stop:
        raise SystemExit("%s is not a finished run of %s (epoch %r and %d epoch rows of %r); pass the --out of a "
                         "berturk_train.py run that finished" % (ckpt, run, doc.get("epoch"), len(rows), stop))
    terms = [row.get("train_s") for row in rows] + [rec.get("load_s")]
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in terms):
        raise SystemExit("%s: the run record lacks a train_s or load_s (WALL_RULE); pass the --out of a "
                         "berturk_train.py run that finished" % ckpt)
    return ident, rid, run, canon, rec, float(sum(terms)), [row.get("loss") for row in rows]


def entry_of(ckpt, proto, pin, mode):
    """(runs-manifest entry, {"identity", "canonical", "base"}) of one few-shot checkpoint, re-validated on its own
    (see the module docstring); SystemExit naming the checkpoint and what fails."""
    block = fewshot._block(proto)
    kind = checkpoint_kind(ckpt)
    ident, rid, run, canon, rec, wall, losses = (_laya_run if kind == "laya" else _berturk_run)(ckpt)
    name = os.path.basename(os.path.normpath(os.path.abspath(ckpt)))
    bind = canon.get("fewshot")
    if not isinstance(bind, dict) or bind.get("stage") not in fewshot.STAGES:
        raise SystemExit("%s (run %s) is not a Phase 9 run: its canonical config carries no few-shot binding; only "
                         "a --fewshot-base or --fewshot-draw checkpoint is listed" % (name, run))
    stage, proof = bind["stage"], rec.get("fewshot")
    if not isinstance(proof, dict) or proof.get("stage") != stage or proof.get("protocol_sha256") != pin:
        raise SystemExit("%s (run %s): its record's masking proof is %s, not a %s proof under the frozen protocol %s; "
                         "retrain it with this trainer" % (name, run, "absent" if not isinstance(proof, dict) else
                                                           "stage %r of protocol %s" % (proof.get("stage"),
                                                                                        proof.get("protocol_sha256")),
                                                           stage, pin))
    if proof.get("tags") != sorted(block["tags"]):
        raise SystemExit("%s (run %s): its proof covers tags %r, not the frozen held-out tags %s"
                         % (name, run, proof.get("tags"), sorted(block["tags"])))
    try:
        if stage == "base":
            rung = fewshot.check_base(proto, pin, kind, canon, fewshot.chosen_canonical(proto, kind), mode)
            if kind == "laya" and rec.get("rung") != rung:
                raise SystemExit("its record names rung %r, its canonical config rung %s" % (rec.get("rung"), rung))
            base, draw, k, digest, base_seed, base_run = None, None, None, None, None, None
        else:
            base = proof.get("base")
            if not isinstance(base, dict) or not isinstance(base.get("canonical"), dict):
                raise SystemExit("its record names no base run id and canonical config")
            base_run = base.get("run_id")
            want = (runs.run_id if kind == "laya" else berturk.run_id)(base["canonical"])
            if base_run != want:
                raise SystemExit("its record's base names run %r, but that canonical config is run %s"
                                 % (base_run, want))
            draw, k, digest = bind.get("draw"), bind.get("k"), bind.get("digest")
            fewshot.check_continued(proto, pin, kind, canon, base["canonical"], base_run, draw, k, mode)
            if (proof.get("draw"), proof.get("k"), proof.get("digest")) != (draw, k, digest):
                raise SystemExit("its proof names draw %r k %r digest %r, its binding draw %r k %r digest %r"
                                 % (proof.get("draw"), proof.get("k"), proof.get("digest"), draw, k, digest))
            base_seed = fewshot.base_seed_of(block, draw, mode)
    except SystemExit as e:
        raise SystemExit("%s (run %s) no longer binds to the frozen protocol: %s"
                         % (name, run, " ".join(str(e.code).split()))) from None
    missing = [key for key in fewshot.PROOF_KEYS[(kind, stage)] if key not in proof]
    if missing:
        raise SystemExit("%s (run %s): its masking proof lacks %s; retrain it with this trainer"
                         % (name, run, ", ".join(missing)))
    entry = {"kind": kind, "stage": stage, "seed": canon.get("seed"), "draw": draw, "k": k, "digest": digest,
             "base_seed": base_seed, "base_run_id": base_run, "train_run_id": run, "dump_run_id": rid,
             "checkpoint_name": name, "rung": rec.get("rung") if kind == "laya" else None,
             "fallback": rec.get("fallback") if kind == "laya" else None,
             "weaker": rec.get("weaker") if kind == "laya" else None,
             "proof": {key: proof[key] for key in fewshot.PROOF_KEYS[(kind, stage)]}, "wall_clock_s": wall,
             "loss_by_epoch": losses}
    fewshot.check_entry(entry, block, mode, "%s (run %s)" % (name, run))
    return entry, {"identity": ident, "canonical": canon, "base": base}


def inspect_mode(ckpts):
    """--inspect: require_frozen first, the recorded mode, then entry_of per checkpoint, printed as JSON."""
    proto, _, pin = frozen_protocol()                           # first: nothing is read without the frozen protocol
    mode = fewshot.read_mode(pin)
    out = [entry_of(c, proto, pin, mode)[0] for c in ckpts]
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))
    return out


def _base_digests(kind, ident):
    """{base_model_sha256, base_config_sha256, base_files_sha256} a continued run's canonical config carries for a
    base checkpoint of this identity (as ray_train.py and berturk_train.py compute them)."""
    return {"base_model_sha256": ident["model_sha256"], "base_config_sha256": ident["config_sha256"],
            "base_files_sha256": dumps.files_digest(ident["files_sha256"])}


def runs_manifest_mode(ckpts, out):
    """--runs-manifest: the allow-list, written once (see the module docstring)."""
    proto, block, pin = frozen_protocol()                       # first: nothing is read without the frozen protocol
    mode = fewshot.read_mode(pin)
    if os.path.lexists(out):
        raise SystemExit("%s exists; the runs manifest is written once, before any few-shot test dump" % out)
    got = [entry_of(c, proto, pin, mode) for c in ckpts]
    bases = {(e["kind"], e["train_run_id"]): (e, info) for e, info in got if e["stage"] == "base"}
    for e, info in got:
        if e["stage"] != "continued":
            continue
        name = e["checkpoint_name"]
        if (e["kind"], e["base_run_id"]) not in bases:
            raise SystemExit("%s continues from base run %s, which is not among the checkpoints given; list its base "
                             "too" % (name, e["base_run_id"]))
        b, b_info = bases[(e["kind"], e["base_run_id"])]
        for key, want in sorted(_base_digests(e["kind"], b_info["identity"]).items()):
            if info["canonical"].get(key) != want:
                raise SystemExit("%s: its canonical %s is %s, but its base %s holds %s; the continued run must come "
                                 "from that base checkpoint, unedited" % (name, key, info["canonical"].get(key),
                                                                         b["checkpoint_name"], want))
        if info["base"]["canonical"] != b_info["canonical"]:
            raise SystemExit("%s: its record's base canonical config is not its base %s's own; pass the base it "
                             "continued from" % (name, b["checkpoint_name"]))
    entries = sorted((e for e, _ in got), key=fewshot.entry_key)
    have = {(e["kind"], e["stage"], e["seed"], e["draw"], e["k"]) for e in entries}
    missing = [s for s in fewshot.slots(block, mode) if s not in have]
    if missing:
        raise SystemExit("the few-shot set is incomplete: no checkpoint for %s; train and list every base and "
                         "continued run of the %s protocol" % ("; ".join(fewshot.slot_name(s) for s in missing),
                                                               mode))
    with open(fewshot.DECISION_PATH, "rb") as f:
        decision = {"path": repo_relpath(fewshot.DECISION_PATH), "sha256": sha256_bytes(f.read())}
    record = {"schema": fewshot.RUNS_SCHEMA, "protocol_sha256": pin, "mode": mode, "decision": decision,
              "tags": sorted(block["tags"]), "wall_rule": fewshot.WALL_RULE, "entries": entries}
    fewshot.check_runs(record, proto, pin, mode)
    write_record(out, record)
    print(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True))
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--support-sets", action="store_true",
                      help="re-check the frozen support-set draws against the verified source and print their digests")
    mode.add_argument("--counts", action="store_true",
                      help="print the rows, items and optimizer steps of every Phase 9 training run")
    mode.add_argument("--projection", action="store_true",
                      help="compute the D-08 projection and compute-fallback decision and write it to --out (once)")
    mode.add_argument("--inspect", nargs="+", metavar="CKPT",
                      help="re-validate few-shot checkpoints and print one runs-manifest entry each")
    mode.add_argument("--runs-manifest", nargs="+", metavar="CKPT",
                      help="write the allow-list of every few-shot checkpoint to --out (once)")
    ap.add_argument("--out", default=None, help="--projection: the decision record (results/fewshot/projection.json); "
                                                "--runs-manifest: the allow-list (results/fewshot/runs.json)")
    ap.add_argument("--ladder-failure", default=None, metavar="TEXT",
                    help="--projection: turn the recorded primary decision at --out into the fallback, naming the "
                         "primary base run that still failed after the LAYA-03 ladder")
    ap.add_argument("--manifest", default=None, help="the splits manifest (default data/splits.json)")
    a = ap.parse_args(argv)
    manifest = a.manifest or pilot.data.DEFAULT_MANIFEST
    if a.ladder_failure is not None and not a.projection:
        ap.error("--out and --ladder-failure belong to --projection (--out also to --runs-manifest)")
    if a.out is not None and not (a.projection or a.runs_manifest):
        ap.error("--out and --ladder-failure belong to --projection (--out also to --runs-manifest)")
    if a.manifest is not None and (a.inspect or a.runs_manifest):
        ap.error("--manifest belongs to --support-sets, --counts and --projection; --inspect and --runs-manifest read "
                 "the checkpoints' own records")
    if a.inspect:
        return inspect_mode(a.inspect)
    if a.runs_manifest:
        if not a.out:
            ap.error("--runs-manifest needs --out PATH (results/fewshot/runs.json)")
        return runs_manifest_mode(a.runs_manifest, a.out)
    if a.projection:
        if not a.out:
            ap.error("--projection needs --out PATH (results/fewshot/projection.json)")
        return projection_mode(manifest, a.out, a.ladder_failure)
    if a.counts:
        return counts_mode(manifest)
    if not a.support_sets:
        ap.error("pass a mode: --support-sets, --counts, --projection, --inspect or --runs-manifest")
    return support_sets(manifest)


if __name__ == "__main__":
    main()
