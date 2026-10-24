"""The synthetic frozen protocol every few-shot test installs (a helper module, no test functions).

synthetic_protocol(tmp_path, synthetic_raw, monkeypatch, ...) runs pilot.data.prepare on the synthetic Multi3NLU++
snapshot (conftest), draws the few-shot support sets on its verified selection_train with pilot.fewshot.draw (tags card,
k 1 and 2, draw seed 0, 1 negative per positive, replay 2 by default), writes a REQUIRED_KEYS-complete protocol/1 file
whose fewshot block carries the settings, the source, the draws (base seed and digest included) and the rules (a deep
copy of pilot.fewshot.RULES, base_runs and continued_training replaced by `rules` when given), writes a minimal compute
decision record (schema fewshot-projection/1, this protocol's sha256, decision.mode and trigger fields that agree with
it), and pins both by monkeypatch: pilot.protocol.PROTOCOL_PATH and PROTOCOL_SHA256, pilot.data.DEFAULT_MANIFEST and
pilot.fewshot.DECISION_PATH. It returns a namespace (manifest, out_dir, protocol_path, pin, decision_path, block, rows,
protocol).

`splits` (optional) replaces the protocol's splits block, e.g. {"val": {"n_messages": 5}, "test": {"n_messages": 7}}
for fewshot.py --projection; by default the block is pilot.protocol.splits_block of the synthetic manifest
(pilot.data.DEFAULT_MANIFEST is pinned to it first), so score.py's protocol gate can name the synthetic test split.

postfit_systems(tmp_path, canonicals) writes one minimal postfit record per system kind ({"chosen": {"canonical":
...}}) and returns the protocol "systems" block that binds each by absolute path and sha256, with an empty final_seeds
list (pilot.protocol.final_seed_allowed reads it).

Planted few-shot checkpoints (no training): bench(tmp_path, synthetic_raw, monkeypatch, ...) installs the
synthetic protocol with PLANT_RULES (base seed 0 only, tiny recipes), postfit records binding the chosen canonical
configs of both kinds (chosen_laya, chosen_berturk over the synthetic selection_train sha256) and a scoring block on
cpu with workers 0. plant_laya(path, canonical, record) is tests/tiny_laya.tiny_checkpoint plus rl_agent_config.json
training.ray_train {run_id, canonical, record} and final.json; plant_berturk(path, canonical, record) the
tests/tiny_berturk.tiny_base files plus berturk_train.json. plant_fewshot_set(ns, root) plants one consistent set per
kind under root: the masked base of each allowed base seed and one continued run per frozen (draw, k), each with a
valid masking proof, and returns {name: dir} (names laya-base-s0, laya-d0-k1, berturk-d0-k2, ...).

Planted results inputs (no checkpoint, no model): plant_runs(ns, path) writes a complete runs manifest
(fewshot-runs/1) for the synthetic protocol `ns` straight from its slots, with made-up run ids, a valid masking proof
and a distinct wall clock per entry, and returns its path (pilot.fewshot.read_runs accepts it). plant_fewshot_dumps(ns,
runs, runs_sha256, dump_root, seed) writes, for every entry of that manifest, its dumps through pilot.dumps.write_dump
on the synthetic snapshot: a base's ungated val dump (no protocol_sha256, no fewshot_runs_sha256) and every entry's
gated test dump (protocol_sha256 the pin, fewshot_runs_sha256 the manifest's), each sidecar carrying the entry's system
kind, identity, dump run id, seed and training run id. The logits are seeded float32 normals unless `logits(entry,
split, Y, rng)` returns them. `extra` of synthetic_protocol sets top-level protocol keys (e.g. the bootstrap block).

fewshot_block(manifest, ...) is the few-shot block synthetic_protocol writes, on its own (the verdict's synthetic
protocol in tests/test_final_report.py carries one over its own snapshot).
"""
import copy
import json
import os
import types
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

import pilot.data
from conftest import SyntheticRaw, prepared
from pilot import fewshot as fs
from pilot import protocol as pr

DECISION_SCHEMA = "fewshot-projection/1"     # pilot.fewshot.DECISION_SCHEMA; tests/test_fewshot_binding.py checks it


def _rows(manifest: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """(selection_train manifest entry, parsed rows) of the synthetic snapshot at `manifest`."""
    with open(manifest, encoding="utf-8") as f:
        entry = json.load(f)["splits"]["selection_train"]
    with open(os.path.join(os.path.dirname(manifest), entry["path"]), "rb") as f:
        return entry, pilot.data.parse_jsonl(f.read())


def postfit_systems(tmp_path: Path, canonicals: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """{kind: {"kind", "postfit": {"path": absolute path, "sha256"}}} over records written under tmp_path/postfit."""
    out = {}
    os.makedirs(str(tmp_path / "postfit"), exist_ok=True)
    for kind, canon in canonicals.items():
        path = str(tmp_path / "postfit" / ("%s.json" % kind))
        raw = (json.dumps({"schema": "postfit/1", "chosen": {"canonical": canon}}, ensure_ascii=False, indent=2)
               + "\n").encode("utf-8")
        with open(path, "wb") as f:
            f.write(raw)
        out[kind] = {"kind": kind, "postfit": {"path": path, "sha256": pilot.data.sha256_bytes(raw)},
                     "final_seeds": []}
    return out


PLANT_LAYA = {"trainer": "ray_train.py", "epochs": 2, "stop_after_epoch": 1, "neg": 2, "encoder_lr": 2e-05,
              "head_lr": 0.0001, "device": "cpu", "workers": 1, "micro_batch": 2, "grad_accum": 1, "max_len": 32}
PLANT_BERTURK = {"trainer": "berturk_train.py", "epochs": 2, "stop_after_epoch": 1, "encoder_lr": 2e-05,
                 "head_lr": 0.0001, "batch": 4, "max_len": 32, "device": "cpu", "dtype": "float32",
                 "mask_label": -100.0}
PLANT_CONTINUED = {"epochs": 1, "encoder_lr": 2e-05, "head_lr": 0.0001, "rule": "the synthetic continued training"}
PLANT_SCORING = {"splits": ["test", "hotels_probe"], "device": "cpu", "workers": 0}


def plant_rules(seeds: Sequence[int] = (0,)) -> dict[str, Any]:
    """The synthetic frozen rules of the planted sets: base seeds `seeds`, tiny recipes, one continued epoch."""
    base_runs = json.loads(json.dumps(fs.RULES["base_runs"]))
    base_runs.update(seeds=list(seeds), laya=dict(PLANT_LAYA), berturk=dict(PLANT_BERTURK))
    return {"base_runs": base_runs, "continued_training": dict(PLANT_CONTINUED)}


def chosen_laya(data_sha256: str) -> dict[str, Any]:
    """A chosen Laya selection run's canonical config (pilot.runs IDENTITY, the PLANT_LAYA recipe, seed 0)."""
    from pilot import runs
    r = PLANT_LAYA
    return runs.canonical({
        "split": "selection_train", "data_sha256": data_sha256, "data_role": "train", "base_model_sha256": "a" * 64,
        "base_config_sha256": "b" * 64, "base_files_sha256": None, "laya_version": "0.4.6", "device": r["device"],
        "workers": r["workers"], "epochs": r["epochs"], "micro_batch": r["micro_batch"], "grad_accum": r["grad_accum"],
        "encoder_lr": r["encoder_lr"], "head_lr": r["head_lr"], "weight_decay": 0.01, "clip_grad_norm": 1.0,
        "loss": "rlcd", "neg": r["neg"], "neg_seed": 0, "freeze_encoder": False, "freeze_embeddings": False,
        "grad_checkpointing": True, "amp": False, "max_len": r["max_len"], "head_max_len": 32, "max_messages": None,
        "shuffle_options": True, "seed": 0, "adamw_foreach": None, "adamw_fused": None, "stop_after_epoch": None,
        "postfit_sha256": None, "mask_tags": None, "fewshot": None})


def chosen_berturk(data_sha256: str) -> dict[str, Any]:
    """A chosen BERTurk selection run's canonical config (pilot.berturk IDENTITY, the PLANT_BERTURK recipe, seed 0)."""
    from pilot import berturk
    r = PLANT_BERTURK
    return berturk.canonical({
        "split": "selection_train", "data_sha256": data_sha256, "data_role": "train", "base_model_sha256": "c" * 64,
        "base_config_sha256": "d" * 64, "base_files_sha256": None, "transformers_version": "5.19.0",
        "accelerate_version": "1.15.0", "device": r["device"], "epochs": r["epochs"], "batch": r["batch"],
        "encoder_lr": r["encoder_lr"], "head_lr": r["head_lr"], "weight_decay": 0.01, "warmup_ratio": 0.1,
        "lr_scheduler": "linear", "max_grad_norm": 1.0, "optim": "adamw_torch", "loss": "masked_bce",
        "max_len": r["max_len"], "max_messages": None, "mask_tags": None, "seed": 0, "stop_after_epoch": None,
        "postfit_sha256": None, "fewshot": None})


def bench(tmp_path: Path, synthetic_raw: SyntheticRaw, monkeypatch: pytest.MonkeyPatch, mode: str = "primary",
          seeds: Sequence[int] = (0,), scoring: dict[str, Any] | None = None) -> types.SimpleNamespace:
    """synthetic_protocol with PLANT_RULES, both kinds' chosen canonicals bound and the scoring block (cpu, 0)."""
    _, pre_manifest = prepared(tmp_path, synthetic_raw, name="pre")
    data_sha = _rows(pre_manifest)[0]["sha256"]              # prepare() is deterministic: the source's sha256
    systems = postfit_systems(tmp_path, {"laya": chosen_laya(data_sha), "berturk": chosen_berturk(data_sha)})
    return synthetic_protocol(tmp_path, synthetic_raw, monkeypatch, rules=plant_rules(seeds), systems=systems,
                              scoring=dict(PLANT_SCORING if scoring is None else scoring), mode=mode)


def _epoch_rows(kind: str, n: int) -> list[dict[str, Any]]:
    """n made-up epoch rows of a finished run record of `kind` (pilot.runs or pilot.berturk EPOCH_FIELDS)."""
    from pilot import berturk, runs
    rows = []
    for e in range(1, n + 1):
        if kind == "laya":
            rows.append(dict(zip(runs.EPOCH_FIELDS, (e, 1.0 / e, 2.0, 0.5, 3, 10, 1, None, 1, None, 0, 0, False))))
        else:
            rows.append(dict(zip(berturk.EPOCH_FIELDS, (e, 1.0 / e, 2.0, 3, 10, 1, None, 1, 0))))
    return rows


def plant_laya(path: str, canonical: dict[str, Any], record: dict[str, Any], seed: int = 0) -> str:
    """A tiny Laya checkpoint holding a finished ray_train.py run: rl_agent_config.json training.ray_train {run_id,
    canonical, record (run_id and canonical filled in)}, training.epoch the run's last epoch, and final.json."""
    from pilot import runs
    from tiny_laya import tiny_checkpoint
    ckpt = tiny_checkpoint(path, seed=seed)
    rid = runs.run_id(canonical)
    cfg_path = os.path.join(ckpt, "rl_agent_config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["fine_tuned"] = True
    cfg["training"] = dict(cfg.get("training") or {}, epoch=canonical.get("stop_after_epoch") or canonical["epochs"],
                           calibrated=False, ray_train={"run_id": rid, "canonical": canonical,
                                                        "record": dict(record, run_id=rid, canonical=canonical)})
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False)
    with open(os.path.join(ckpt, "final.json"), "w", encoding="utf-8") as f:
        json.dump({"kind": "final", "run_id": rid}, f)
    return ckpt


def plant_berturk(path: str, canonical: dict[str, Any], record: dict[str, Any], seed: int = 0) -> str:
    """The tiny BERTurk base files plus a finished run's berturk_train.json (schema berturk-train/1)."""
    from pilot import berturk
    from tiny_berturk import tiny_base
    ckpt = tiny_base(path, seed=seed)
    stop = canonical.get("stop_after_epoch") or canonical["epochs"]
    doc = {"schema": berturk.RECORD_SCHEMA, "run_id": berturk.run_id(canonical), "canonical": canonical,
           "epoch": stop, "epochs_schedule": canonical["epochs"], "stop_after_epoch": canonical.get("stop_after_epoch"),
           "calibrated": False, "temperature": 1.0, "postfit": None, "record": record, "provenance": {}}
    with open(os.path.join(ckpt, berturk.RECORD), "wb") as f:
        f.write(berturk.record_bytes(doc))
    return ckpt


def _base_canonical(ns: types.SimpleNamespace, kind: str) -> dict[str, Any]:
    """The canonical config of a masked base of `kind` bound to the synthetic protocol (seed left to the caller)."""
    canon = dict(fs.chosen_canonical(ns.protocol, kind), mask_tags=sorted(ns.block["tags"]),
                 fewshot=fs.binding(ns.pin, "base"))
    canon["stop_after_epoch"] = ns.block["rules"]["base_runs"][kind]["stop_after_epoch"]
    return canon


def _continued_canonical(ns: types.SimpleNamespace, kind: str, base_dir: str, base_canon: dict[str, Any], base_rid: str,
                         draw: int, k: int) -> dict[str, Any]:
    """The canonical config of a continued run of (draw, k) from the planted base at `base_dir`."""
    from pilot import berturk, dumps
    from pilot.weights import identity
    ident = berturk.identity(base_dir) if kind == "berturk" else identity(base_dir)
    d = ns.block["draws"][draw]
    canon = {key: v for key, v in base_canon.items() if key not in ("stop_after_epoch", "mask_tags")}
    canon.update(base_model_sha256=ident["model_sha256"], base_config_sha256=ident["config_sha256"],
                 base_files_sha256=dumps.files_digest(ident["files_sha256"]),
                 epochs=ns.block["rules"]["continued_training"]["epochs"], seed=d["seed"],
                 fewshot=fs.binding(ns.pin, "continued", draw, k, d["digest"], base_rid))
    if kind == "laya":
        canon["neg_seed"] = d["seed"]
    return canon


def plant_fewshot_set(ns: types.SimpleNamespace, root: str,
                      kinds: Sequence[str] = ("laya", "berturk")) -> dict[str, str]:
    """One consistent planted set per kind under `root` for the synthetic protocol `ns` (see the module docstring);
    returns {name: checkpoint dir}."""
    from pilot import berturk, runs
    tags, out = sorted(ns.block["tags"]), {}
    mode = fs.read_mode(ns.pin)
    seeds = fs.allowed_base_seeds(ns.block, mode)
    for kind in kinds:
        bases = {}
        for s in seeds:
            canon = dict(_base_canonical(ns, kind), seed=s)
            if kind == "laya":
                canon["neg_seed"] = s
                rec = {"rung": "none", "fallback": False, "weaker": False, "load_s": 1.0, "final_save_s": 0.5,
                       "calib_s": None, "epochs": _epoch_rows(kind, canon["stop_after_epoch"]),
                       "fewshot": {"protocol_sha256": ns.pin, "stage": "base", "tags": tags,
                                   "held_out_items": {t: 0 for t in tags}, "items_per_question": {}}}
                path, rid = plant_laya(os.path.join(root, "laya-base-s%d" % s), canon, rec, seed=s), runs.run_id(canon)
            else:
                rec = {"load_s": 1.0, "epochs": _epoch_rows(kind, canon["stop_after_epoch"]),
                       "fewshot": {"protocol_sha256": ns.pin, "stage": "base", "tags": tags,
                                   "unmasked_held_out_labels": {t: 0 for t in tags}}}
                path = plant_berturk(os.path.join(root, "berturk-base-s%d" % s), canon, rec, seed=s)
                rid = berturk.run_id(canon)
            out["%s-base-s%d" % (kind, s)] = path
            bases[s] = (path, canon, rid)
        for draw in range(len(ns.block["draws"])):
            for k in ns.block["k_values"]:
                base_dir, base_canon, base_rid = bases[fs.base_seed_of(ns.block, draw, mode)]
                canon = _continued_canonical(ns, kind, base_dir, base_canon, base_rid, draw, k)
                sel = fs.selection(ns.block, draw, k)
                n_sup, n_rep = len(sel["support"]), len(sel["replay"])
                proof = {"protocol_sha256": ns.pin, "stage": "continued", "draw": draw, "k": k,
                         "digest": ns.block["draws"][draw]["digest"], "tags": tags, "support_rows": n_sup,
                         "replay_rows": n_rep, "base": {"run_id": base_rid, "canonical": base_canon}}
                name = "%s-d%d-k%d" % (kind, draw, k)
                epochs = _epoch_rows(kind, canon["epochs"])
                if kind == "laya":
                    proof.update(new_tag_items={t: n_sup for t in tags}, items_per_question={})
                    rec = {"rung": "none", "fallback": False, "weaker": False, "load_s": 1.0, "final_save_s": 0.5,
                           "calib_s": None, "epochs": epochs, "fewshot": proof}
                    out[name] = plant_laya(os.path.join(root, name), canon, rec, seed=10 + 10 * draw + k)
                else:
                    proof.update(new_tag_labels={t: n_sup for t in tags}, replay_unmasked_held_out_labels=0)
                    rec = {"load_s": 1.0, "epochs": epochs, "fewshot": proof}
                    out[name] = plant_berturk(os.path.join(root, name), canon, rec, seed=10 + 10 * draw + k)
    return out


def decision_record(pin: str, mode: str = "primary", projection_fired: bool | None = None,
                    ladder_failure: bool = False) -> dict[str, Any]:
    """A minimal compute decision record (schema fewshot-projection/1) whose trigger fields agree with `mode` unless
    projection_fired is given."""
    fired = (mode == "fallback") if projection_fired is None else projection_fired
    return {"schema": DECISION_SCHEMA, "protocol_sha256": pin, "decision": {"mode": mode},
            "trigger": {"projection_fired": fired, "ladder_failure": ladder_failure}}


def write_decision(path: str, record: dict[str, Any]) -> str:
    """Write the decision record as JSON to `path` (creating its dir); returns `path`."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return path


def fewshot_block(manifest: str, tags: Sequence[str] = ("card",), k_values: Sequence[int] = (1, 2),
                  draw_seeds: Sequence[int] = (0,), neg_per_pos: int = 1, replay: int = 2,
                  rules: dict[str, Any] | None = None) -> dict[str, Any]:
    """The protocol's few-shot block over the synthetic snapshot's selection_train at `manifest`: the draws (base seed
    and digest included) and a deep copy of pilot.fewshot.RULES with the top-level keys of `rules` replaced."""
    entry, rows = _rows(manifest)
    draws = []
    for i, s in enumerate(draw_seeds):
        d = fs.draw(rows, s, tags, k_values, neg_per_pos, replay)
        draws.append({"draw": i, "seed": s, "base_seed": s, "support": d["support"], "replay": d["replay"],
                      "positives": d["positives"], "digest": fs.draw_digest(d)})
    block_rules = copy.deepcopy(fs.RULES)
    for key, value in (rules or {}).items():
        block_rules[key] = copy.deepcopy(value)
    return {"tags": list(tags), "k_values": list(k_values), "draw_seeds": list(draw_seeds),
            "neg_per_pos": neg_per_pos, "replay": replay, "epochs": fs.EPOCHS,
            "source": {"split": "selection_train", "sha256": entry["sha256"], "uids_sha256": entry["uids_sha256"]},
            "draws": draws, "rules": json.loads(json.dumps(block_rules))}


def synthetic_protocol(tmp_path: Path, synthetic_raw: SyntheticRaw, monkeypatch: pytest.MonkeyPatch,
                       tags: Sequence[str] = ("card",), k_values: Sequence[int] = (1, 2),
                       draw_seeds: Sequence[int] = (0,), neg_per_pos: int = 1, replay: int = 2,
                       rules: dict[str, Any] | None = None, systems: dict[str, Any] | None = None,
                       scoring: dict[str, Any] | None = None, mode: str = "primary",
                       splits: dict[str, Any] | None = None,
                       extra: dict[str, Any] | None = None) -> types.SimpleNamespace:
    """The synthetic frozen protocol, written and pinned with its decision record (see the module docstring); returns
    the namespace (manifest, out_dir, protocol_path, pin, decision_path, block, rows, protocol)."""
    out_dir, manifest = prepared(tmp_path, synthetic_raw)
    monkeypatch.setattr(pilot.data, "DEFAULT_MANIFEST", manifest)   # first: splits_block reads the committed one
    _, rows = _rows(manifest)
    block = fewshot_block(manifest, tags, k_values, draw_seeds, neg_per_pos, replay, rules)
    protocol = {k: {} for k in pr.REQUIRED_KEYS}
    protocol.update(schema=pr.SCHEMA, fewshot=block)
    if systems is not None:
        protocol["systems"] = systems
    if scoring is not None:
        protocol["scoring"] = scoring
    for key, value in (extra or {}).items():
        protocol[key] = copy.deepcopy(value)
    protocol["splits"] = pr.splits_block(manifest) if splits is None else splits
    path = str(tmp_path / "protocol.json")
    pin = pr.write_protocol(protocol, path)
    decision_path = write_decision(str(tmp_path / "fewshot" / "projection.json"), decision_record(pin, mode))
    monkeypatch.setattr(pr, "PROTOCOL_PATH", path)
    monkeypatch.setattr(pr, "PROTOCOL_SHA256", pin)
    monkeypatch.setattr(fs, "DECISION_PATH", decision_path, raising=False)
    return types.SimpleNamespace(manifest=manifest, out_dir=out_dir, protocol_path=path, pin=pin,
                                 decision_path=decision_path, block=json.loads(json.dumps(block)), rows=rows,
                                 protocol=json.loads(json.dumps(protocol)))


def plant_runs(ns: types.SimpleNamespace, path: str) -> str:
    """A complete runs manifest (pilot.fewshot RUNS_SCHEMA) of the synthetic protocol `ns` under its recorded mode,
    built from pilot.fewshot.slots without any checkpoint, written to `path` (key-sorted JSON); returns `path`."""
    from pilot.data import repo_relpath, sha256_bytes
    mode = fs.read_mode(ns.pin)
    tags = sorted(ns.block["tags"])
    entries, wall = [], 100.0
    for kind, stage, seed, draw, k in fs.slots(ns.block, mode):
        name = "%s-base-s%d" % (kind, seed) if stage == "base" else "%s-d%d-k%d" % (kind, draw, k)
        e = {"kind": kind, "stage": stage, "seed": seed, "draw": draw, "k": k, "digest": None, "base_seed": None,
             "base_run_id": None, "train_run_id": "%s-train-%s" % (kind, name.split("-", 1)[1]),
             "dump_run_id": "%s-dump-%s" % (kind, name.split("-", 1)[1]), "checkpoint_name": "ckpt-" + name,
             "rung": "none" if kind == "laya" else None, "fallback": False if kind == "laya" else None,
             "weaker": False if kind == "laya" else None, "wall_clock_s": wall, "loss_by_epoch": [1.0, 0.5]}
        wall += 1.5
        if stage == "base":
            key = "held_out_items" if kind == "laya" else "unmasked_held_out_labels"
            e["proof"] = {key: {t: 0 for t in tags}}
        else:
            base_seed = fs.base_seed_of(ns.block, draw, mode)
            sel = fs.selection(ns.block, draw, k)
            n_sup, n_rep = len(sel["support"]), len(sel["replay"])
            e.update(digest=ns.block["draws"][draw]["digest"], base_seed=base_seed,
                     base_run_id="%s-train-base-s%d" % (kind, base_seed))
            key = "new_tag_items" if kind == "laya" else "new_tag_labels"
            e["proof"] = {key: {t: n_sup for t in tags}, "support_rows": n_sup, "replay_rows": n_rep}
            if kind == "berturk":
                e["proof"]["replay_unmasked_held_out_labels"] = 0
        entries.append(e)
    entries.sort(key=fs.entry_key)
    with open(fs.DECISION_PATH, "rb") as f:
        decision = {"path": repo_relpath(fs.DECISION_PATH), "sha256": sha256_bytes(f.read())}
    record = {"schema": fs.RUNS_SCHEMA, "protocol_sha256": ns.pin, "mode": mode, "decision": decision, "tags": tags,
              "wall_rule": fs.WALL_RULE, "entries": entries}
    fs.check_runs(record, ns.protocol, ns.pin, mode)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write((json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return path


def split_gold(manifest: str, split: str) -> tuple[dict[str, Any], list[str], Any]:
    """(manifest entry, uids in file order, bool [n, 48] gold) of a verified split of the synthetic snapshot."""
    from pilot import dumps
    with open(manifest, encoding="utf-8") as f:
        entry = json.load(f)["splits"][split]
    with open(os.path.join(os.path.dirname(manifest), entry["path"]), "rb") as f:
        uids = [r["uid"] for r in pilot.data.parse_jsonl(f.read())]
    return entry, uids, dumps.read_gold(entry, manifest, uids)


def plant_fewshot_dumps(ns: types.SimpleNamespace, runs: dict[str, Any], runs_sha256: str, dump_root: str,
                        seed: int = 0, logits: Callable[..., Any] | None = None) -> dict[tuple[str, str], str]:
    """Every dump the few-shot record reads, for the entries of `runs` (see the module docstring); returns {(dump run
    id, split): dump dir}."""
    import zlib

    import numpy as np

    import score
    from pilot import dumps
    out = {}
    for e in runs["entries"]:
        for split in (("val", "test") if e["stage"] == "base" else ("test",)):
            entry, uids, Y = split_gold(ns.manifest, split)
            side = score.sidecar_base(entry, split, ns.manifest, None)
            side.update(run_id=e["dump_run_id"],
                        system={"kind": e["kind"], "checkpoint": "<outside-repo>/" + e["checkpoint_name"],
                                "checkpoint_identity": {"local": True, "model_sha256": "%064x" % zlib.crc32(
                                    e["dump_run_id"].encode("utf-8"))}},
                        temperature={"shipped": 1.0, "source": "planted"},
                        scorer={"device": "cpu", "dtype": "float32", "amp_enabled": False}, seed=e["seed"],
                        training={"run_id": e["train_run_id"], "split": "selection_train", "calibrated": False})
            if split == "test":
                side.update(protocol_sha256=ns.pin, fewshot_runs_sha256=runs_sha256)
            rng = np.random.default_rng([seed, zlib.crc32(("%s/%s" % (e["dump_run_id"], split)).encode("utf-8"))])
            L = logits(e, split, Y, rng) if logits is not None else rng.normal(0.0, 3.0, size=Y.shape)
            run_dir = os.path.join(dump_root, e["dump_run_id"])
            dumps.write_dump(run_dir, split, np.asarray(L, dtype=np.float32), uids, side)
            out[(e["dump_run_id"], split)] = run_dir
    return out
