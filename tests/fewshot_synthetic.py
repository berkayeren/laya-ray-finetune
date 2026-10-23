"""The synthetic frozen protocol every Phase 9 test installs (a helper module, no test functions).

synthetic_protocol(tmp_path, synthetic_raw, monkeypatch, ...) runs pilot.data.prepare on the synthetic Multi3NLU++
snapshot (conftest), draws the few-shot support sets on its verified selection_train with pilot.fewshot.draw (tags
card, k 1 and 2, draw seed 0, 1 negative per positive, replay 2 by default), writes a REQUIRED_KEYS-complete protocol/1
file whose fewshot block carries the settings, the source, the draws (base seed and digest included) and the rules (a
deep copy of pilot.fewshot.RULES, base_runs and continued_training replaced by `rules` when given), writes a minimal
D-08 decision record (schema fewshot-projection/1, this protocol's sha256, decision.mode and trigger fields that agree
with it), and pins both by monkeypatch: pilot.protocol.PROTOCOL_PATH and PROTOCOL_SHA256, pilot.data.DEFAULT_MANIFEST
and pilot.fewshot.DECISION_PATH. It returns a namespace (manifest, out_dir, protocol_path, pin, decision_path, block,
rows, protocol).

`splits` (optional) replaces the protocol's splits block, e.g. {"val": {"n_messages": 5}, "test": {"n_messages": 7}}
for fewshot.py --projection (09-04).

postfit_systems(tmp_path, canonicals) writes one minimal postfit record per system kind ({"chosen": {"canonical":
...}}) and returns the protocol "systems" block that binds each by absolute path and sha256.
"""
import copy
import json
import os
import types

import pilot.data
from conftest import prepared
from pilot import fewshot as fs
from pilot import protocol as pr

DECISION_SCHEMA = "fewshot-projection/1"     # pilot.fewshot.DECISION_SCHEMA; tests/test_fewshot_binding.py checks it


def _rows(manifest):
    with open(manifest, encoding="utf-8") as f:
        entry = json.load(f)["splits"]["selection_train"]
    with open(os.path.join(os.path.dirname(manifest), entry["path"]), "rb") as f:
        return entry, pilot.data.parse_jsonl(f.read())


def postfit_systems(tmp_path, canonicals):
    """{kind: {"kind", "postfit": {"path": absolute path, "sha256"}}} over records written under tmp_path/postfit."""
    out = {}
    os.makedirs(str(tmp_path / "postfit"), exist_ok=True)
    for kind, canon in canonicals.items():
        path = str(tmp_path / "postfit" / ("%s.json" % kind))
        raw = (json.dumps({"schema": "postfit/1", "chosen": {"canonical": canon}}, ensure_ascii=False, indent=2)
               + "\n").encode("utf-8")
        with open(path, "wb") as f:
            f.write(raw)
        out[kind] = {"kind": kind, "postfit": {"path": path, "sha256": pilot.data.sha256_bytes(raw)}}
    return out


def decision_record(pin, mode="primary", projection_fired=None, ladder_failure=False):
    """A minimal D-08 decision record (schema fewshot-projection/1) whose trigger fields agree with `mode` unless
    projection_fired is given."""
    fired = (mode == "fallback") if projection_fired is None else projection_fired
    return {"schema": DECISION_SCHEMA, "protocol_sha256": pin, "decision": {"mode": mode},
            "trigger": {"projection_fired": fired, "ladder_failure": ladder_failure}}


def write_decision(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return path


def synthetic_protocol(tmp_path, synthetic_raw, monkeypatch, tags=("card",), k_values=(1, 2), draw_seeds=(0,),
                       neg_per_pos=1, replay=2, rules=None, systems=None, scoring=None, mode="primary", splits=None):
    out_dir, manifest = prepared(tmp_path, synthetic_raw)
    entry, rows = _rows(manifest)
    draws = []
    for i, s in enumerate(draw_seeds):
        d = fs.draw(rows, s, tags, k_values, neg_per_pos, replay)
        draws.append({"draw": i, "seed": s, "base_seed": s, "support": d["support"], "replay": d["replay"],
                      "positives": d["positives"], "digest": fs.draw_digest(d)})
    block_rules = copy.deepcopy(fs.RULES)
    for key, value in (rules or {}).items():
        block_rules[key] = copy.deepcopy(value)
    block = {"tags": list(tags), "k_values": list(k_values), "draw_seeds": list(draw_seeds),
             "neg_per_pos": neg_per_pos, "replay": replay, "epochs": fs.EPOCHS,
             "source": {"split": "selection_train", "sha256": entry["sha256"], "uids_sha256": entry["uids_sha256"]},
             "draws": draws, "rules": json.loads(json.dumps(block_rules))}
    protocol = {k: {} for k in pr.REQUIRED_KEYS}
    protocol.update(schema=pr.SCHEMA, fewshot=block)
    if systems is not None:
        protocol["systems"] = systems
    if scoring is not None:
        protocol["scoring"] = scoring
    if splits is not None:
        protocol["splits"] = splits
    path = str(tmp_path / "protocol.json")
    pin = pr.write_protocol(protocol, path)
    decision_path = write_decision(str(tmp_path / "fewshot" / "projection.json"), decision_record(pin, mode))
    monkeypatch.setattr(pr, "PROTOCOL_PATH", path)
    monkeypatch.setattr(pr, "PROTOCOL_SHA256", pin)
    monkeypatch.setattr(pilot.data, "DEFAULT_MANIFEST", manifest)
    monkeypatch.setattr(fs, "DECISION_PATH", decision_path, raising=False)
    return types.SimpleNamespace(manifest=manifest, out_dir=out_dir, protocol_path=path, pin=pin,
                                 decision_path=decision_path, block=json.loads(json.dumps(block)), rows=rows,
                                 protocol=json.loads(json.dumps(protocol)))
