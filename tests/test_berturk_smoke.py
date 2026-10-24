"""The BERTurk tracer on CPU, offline: tiny BERT base -> berturk_train.py (HF Trainer, masked BCE, 2 epochs) -> two
per-epoch exports -> score.py --model berturk val dumps that pilot.dumps.load_dump verifies -> pilot.postfit.candidate
and fit accept them -> report.py run_report reports them; score.py's protocol gate refuses an export on the synthetic
test split.

test_berturk_final_seed_chain continues the same chain to the final manifest: postfit record -> the BERTurk reference
check (with a resolution: the tiny model is far below 0.707) -> the printed final commands -> 3 bound final seeds
serving the val-fitted T -> a final seed's own val dump serves T and is refused as in-sample by postfit and by report.py
(run_report and either side of --compare) -> postfit.py --final-manifest.

No marker: it runs in `make test` (no Ray, no weights, no data download; seconds).
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

import pilot.data
import report
import score
from conftest import prepared
from intents import INTENTS
from pilot import berturk, dumps, postfit
from pilot.data import sha256_file
from tiny_berturk import tiny_base

REPO = pilot.data.REPO_ROOT
TIMEOUT = 600
SyntheticRaw = Callable[..., tuple[str, dict[str, str]]]   # conftest's synthetic_raw fixture


def _run(*args: Any) -> subprocess.CompletedProcess[str]:
    """Run a repo script with the test interpreter (no PYTHONPATH, so it imports like a user's command)."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable] + [str(x) for x in args], cwd=REPO, env=env, capture_output=True, text=True,
                       timeout=TIMEOUT)
    assert r.returncode == 0, "%s failed (exit %d):\n%s" % (args[0], r.returncode, r.stderr[-4000:])
    return r


def _bytes(path: str) -> bytes:
    """The bytes of a file."""
    with open(path, "rb") as f:
        return f.read()


@pytest.mark.timeout(TIMEOUT)
def test_berturk_train_score_postfit_chain(tmp_path: Path, synthetic_raw: SyntheticRaw) -> None:
    import torch
    out_dir, manifest = prepared(tmp_path, synthetic_raw)
    splits = os.path.join(out_dir, "splits")
    base = tiny_base(tmp_path / "base")
    storage = tmp_path / "runs"
    argv = ["berturk_train.py", "--data", os.path.join(splits, "selection_train.jsonl"), "--manifest", manifest,
            "--base", base, "--device", "cpu", "--epochs", "2", "--batch", "4", "--max-len", "32", "--seed", "0",
            "--storage", storage]
    rid = _run(*(argv + ["--print-run-id"])).stdout.strip()
    assert re.match(r"^berturk-selection_train-s0-[0-9a-f]{8}$", rid), rid
    assert not storage.exists()                              # --print-run-id creates nothing
    _run(*argv)
    run_dir = storage / rid
    assert sorted(os.listdir(run_dir / "exports")) == ["epoch-001", "epoch-002"]
    assert (run_dir / berturk.DONE).is_file()
    exports = [str(run_dir / "exports" / e) for e in ("epoch-001", "epoch-002")]
    for k, exp in enumerate(exports, 1):
        assert sorted(os.listdir(exp)) == sorted(berturk.CHECKPOINT_FILES)
        for f in berturk.TOKENIZER_FILES:                    # byte copies of the base, never save_pretrained output
            assert _bytes(os.path.join(exp, f)) == _bytes(os.path.join(base, f))
        with open(os.path.join(exp, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        assert [cfg["id2label"][str(i)] for i in range(len(INTENTS))] == list(INTENTS)
        with open(os.path.join(exp, berturk.RECORD), encoding="utf-8") as f:
            rec = json.load(f)
        assert rec["schema"] == berturk.RECORD_SCHEMA and rec["run_id"] == rid and rec["epoch"] == k
        assert rec["calibrated"] is False and rec["temperature"] == 1.0
        assert [row["epoch"] for row in rec["record"]["epochs"]] == list(range(1, k + 1))
        assert all(set(row) == set(berturk.EPOCH_FIELDS) for row in rec["record"]["epochs"])
        assert rec["record"]["param_device"] == "cpu" and rec["record"]["dtype"] == "torch.float32"
        # head LR 1e-4: the first reference-check fix
        assert rec["canonical"]["encoder_lr"] == 2e-05 and rec["canonical"]["head_lr"] == 0.0001
        groups = rec["record"].get("optimizer_groups") or []      # the fresh head at Laya's head LR, own groups
        assert [(g["group"], g["decay"], g["lr"]) for g in groups] == [
            ("encoder", True, 2e-05), ("encoder", False, 2e-05), ("head", True, 0.0001), ("head", False, 0.0001)]
        assert [g["weight_decay"] for g in groups] == [0.01, 0.0, 0.01, 0.0] and groups[2]["n_tensors"] == 1

    dump_root = str(tmp_path / "dumps")
    val = os.path.join(splits, "val.jsonl")
    threads = torch.get_num_threads()
    try:
        run_dirs = []
        for k, exp in enumerate(exports, 1):
            d, _ = score.main([val, "--model", "berturk", "--checkpoint", exp, "--workers", "0", "--manifest",
                               manifest, "--out-root", dump_root])
            assert os.path.basename(d) == berturk.dump_run_id(berturk.identity(exp))
            logits, uids, side, entry = dumps.load_dump(d, "val", manifest)
            assert logits.shape == (entry["n_messages"], len(INTENTS))
            assert side["system"]["kind"] == "berturk" and side["temperature"]["shipped"] == 1.0
            assert (side["scorer"]["device"], side["scorer"]["dtype"], side["scorer"]["amp_enabled"]) \
                == ("cpu", "torch.float32", False)
            tr = side["training"]
            assert side["seed"] == 0 and tr["epoch"] == k and tr["epochs_schedule"] == 2
            assert tr["encoder_lr"] == 2e-05 and tr["head_lr"] == 0.0001 and tr["calibrated"] is False
            assert side["logit"] == berturk.LOGIT_NOTE
            assert postfit.candidate(d, manifest)["system_kind"] == "berturk"
            run_dirs.append(d)
        rec = postfit.fit(run_dirs, manifest, lr_grid=(2e-05,), epochs=2, head_lr=berturk.HEAD_LR, selection_seed=0,
                          final_seeds=(0, 1, 2))
        assert rec["chosen"]["epoch"] in (1, 2) and postfit.T_MIN <= rec["chosen"]["temperature"] <= postfit.T_MAX
        rep = report.run_report(run_dirs[0], "val", manifest)
        assert sorted(rep["variants"]) == ["raw", "shipped"]
        with pytest.raises(SystemExit, match="protocol gate"):
            score.main([os.path.join(splits, "test.jsonl"), "--model", "berturk", "--checkpoint", exports[0],
                        "--workers", "0", "--manifest", manifest, "--out-root", dump_root])
        assert not any(os.path.exists(os.path.join(d, "test.npz")) or os.path.exists(os.path.join(d, "test.json"))
                       for d in run_dirs)
    finally:
        torch.set_num_threads(threads)


@pytest.mark.timeout(TIMEOUT)
def test_berturk_final_seed_chain(tmp_path: Path, synthetic_raw: SyntheticRaw,
                                  capsys: pytest.CaptureFixture[str]) -> None:
    import postfit as postfit_cli
    import torch
    out_dir, manifest = prepared(tmp_path, synthetic_raw)
    splits = os.path.join(out_dir, "splits")
    base = tiny_base(tmp_path / "base")
    storage = tmp_path / "runs"
    argv = ["berturk_train.py", "--data", os.path.join(splits, "selection_train.jsonl"), "--manifest", manifest,
            "--base", base, "--device", "cpu", "--epochs", "2", "--batch", "4", "--max-len", "32", "--seed", "0",
            "--storage", storage]
    rid = _run(*(argv + ["--print-run-id"])).stdout.strip()
    _run(*argv)                                              # the tiny selection run
    dump_root = str(tmp_path / "dumps")
    val = os.path.join(splits, "val.jsonl")
    threads = torch.get_num_threads()
    try:
        run_dirs = [score.main([val, "--model", "berturk", "--checkpoint", str(storage / rid / "exports" / e),
                                "--workers", "0", "--manifest", manifest, "--out-root", dump_root])[0]
                    for e in ("epoch-001", "epoch-002")]
        assert [dumps.load_dump(d, "val", manifest)[2]["training"]["head_lr"] for d in run_dirs] == [0.0001, 0.0001]
        record = postfit.fit(run_dirs, manifest, lr_grid=(2e-05,), epochs=2, head_lr=berturk.HEAD_LR,
                             selection_seed=0, final_seeds=(0, 1, 2))
        path = postfit.write_record(record, str(tmp_path / "postfit" / "berturk.json"))
        sha, chosen = sha256_file(path), record["chosen"]
        T = chosen["temperature"]
        assert record["system_kind"] == "berturk" and chosen["train_run_id"] == rid
        assert chosen["stop_after_epoch"] == (1 if chosen["epoch"] == 1 else None)

        ref = str(tmp_path / "postfit" / "berturk-reference.json")
        check = postfit_cli.main(["--reference-check", path, "--resolution", "tiny offline test model", "--out", ref])
        assert check["status"] == "investigate" and check["postfit"]["sha256"] == sha
        assert berturk.check_reference_for_final(ref, sha, record)["resolution"] == "tiny offline test model"

        prefix, final_storage = str(tmp_path / "ckpt" / "berturk-final"), str(tmp_path / "final_runs")
        final_train = os.path.join(splits, "final_train.jsonl")
        for seed in (0, 1, 2):
            cmd = berturk.final_command(record, seed, path, ref, final_storage, prefix, base=base, data=final_train,
                                        manifest=manifest)
            assert cmd[0] == "berturk_train.py" and ("--stop-after-epoch" in cmd) == (chosen["epoch"] == 1)
            assert cmd[cmd.index("--head-lr") + 1] == "0.0001"            # the head LR, from the chosen canonical
            _run(*cmd)
            ckpt = "%s-s%d" % (prefix, seed)
            assert sorted(os.listdir(ckpt)) == sorted(berturk.CHECKPOINT_FILES)
            with open(os.path.join(ckpt, berturk.RECORD), encoding="utf-8") as f:
                rec = json.load(f)
            assert json.dumps(rec["temperature"]) == json.dumps(T) and rec["temperature"] == T   # bit for bit
            assert rec["calibrated"] is True and rec["epoch"] == chosen["epoch"]
            assert rec["postfit"] == postfit.final_block(record, sha, path) and rec["postfit"]["sha256"] == sha
            canon = rec["canonical"]
            assert canon["split"] == "final_train" and canon["seed"] == seed and canon["postfit_sha256"] == sha
            assert canon.get("stop_after_epoch") == chosen["stop_after_epoch"]
            assert canon["head_lr"] == berturk.HEAD_LR and canon["encoder_lr"] == 2e-05
            assert [row["epoch"] for row in rec["record"]["epochs"]] == list(range(1, chosen["epoch"] + 1))

        final_dump, _ = score.main([val, "--model", "berturk", "--checkpoint", prefix + "-s0", "--workers", "0",
                                    "--manifest", manifest, "--out-root", dump_root])
        _, _, side, _ = dumps.load_dump(final_dump, "val", manifest)
        assert side["temperature"]["shipped"] == T and side["seed"] == 0
        assert side["training"]["postfit_sha256"] == sha and side["training"]["calibrated"] is True
        with pytest.raises(SystemExit, match="selection_train"):                 # a final seed's val is in-sample
            postfit.candidate(final_dump, manifest)
        # report.py never selects θ on a BERTurk final seed's val dump, alone or on either side of
        # --compare (its training block names final_train and carries the postfit sha256)
        with pytest.raises(SystemExit, match="trained on final_train \\(postfit record %s\\)" % sha):
            report.run_report(final_dump, "val", manifest)
        for a_dirs, b_dirs in (([final_dump], run_dirs[:1]), (run_dirs[:1], [final_dump])):
            with pytest.raises(SystemExit, match="in-sample"):
                report.compare(a_dirs, b_dirs, "val", manifest, B=10)

        out = str(tmp_path / "postfit" / "berturk-final.json")
        man = postfit_cli.main(["--final-manifest", path] + ["%s-s%d" % (prefix, s) for s in (2, 0, 1)]
                               + ["--out", out])
        assert man["schema"] == postfit.FINAL_SCHEMA and man["system_kind"] == "berturk"
        assert [e["seed"] for e in man["seeds"]] == [0, 1, 2] and man["postfit"]["sha256"] == sha
        assert len({e["temperature"] for e in man["seeds"]}) == 1 and man["chosen"]["temperature"] == T
        assert len(man["config_sha256_sans_seed"]) == 64
        with open(out, encoding="utf-8") as f:
            assert json.load(f) == man
        # a record on the committed manifest binds only final seeds trained on it (these seeds were
        # trained on the synthetic manifest, so the same seeds are refused for a copy of the record marked committed)
        assert record["manifest"]["is_committed"] is False
        marked = postfit.write_record(dict(record, manifest=dict(record["manifest"], is_committed=True)),
                                      str(tmp_path / "postfit" / "marked.json"))
        with pytest.raises(SystemExit, match="provenance manifest is_committed False"):
            berturk.final_manifest(marked, ["%s-s%d" % (prefix, s) for s in (0, 1, 2)])

        capsys.readouterr()
        lines = postfit_cli.main(["--print-final-commands", path, "--reference", ref, "--storage", final_storage,
                                  "--out-prefix", prefix, "--base", base, "--manifest", manifest])
        printed = capsys.readouterr().out.splitlines()
        assert printed == lines and len(lines) == 3 and all(l.startswith("berturk_train.py ") for l in lines)
        assert all(" --head-lr 0.0001 " in l for l in lines)
    finally:
        torch.set_num_threads(threads)
