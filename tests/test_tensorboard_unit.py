"""pilot/tensorboard.py: the scalar builders (tags and units), the tag-checked write, the lazy writer and the
--tensorboard path guard, offline.

The writer round trip goes through tensorboardX and is read back with tests/tb_events.py (crc-checked, no tensorboard
package); the builders and write are checked on a fake writer that records its add_scalar calls. The path guard is
checked against the repo's own frozen and tracked paths through symlinks, case variants and file locations, and end to
end through ray_train.py --print-run-id with a DIR/RUN_ID symlink into results/ (no Ray start, nothing created).
"""
import os
from pathlib import Path
from typing import Any

import pytest

import pilot.data
import tb_events
from conftest import make_raw
from pilot import tensorboard as ptb
from pilot.data import REPO_ROOT
from tiny_laya import tiny_checkpoint

pytestmark = pytest.mark.unit


class FakeWriter:
    """Records every add_scalar(tag, value, step) call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, float, int]] = []

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.calls.append((tag, value, step))


def test_memory_scalars_convert_bytes_and_keep_none() -> None:
    m = ptb.memory_scalars(3 * ptb.GIB, None, ptb.GIB, 0)
    assert m == {"memory/device_current_gib": 3.0, "memory/mps_driver_gib": None, "memory/rss_gib": 1.0,
                 "memory/swap_used_gib": 0.0}


def test_step_scalars_rate_and_tags() -> None:
    mem = ptb.memory_scalars(None, None, ptb.GIB, 0)
    s = ptb.step_scalars(0.5, None, 1e-4, 2.0, None, "train/items_per_s", 12, 2.0, mem)
    assert s["train/items_per_s"] == 6.0 and "train/messages_per_s" not in s
    assert s["train/loss"] == 0.5 and s["train/lr_head"] == 1e-4 and s["train/grad_norm"] == 2.0
    assert s["train/lr_encoder"] is None and s["train/sigma"] is None
    assert s["memory/rss_gib"] == 1.0 and s["memory/device_current_gib"] is None
    assert set(s) <= set(ptb.TAGS)
    z = ptb.step_scalars(0.5, 2e-5, 1e-4, 2.0, 0.3, "train/messages_per_s", 12, 0.0, mem)
    assert z["train/messages_per_s"] is None and "train/items_per_s" not in z
    assert z["train/lr_encoder"] == 2e-5 and z["train/sigma"] == 0.3


def test_step_scalars_refuse_an_unknown_rate_tag() -> None:
    with pytest.raises(ValueError, match="train/x"):
        ptb.step_scalars(0.5, None, 1e-4, 2.0, None, "train/x", 12, 2.0, {})


def test_epoch_scalars_convert_units() -> None:
    e = ptb.epoch_scalars(0.25, 10.0, 12.5, None, 2 * ptb.GIB, -3 * ptb.MIB)
    assert e == {"epoch/loss": 0.25, "epoch/train_s": 10.0, "epoch/wall_s": 12.5, "epoch/save_s": None,
                 "epoch/peak_total_gib": 2.0, "epoch/swap_delta_mib": -3.0}
    assert set(e) == set(ptb.EPOCH_TAGS)


def test_write_refuses_an_unknown_tag() -> None:
    w = FakeWriter()
    with pytest.raises(ValueError, match="train/text"):
        ptb.write(w, 3, {"train/loss": 1.0, "train/text": 2.0})
    assert w.calls == []                                    # nothing written before the refusal


def test_write_skips_none_in_sorted_order() -> None:
    w = FakeWriter()
    ptb.write(w, 7, {"train/loss": 1, "memory/rss_gib": 0.5, "train/sigma": None, "epoch/loss": 2.0})
    assert w.calls == [("epoch/loss", 2.0, 7), ("memory/rss_gib", 0.5, 7), ("train/loss", 1.0, 7)]
    assert all(type(v) is float for _, v, _ in w.calls)


def test_open_writer_round_trip(tmp_path: Path) -> None:
    logdir = str(tmp_path / "tb" / "rid")
    w = ptb.open_writer(logdir, 3, ptb.segment_suffix(0))
    ptb.write(w, 3, {"train/loss": 0.5})
    ptb.write(w, 4, {"train/loss": 0.25, "epoch/loss": 0.375})
    w.close()
    files = tb_events.event_files(logdir)
    assert len(files) == 1 and files[0].endswith(".segment-000"), files
    events = tb_events.read_events(logdir)
    assert {kind for _, _, kind, _ in events} == {"file_version", "session_log", "train/loss", "epoch/loss"}
    assert [(step, value) for _, step, kind, value in events if kind == "session_log"] == [(3, 1)]
    assert tb_events.scalars(events, "train/loss") == {3: 0.5, 4: 0.25}
    assert tb_events.scalars(events, "epoch/loss") == {4: 0.375}


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs a dir the user cannot write")
def test_open_writer_refuses_an_unwritable_dir(tmp_path: Path) -> None:
    parent = tmp_path / "ro"
    parent.mkdir()
    os.chmod(parent, 0o555)
    try:
        with pytest.raises(SystemExit) as e:
            ptb.open_writer(str(parent / "tb"), 1, ptb.segment_suffix(0))
    finally:
        os.chmod(parent, 0o755)
    msg: Any = e.value.code
    assert str(msg).startswith("--tensorboard: cannot write"), msg


def _refused(path: str, outputs: dict[str, str | None] | None = None) -> str:
    """The SystemExit message of check_logdir(path, outputs), asserting it refuses with the --tensorboard prefix."""
    with pytest.raises(SystemExit) as e:
        ptb.check_logdir(path, outputs or {})
    msg = str(e.value.code)
    assert msg.startswith("--tensorboard "), msg
    return msg


@pytest.mark.parametrize("rel", ["results", "results/new/deep", "data", "data/tb", "ui", "ui/dist/tb",
                                 "protocol.json", "protocol.json/tb"])
def test_check_logdir_refuses_frozen_and_guarded_paths(rel: str) -> None:
    path = os.path.join(REPO_ROOT, rel)
    existed = os.path.exists(path)
    msg = _refused(path)
    assert rel.split("/")[0] in msg, msg
    assert os.path.exists(path) == existed                   # nothing created


def test_check_logdir_follows_symlinks(tmp_path: Path) -> None:
    for name in ("data", "protocol.json"):
        link = tmp_path / ("to-" + name)
        link.symlink_to(os.path.join(REPO_ROOT, name))
        for p in (str(link), str(link / "tb")):
            assert name in _refused(p)


def test_check_logdir_refuses_a_case_variant() -> None:
    if not os.path.exists(os.path.join(REPO_ROOT, "RESULTS")):
        pytest.skip("case-sensitive filesystem: no case variant to refuse")
    path = os.path.join(REPO_ROOT, "RESULTS", "tb")
    assert "results" in _refused(path)
    assert not os.path.exists(path)


def test_check_logdir_refuses_a_path_inside_a_file(tmp_path: Path) -> None:
    f = tmp_path / "file"
    f.write_text("x", encoding="utf-8")
    assert "lies inside the file %s" % f in _refused(str(f / "x"))
    assert "exists and is not a directory" in _refused(str(f))
    for name in ("README.md", "Makefile"):
        assert "lies inside the file" in _refused(os.path.join(REPO_ROOT, name, "tb"))


def test_check_logdir_refuses_the_run_outputs(tmp_path: Path) -> None:
    storage, out = tmp_path / "storage", tmp_path / "out"
    storage.mkdir()
    outputs: dict[str, str | None] = {"--storage": str(storage), "--out": str(out), "--other": None}
    assert "--storage" in _refused(str(storage), outputs)
    assert "--out" in _refused(str(out / "tb"), outputs)
    link = tmp_path / "to-storage"
    link.symlink_to(storage)
    assert "--storage" in _refused(str(link / "tb"), outputs)
    assert ptb.check_logdir(str(tmp_path / "tb"), outputs) == os.path.realpath(tmp_path / "tb")   # None skipped
    assert not (out / "tb").exists() and not (tmp_path / "tb").exists()


def test_check_logdir_accepts_and_creates_nothing(tmp_path: Path) -> None:
    deep = tmp_path / "a" / "b" / "c"
    assert ptb.check_logdir(str(deep), {}) == os.path.realpath(deep)
    assert not (tmp_path / "a").exists()
    empty = tmp_path / "empty"
    empty.mkdir()
    assert ptb.check_logdir(str(empty), {}) == os.path.realpath(empty)
    assert os.listdir(empty) == []


@pytest.mark.timeout(300)
def test_symlinked_run_dir_into_results_is_refused_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                              capsys: pytest.CaptureFixture[str]) -> None:
    """DIR is an ordinary dir, but DIR/RUN_ID is a symlink into results/: ray_train.py refuses it once the run id is
    known, before --print-run-id prints it and before ray.init, and results/ is left as it was."""
    import ray_train
    root, raw_sha256 = make_raw(str(tmp_path / "raw"))
    out_dir = str(tmp_path / "data")
    pilot.data.prepare(root, out_dir, raw_sha256)
    base = tiny_checkpoint(tmp_path / "base")
    args = ["--data", os.path.join(out_dir, "splits", "selection_train.jsonl"),
            "--manifest", os.path.join(out_dir, "splits.json"), "--base", base, "--device", "cpu", "--workers", "1",
            "--cpus-per-worker", "1", "--epochs", "2", "--max-messages", "12", "--neg", "2",
            "--out", str(tmp_path / "o"), "--storage", str(tmp_path / "s")]

    def no_ray(*a: Any, **kw: Any) -> None:
        raise AssertionError("ray.init reached; the symlinked run dir must be refused first")
    monkeypatch.setattr(ray_train.ray, "init", no_ray)
    ray_train.main(args + ["--print-run-id"])
    rid = capsys.readouterr().out.strip().splitlines()[-1]
    assert rid.startswith("laya-"), rid
    tb = tmp_path / "tb"
    tb.mkdir()
    (tb / rid).symlink_to(os.path.join(REPO_ROOT, "results"))
    before = sorted(os.listdir(os.path.join(REPO_ROOT, "results")))
    with pytest.raises(SystemExit) as e:
        ray_train.main(args + ["--tensorboard", str(tb), "--print-run-id"])
    msg = str(e.value.code)
    assert msg.startswith("--tensorboard") and "results" in msg, msg
    assert rid not in capsys.readouterr().out                # no run id printed
    assert sorted(os.listdir(os.path.join(REPO_ROOT, "results"))) == before
    assert not (tmp_path / "s").exists() and not (tmp_path / "o").exists()
