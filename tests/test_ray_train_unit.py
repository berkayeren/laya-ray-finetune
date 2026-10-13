"""ray_train.py helpers that run inside Ray workers, tested without starting Ray or loading a model."""
import hashlib
import json

import pytest

ray_train = pytest.importorskip("ray_train")                # imports ray, torch and laya (the locked env has them)

pytestmark = pytest.mark.unit


def _write(path, rows):
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def test_read_verified_rows_parses_the_verified_bytes(tmp_path):
    rows = [{"uid": "a", "state": "Kartım kayboldu"}, {"uid": "b", "state": "Evet."}]
    digest = _write(tmp_path / "s.jsonl", rows)
    assert ray_train.read_verified_rows(str(tmp_path / "s.jsonl"), digest) == rows


def test_read_verified_rows_refuses_a_file_changed_after_driver_verification(tmp_path):
    digest = _write(tmp_path / "s.jsonl", [{"uid": "a", "state": "x"}])
    _write(tmp_path / "s.jsonl", [{"uid": "a", "state": "y"}])
    with pytest.raises(SystemExit, match="verified on the driver"):
        ray_train.read_verified_rows(str(tmp_path / "s.jsonl"), digest)
