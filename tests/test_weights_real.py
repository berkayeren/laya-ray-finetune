"""Pinned weights against the real Hugging Face hub: pinned revisions resolve, a moving ref is refused,
laya's own digest check engages, and BERTurk loads from safetensors. Network on the first run (~1.1 GB), cached
afterwards; the dry-run revision checks always ask the hub.
"""
import os
from typing import Any

import pytest
from huggingface_hub.errors import RevisionNotFoundError

from pilot import pins
from pilot.weights import check_revision, fetch

pytestmark = pytest.mark.weights

PICKLE_SUFFIXES = (".bin", ".h5", ".msgpack")


@pytest.fixture(scope="module")
def laya_dir() -> str:
    return fetch(pins.LAYA)


@pytest.fixture(scope="module")
def laya_agent(laya_dir: str) -> Any:
    import laya
    return laya.load(laya_dir, device="cpu", expected_sha256=pins.LAYA["sha256"])


def _snapshot_files(d: str) -> list[str]:
    return [os.path.join(dp, n) for dp, _, names in os.walk(d) for n in names]


@pytest.mark.parametrize("pin", [pins.LAYA, pins.BERTURK], ids=["laya", "berturk"])
def test_pinned_revision_resolves_every_file(pin: dict[str, Any]) -> None:
    assert check_revision(pin) >= 4


def test_moving_ref_is_refused() -> None:
    # "main" resolves to a concrete commit hash, which is never the requested ref string: refused before download
    with pytest.raises(SystemExit, match="resolves to commit"):
        check_revision(dict(pins.LAYA, revision="main"))


def test_nonexistent_revision_raises() -> None:
    with pytest.raises(RevisionNotFoundError):                # not a network error passing as a refusal
        check_revision(dict(pins.LAYA, revision="0123456789abcdef0123456789abcdef01234567"))


@pytest.mark.parametrize("pin", [pins.LAYA, pins.BERTURK], ids=["laya", "berturk"])
def test_fetch_downloads_no_pickle_tf_or_flax(pin: dict[str, Any]) -> None:
    d = fetch(pin)
    root = os.path.dirname(d) if pin["subfolder"] else d     # scan the whole snapshot, not only the subfolder
    assert os.path.basename(root) == pin["revision"]
    files = _snapshot_files(root)
    assert any(f.endswith("model.safetensors") for f in files), files
    assert not [f for f in files if f.endswith(PICKLE_SUFFIXES)]


def test_laya_loads_with_pinned_digests(laya_agent: Any) -> None:
    assert callable(getattr(laya_agent, "predict_batch", None))


def test_laya_digest_check_engages(laya_dir: str) -> None:
    import laya
    bad = dict(pins.LAYA["sha256"], **{"rl_agent_config.json": "0" * 64})
    with pytest.raises(ValueError):
        laya.load(laya_dir, device="cpu", expected_sha256=bad)


def test_berturk_loads_from_safetensors() -> None:
    from transformers import AutoModel, AutoTokenizer
    d = fetch(pins.BERTURK)
    tok = AutoTokenizer.from_pretrained(d)
    ids = tok("İstanbul'daki hesabım")["input_ids"]
    assert len(ids) > 2                                     # more than [CLS] [SEP]
    model = AutoModel.from_pretrained(d, use_safetensors=True)
    assert next(model.parameters()).device.type == "cpu"
