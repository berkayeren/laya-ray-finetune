"""Manifest contract of pilot.data: shared verifier, determinism, producer refusal, atomic writes, hygiene.

Every test passes manifest_path explicitly and works in tmp_path; the repo's data/ is never touched.
"""
import json
import os
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

import pilot.data
from conftest import SyntheticRaw, prepared

pytestmark = pytest.mark.unit

SPLIT_NAMES = ("selection_train", "val", "final_train", "test")


def _split(out_dir: str, name: str) -> str:
    return os.path.join(out_dir, "splits", "%s.jsonl" % name)


def _read(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _rewrite_manifest(manifest_path: str, edit: Callable[[Any], object]) -> None:
    with open(manifest_path, encoding="utf-8") as f:
        m = json.load(f)
    edit(m)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2, sort_keys=True)


def test_verify_accepts_every_registered_split(tmp_path: Path, synthetic_raw: SyntheticRaw) -> None:
    out_dir, manifest_path = prepared(tmp_path, synthetic_raw)
    with open(manifest_path, encoding="utf-8") as f:
        m = json.load(f)
    for name in SPLIT_NAMES:
        entry = pilot.data.verify_split_file(_split(out_dir, name), manifest_path=manifest_path)
        assert entry == m["splits"][name]


def test_split_contents_label_all_48_and_keep_no_intent_rows(tmp_path: Path, synthetic_raw: SyntheticRaw) -> None:
    out_dir, manifest_path = prepared(tmp_path, synthetic_raw)
    with open(_split(out_dir, "selection_train"), encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    by_uid = {r["uid"]: r for r in rows}
    assert all(len(r["questions"]) == 48 and len(r["expected"]) == 48 for r in rows)
    assert "bank_3_1" in by_uid and not any(by_uid["bank_3_1"]["expected"].values())
    with open(manifest_path, encoding="utf-8") as f:
        s = json.load(f)["splits"]
    assert [s[k]["n_no_intent"] for k in SPLIT_NAMES] == [1, 1, 2, 1]
    assert s["val"]["folds"] == [14, 15] and s["test"]["role"] == "test"


def test_one_byte_tamper_is_refused(tmp_path: Path, synthetic_raw: SyntheticRaw) -> None:
    out_dir, manifest_path = prepared(tmp_path, synthetic_raw)
    with open(_split(out_dir, "test"), "ab") as f:
        f.write(b"x")
    with pytest.raises(SystemExit, match="sha256"):
        pilot.data.verify_split_file(_split(out_dir, "test"), manifest_path=manifest_path)


@pytest.mark.parametrize("field,edit", [
    ("questions_sha256", lambda m: m.update(questions_sha256="0" * 64)),
    ("intents_order", lambda m: m.update(intents_order=m["intents_order"][::-1])),
    ("schema_version", lambda m: m.update(schema_version=m["schema_version"] + 1)),
    ("dataset.revision", lambda m: m["dataset"].update(revision="0" * 40)),
])
def test_manifest_from_other_code_or_pins_is_refused(tmp_path: Path, synthetic_raw: SyntheticRaw, field: str,
                                                     edit: Callable[[Any], object]) -> None:
    out_dir, manifest_path = prepared(tmp_path, synthetic_raw)
    _rewrite_manifest(manifest_path, edit)
    with pytest.raises(SystemExit, match="different code/pins.*%s" % field.replace(".", r"\.")):
        pilot.data.verify_split_file(_split(out_dir, "test"), manifest_path=manifest_path)
    with pytest.raises(SystemExit, match="different code/pins"):
        pilot.data.verify_audit_file(os.path.join(out_dir, "audit", "audit.json"), manifest_path=manifest_path)


@pytest.mark.parametrize("edit", [
    lambda m: (m["splits"]["test"].update(role="train"), m["splits"]["hotels_probe"].update(role="test")),
    lambda m: m["splits"]["val"].update(folds=[16, 17, 18, 19]),
    lambda m: m["splits"]["hotels_probe"].update(domain="banking"),
    lambda m: m["splits"].update(extra=dict(m["splits"]["test"], path="splits/extra.jsonl", role="train")),
    lambda m: m["splits"].pop("val"),
], ids=["role_relabel", "folds", "domain", "extra_split", "missing_split"])
def test_manifest_cannot_relabel_splits(tmp_path: Path, synthetic_raw: SyntheticRaw,
                                        edit: Callable[[Any], object]) -> None:
    # a manifest may differ only in what prepare() varies; it can never turn test into train or the probe into test
    out_dir, manifest_path = prepared(tmp_path, synthetic_raw)
    _rewrite_manifest(manifest_path, edit)
    for name, roles in (("test", ("train",)), ("hotels_probe", ("test", "val")), ("selection_train", None)):
        with pytest.raises(SystemExit, match="SPLITS|registers splits"):
            pilot.data.verify_split_file(_split(out_dir, name), manifest_path=manifest_path, roles=roles)


def test_manifest_cannot_move_audit_entries(tmp_path: Path, synthetic_raw: SyntheticRaw) -> None:
    out_dir, manifest_path = prepared(tmp_path, synthetic_raw)
    _rewrite_manifest(manifest_path, lambda m: m["audit"].update(extra={"path": "audit/x.json", "sha256": "0" * 64}))
    with pytest.raises(SystemExit, match="AUDIT_FILES"):
        pilot.data.verify_split_file(_split(out_dir, "test"), manifest_path=manifest_path)


def test_manifest_is_committed_flags_other_manifests(tmp_path: Path, synthetic_raw: SyntheticRaw) -> None:
    _, manifest_path = prepared(tmp_path, synthetic_raw)
    assert pilot.data.manifest_is_committed(pilot.data.DEFAULT_MANIFEST) is True
    assert pilot.data.manifest_is_committed(manifest_path) is False


def test_committed_manifest_passes_the_current_code_check() -> None:
    with open(pilot.data.DEFAULT_MANIFEST, encoding="utf-8") as f:
        pilot.data.check_manifest_current(json.load(f), pilot.data.DEFAULT_MANIFEST)


def test_verify_hashes_the_bytes_the_caller_parses(tmp_path: Path, synthetic_raw: SyntheticRaw) -> None:
    out_dir, manifest_path = prepared(tmp_path, synthetic_raw)
    path = _split(out_dir, "test")
    raw = _read(path)
    assert pilot.data.verify_split_file(path, manifest_path=manifest_path, data=raw)["role"] == "test"
    with pytest.raises(SystemExit, match="sha256"):                # file on disk is intact, the parsed bytes are not
        pilot.data.verify_split_file(path, manifest_path=manifest_path, data=raw + b"x")


def test_missing_split_file_gets_a_make_data_hint(tmp_path: Path, synthetic_raw: SyntheticRaw) -> None:
    out_dir, manifest_path = prepared(tmp_path, synthetic_raw)
    os.remove(_split(out_dir, "selection_train"))
    with pytest.raises(SystemExit, match="not found.*make data"):
        pilot.data.verify_split_file(_split(out_dir, "selection_train"), manifest_path=manifest_path)


def test_unregistered_identical_copy_is_refused(tmp_path: Path, synthetic_raw: SyntheticRaw) -> None:
    out_dir, manifest_path = prepared(tmp_path, synthetic_raw)
    copy = os.path.join(out_dir, "splits", "test_copy.jsonl")
    shutil.copyfile(_split(out_dir, "test"), copy)
    with pytest.raises(SystemExit, match="not a split registered"):
        pilot.data.verify_split_file(copy, manifest_path=manifest_path)


def test_absolute_manifest_path_is_refused(tmp_path: Path, synthetic_raw: SyntheticRaw) -> None:
    out_dir, manifest_path = prepared(tmp_path, synthetic_raw)
    target = os.path.realpath(_split(out_dir, "test"))
    _rewrite_manifest(manifest_path, lambda m: m["splits"]["test"].update(path=target))
    with pytest.raises(SystemExit, match="unsafe path"):
        pilot.data.verify_split_file(target, manifest_path=manifest_path)


def test_dotdot_manifest_path_is_refused(tmp_path: Path, synthetic_raw: SyntheticRaw) -> None:
    out_dir, manifest_path = prepared(tmp_path, synthetic_raw)
    outside = tmp_path / "outside"
    outside.mkdir()
    shutil.copyfile(_split(out_dir, "test"), str(outside / "test.jsonl"))
    _rewrite_manifest(manifest_path, lambda m: m["splits"]["test"].update(path="../outside/test.jsonl"))
    with pytest.raises(SystemExit, match="unsafe path"):
        pilot.data.verify_split_file(str(outside / "test.jsonl"), manifest_path=manifest_path)


def test_role_restriction_refuses_test_split(tmp_path: Path, synthetic_raw: SyntheticRaw) -> None:
    out_dir, manifest_path = prepared(tmp_path, synthetic_raw)
    assert pilot.data.verify_split_file(_split(out_dir, "final_train"), manifest_path=manifest_path,
                                        roles=("train",))["role"] == "train"
    with pytest.raises(SystemExit, match="role"):
        pilot.data.verify_split_file(_split(out_dir, "test"), manifest_path=manifest_path, roles=("train",))


def test_prepare_is_deterministic_across_dirs(tmp_path: Path, synthetic_raw: SyntheticRaw) -> None:
    out_a, manifest_a = prepared(tmp_path, synthetic_raw, name="a")
    out_b, manifest_b = prepared(tmp_path, synthetic_raw, name="b")
    assert _read(manifest_a) == _read(manifest_b)
    for name in SPLIT_NAMES:
        assert _read(_split(out_a, name)) == _read(_split(out_b, name))


def test_prepare_refuses_changed_input_unless_rewrite(tmp_path: Path, synthetic_raw: SyntheticRaw) -> None:
    out_dir, manifest_path = prepared(tmp_path, synthetic_raw)
    before = {p: _read(p) for p in [manifest_path] + [_split(out_dir, n) for n in SPLIT_NAMES]}
    root, raw_sha256 = synthetic_raw()
    fold = os.path.join(root, "turkish", "banking", "fold3.json")
    with open(fold, encoding="utf-8") as f:
        rows = json.load(f)
    rows.append({"uid": "bank_3_9", "text": "Bakiyem ne kadar", "intents": ["balance", "how_much"], "lang": "tr"})
    with open(fold, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    raw_sha256["turkish/banking/fold3.json"] = pilot.data.sha256_file(fold)
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        pilot.data.prepare(root, out_dir, raw_sha256)
    assert {p: _read(p) for p in before} == before
    m = pilot.data.prepare(root, out_dir, raw_sha256, rewrite_manifest=True)
    assert m["splits"]["selection_train"]["n_messages"] == json.loads(before[manifest_path])["splits"][
        "selection_train"]["n_messages"] + 1
    assert _read(manifest_path) == pilot.data.serialize_manifest(m)


def test_prepare_refuses_raw_hash_mismatch(tmp_path: Path, synthetic_raw: SyntheticRaw) -> None:
    root, raw_sha256 = synthetic_raw()
    raw_sha256["turkish/hotels/fold7.json"] = "0" * 64
    with pytest.raises(SystemExit, match="turkish/hotels/fold7.json"):
        pilot.data.prepare(root, str(tmp_path / "data"), raw_sha256)
    assert not os.path.exists(str(tmp_path / "data"))


@pytest.mark.parametrize("edit", [lambda h: h.pop("turkish/banking/fold19.json"),
                                  lambda h: h.update({"turkish/banking/fold20.json": "0" * 64})],
                         ids=["missing_fold", "extra_fold"])
def test_prepare_refuses_raw_pins_that_do_not_cover_the_read_folds(tmp_path: Path, synthetic_raw: SyntheticRaw,
                                                                   edit: Callable[[Any], object]) -> None:
    root, raw_sha256 = synthetic_raw()
    edit(raw_sha256)
    with pytest.raises(SystemExit, match="does not pin exactly the 40 folds"):
        pilot.data.prepare(root, str(tmp_path / "data"), raw_sha256)
    assert not os.path.exists(str(tmp_path / "data"))


def test_interrupted_write_leaves_no_partial_file(tmp_path: Path, synthetic_raw: SyntheticRaw,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    root, raw_sha256 = synthetic_raw()
    files, manifest = pilot.data.build(root, raw_sha256)
    out_dir = str(tmp_path / "data")
    real_replace, calls = os.replace, []

    def flaky_replace(src: str, dst: str) -> None:
        calls.append(dst)
        if len(calls) == 2:
            raise OSError("simulated crash during os.replace")
        return real_replace(src, dst)

    monkeypatch.setattr(pilot.data.os, "replace", flaky_replace)
    with pytest.raises(OSError, match="simulated crash"):
        pilot.data.prepare(root, out_dir, raw_sha256)
    monkeypatch.undo()
    assert not os.path.exists(os.path.join(out_dir, "splits.json"))
    for rel, data in files.items():
        dest = os.path.join(out_dir, rel)
        if os.path.exists(dest):
            assert _read(dest) == data, "%s is not a complete file" % rel
    assert sum(os.path.exists(os.path.join(out_dir, rel)) for rel in files) == 1
    leftovers = [os.path.join(d, n) for d, _, names in os.walk(out_dir) for n in names if n.endswith(".tmp")]
    assert leftovers == [], leftovers


def test_manifest_hygiene(tmp_path: Path, synthetic_raw: SyntheticRaw) -> None:
    out_dir, manifest_path = prepared(tmp_path, synthetic_raw)
    text = _read(manifest_path).decode("utf-8")
    assert str(tmp_path) not in text and os.path.realpath(str(tmp_path)) not in text
    assert os.path.expanduser("~") not in text and "/Users/" not in text and "/home/" not in text

    def keys(o: object) -> Iterator[str]:
        if isinstance(o, dict):
            for k, v in o.items():
                yield k
                yield from keys(v)
        elif isinstance(o, list):
            for v in o:
                yield from keys(v)

    m = json.loads(text)
    bad = [k for k in keys(m) if "time" in k.lower() or "date" in k.lower()]
    assert not bad, bad
    assert all(not os.path.isabs(e["path"]) for e in m["splits"].values())
    assert m["questions_sha256"] == pilot.data.questions_sha256() and m["schema_version"] == 1
    assert text.endswith("}\n")
