"""Environment-drift guard: the active interpreter must be the locked Python 3.12 environment.

make test runs in whatever interpreter is active, so an unpinned conda env would silently produce numbers the lock
cannot reproduce. These tests compare every requirements.txt pin that applies to this platform against the installed
distributions, refuse installed distributions the lock does not list (installed == lock, apart from the installer
itself), and check the lock itself: hashes on every entry, pyproject pins equal to the lock, and no package-index
override lines (every package comes from the default index, hash-checked). The drift detector is tested against a
synthetic wrong pin.
"""
import importlib.metadata as md
import os
import sys
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

pytestmark = pytest.mark.unit

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK = os.path.join(REPO_ROOT, "requirements.txt")
PYPROJECT = os.path.join(REPO_ROOT, "pyproject.toml")
INSTALLER = {"pip", "setuptools", "wheel"}               # venv bootstrap, never imported by the pilot
INDEX_OVERRIDES = ("--index-url", "--extra-index-url", "--find-links", "-i ", "-i=", "-f ", "-f=")


def requirement_entries(path: str = LOCK) -> list[str]:
    """Requirement entries of a hashed lock: continuation lines joined; blank, comment and option lines skipped."""
    with open(path, encoding="utf-8") as f:
        text = f.read().replace("\\\n", " ")
    entries = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "--")):
            continue
        entries.append(line)
    return entries


def _pinned_version(req: Requirement) -> str:
    specs = list(req.specifier)
    assert len(specs) == 1 and specs[0].operator in ("==", "==="), \
        "lock entry for %s is not an exact pin: %s" % (req.name, req.specifier)
    return specs[0].version


def locked_pins(path: str = LOCK) -> dict[str, str]:
    """canonical name -> pinned version for every lock entry whose marker is absent or true on this platform."""
    pins = {}
    for entry in requirement_entries(path):
        req = Requirement(entry.split(" --hash")[0].strip())
        if req.marker is None or req.marker.evaluate():
            pins[canonicalize_name(req.name)] = _pinned_version(req)
    return pins


def _same_version(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    try:
        return Version(a) == Version(b)
    except InvalidVersion:
        return a == b


def mismatches(lock_path: str = LOCK) -> dict[str, tuple[str, str | None]]:
    """{name: (locked, installed_or_None)} for every applicable pin the active interpreter does not satisfy."""
    installed = {canonicalize_name(d.metadata["Name"]): d.version for d in md.distributions()}
    return {name: (version, installed.get(name)) for name, version in sorted(locked_pins(lock_path).items())
            if not _same_version(version, installed.get(name))}


def unlocked(lock_path: str = LOCK) -> dict[str, str]:
    """Installed distributions that no lock entry names (on any platform), apart from the installer."""
    names = {canonicalize_name(Requirement(e.split(" --hash")[0].strip()).name) for e in requirement_entries(lock_path)}
    installed = {canonicalize_name(d.metadata["Name"]): d.version for d in md.distributions()}
    return {n: v for n, v in sorted(installed.items()) if n not in names and n not in INSTALLER}


def test_python_is_312() -> None:
    assert sys.version_info[:2] == (3, 12), \
        "interpreter %s is Python %d.%d, the lock is for 3.12; run `make env` and use .venv/bin/python" \
        % (sys.executable, sys.version_info[0], sys.version_info[1])


def test_installed_matches_lock() -> None:
    bad = mismatches(LOCK)
    assert not bad, "installed packages differ from requirements.txt in %s (run `make env`):\n%s" % (
        sys.executable,
        "\n".join("  %s: locked %s, installed %s" % (n, v, i or "MISSING") for n, (v, i) in bad.items()))


def test_nothing_installed_beyond_the_lock() -> None:
    extra = unlocked(LOCK)
    assert not extra, "installed packages not in requirements.txt in %s (recreate with `make env`):\n%s" % (
        sys.executable, "\n".join("  %s %s" % kv for kv in extra.items()))


def test_every_lock_entry_is_hashed() -> None:
    entries = requirement_entries(LOCK)
    assert len(entries) > 50, "requirements.txt parsed to only %d entries" % len(entries)
    unhashed = [e.split(" --hash")[0].strip() for e in entries if "--hash=sha256:" not in e]
    assert not unhashed, "requirements.txt entries without --hash=sha256: %r" % unhashed


def test_pyproject_pins_equal_lock() -> None:
    with open(PYPROJECT, "rb") as f:
        project = tomllib.load(f)["project"]
    direct = list(project["dependencies"]) + list(project["optional-dependencies"]["dev"])
    pins = locked_pins(LOCK)
    bad = {}
    for spec in direct:
        req = Requirement(spec)
        name = canonicalize_name(req.name)
        if not _same_version(_pinned_version(req), pins.get(name)):
            bad[name] = (_pinned_version(req), pins.get(name))
    assert not bad, "pyproject.toml pins differ from requirements.txt (run `make lock`): %r" % bad


def test_no_package_index_override() -> None:
    with open(LOCK, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f]
    bad = [ln for ln in lines if ln.startswith(INDEX_OVERRIDES)]
    assert not bad, "requirements.txt overrides the package index: %r" % bad


def test_drift_detector_reports_wrong_pin(tmp_path: Path) -> None:
    dummy = "--hash=sha256:" + "0" * 64
    lock = tmp_path / "requirements.txt"
    lock.write_text(
        "# synthetic lock\n"
        "laya==0.0.1 \\\n    %s\n    # via laya-pilot\n"
        "packaging==%s \\\n    %s\n"
        "not-a-real-package==1.0 ; sys_platform == 'no-such-platform' \\\n    %s\n"
        % (dummy, md.version("packaging"), dummy, dummy), encoding="utf-8")
    bad = mismatches(str(lock))
    assert bad == {"laya": ("0.0.1", md.version("laya"))}, bad


def test_unlocked_detector_reports_an_extra_distribution(tmp_path: Path) -> None:
    lock = tmp_path / "requirements.txt"
    lock.write_text("laya==0.0.1 \\\n    --hash=sha256:%s\n" % ("0" * 64), encoding="utf-8")
    extra = unlocked(str(lock))
    assert "laya" not in extra and "pip" not in extra
    assert extra.get("torch") == md.version("torch")
