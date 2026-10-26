"""A crc-checked reader of TensorBoard event files, for tests, without the tensorboard package.

An event file is a sequence of TFRecord frames: an 8-byte little-endian length, the masked crc32c of those 8 bytes,
the data (one serialized tensorboardX Event), and the masked crc32c of the data. read_events checks both crcs of every
frame and parses the data with tensorboardX's own Event proto, so a truncated or corrupted file fails loudly. visible
applies TensorBoard's purge rule, so a test sees what a TensorBoard viewer shows.
"""
import glob
import os
import struct
from typing import Any

from tensorboardX.crc32c import crc32c
from tensorboardX.proto import event_pb2


def _masked(x: int) -> int:
    """The TFRecord masked crc32c of a raw crc value."""
    return (((x >> 15) | (x << 17)) + 0xa282ead8) & 0xffffffff


def event_files(logdir: str) -> list[str]:
    """The sorted paths of the events.out.tfevents.* files directly under logdir."""
    return sorted(glob.glob(os.path.join(logdir, "events.out.tfevents.*")))


def read_events(logdir: str) -> list[tuple[str, int, str, Any]]:
    """(file name, step, kind, value) of every event of every event file under logdir, file by file in sorted order.

    kind is the scalar tag (value: its float simple_value), "session_log" (value: the status int; START is 1) or
    "file_version" (value: the version string). AssertionError on a bad crc, a short frame, another event kind or a
    summary value without simple_value.
    """
    out: list[tuple[str, int, str, Any]] = []
    for path in event_files(logdir):
        with open(path, "rb") as f:
            raw = f.read()
        name, i = os.path.basename(path), 0
        while i < len(raw):
            assert i + 12 <= len(raw), "%s: short frame header at byte %d" % (name, i)
            (n,) = struct.unpack("<Q", raw[i:i + 8])
            assert struct.unpack("<I", raw[i + 8:i + 12])[0] == _masked(crc32c(raw[i:i + 8])), \
                "%s: bad length crc at byte %d" % (name, i)
            assert i + 16 + n <= len(raw), "%s: short frame at byte %d" % (name, i)
            data = raw[i + 12:i + 12 + n]
            assert struct.unpack("<I", raw[i + 12 + n:i + 16 + n])[0] == _masked(crc32c(data)), \
                "%s: bad data crc at byte %d" % (name, i)
            e, i = event_pb2.Event.FromString(data), i + 16 + n
            kind = e.WhichOneof("what")
            if kind == "summary":
                for v in e.summary.value:
                    assert v.WhichOneof("value") == "simple_value", "%s: %s is not a simple_value" % (name, v.tag)
                    out.append((name, e.step, v.tag, v.simple_value))
            elif kind == "session_log":
                out.append((name, e.step, "session_log", e.session_log.status))
            elif kind == "file_version":
                out.append((name, e.step, "file_version", e.file_version))
            else:
                raise AssertionError("%s: unexpected event kind %r at step %d" % (name, kind, e.step))
    return out


def scalars(events: list[tuple[str, int, str, Any]], tag: str) -> dict[int, float]:
    """{step: value} of one tag over `events`; AssertionError when a step repeats for that tag."""
    out: dict[int, float] = {}
    for _, step, kind, value in events:
        if kind == tag:
            assert step not in out, "%s logged twice at step %d" % (tag, step)
            out[step] = value
    return out


def visible(events: list[tuple[str, int, str, Any]]) -> list[tuple[str, int, str, Any]]:
    """The events a TensorBoard viewer keeps, in read order: its EventAccumulator purge rule, applied while reading the
    files in sorted order (as read_events returns them). A session_log START (status 1) at step P discards every event
    read before it at a step >= P, of every tag and every earlier file; the START event itself is kept."""
    out: list[tuple[str, int, str, Any]] = []
    for e in events:
        if e[2] == "session_log" and e[3] == 1:
            out = [k for k in out if k[1] < e[1]]
        out.append(e)
    return out
