# SPDX-FileCopyrightText: Copyright (c) 2026 SpaceAI
# SPDX-License-Identifier: Apache-2.0
"""Index-only MP4 files: the boxes the format's video rules read, and no frame to decode.

Only the standard library, so that the fake ffmpeg of the tests can import it.
"""

import math
import struct

DEFAULT = object()


def box(kind, payload):
    return struct.pack(">I4s", 8 + len(payload), kind) + payload


def sample_entry(codec=b"av01", width=96, height=64, flags=0x0C, av1c=True):
    """A visual sample entry; flags is the third byte of av1C (0x0C: 8-bit 4:2:0)."""
    body = (
        bytes(6)
        + struct.pack(">H", 1)
        + bytes(16)
        + struct.pack(">HHII", width, height, 0x00480000, 0x00480000)
        + bytes(4)
        + struct.pack(">H", 1)
        + bytes(32)
        + struct.pack(">Hh", 0x18, -1)
    )
    child = b""
    if codec == b"av01" and av1c:
        child = box(b"av1C", bytes([0x81, 0x00, flags, 0x00]))
    return box(codec, body + child)


def full_box(kind, version, payload):
    return box(kind, struct.pack(">B3s", version, bytes(3)) + payload)


def mvhd(timescale=1000, duration=0):
    return full_box(
        b"mvhd", 0, struct.pack(">IIII", 0, 0, timescale, duration) + bytes(80)
    )


def mdhd(timescale, duration, version=0):
    if version == 1:
        return full_box(
            b"mdhd", 1, struct.pack(">QQIQ", 0, 0, timescale, duration) + bytes(4)
        )
    return full_box(
        b"mdhd", 0, struct.pack(">IIII", 0, 0, timescale, duration) + bytes(4)
    )


def elst(entries, version=0):
    form = ">Qqhh" if version == 1 else ">Iihh"
    return full_box(
        b"elst",
        version,
        struct.pack(">I", len(entries))
        + b"".join(struct.pack(form, *e) for e in entries),
    )


def track(
    n_frames,
    timescale=15360,
    delta=512,
    entry=None,
    handler=b"vide",
    mdhd_box=None,
    descriptions=1,
    runs=None,
    edits=DEFAULT,
    ctts=None,
    duration=None,
    movie_timescale=1000,
    edts_box=None,
    ctts_box=None,
):
    """A trak box. runs are the time-to-sample table's (count, duration) pairs; edits are
    elst entries (segment duration, media time, rate, rate fraction), one that plays the
    whole track by default, None for no edts box; ctts the composition offsets. A box
    given whole (mdhd_box, edts_box, ctts_box) is used as it is."""
    runs = runs or [(n_frames, delta)]
    total = sum(count * d for count, d in runs)
    if duration is None:
        duration = total
    if edits is DEFAULT:
        edits = [(math.ceil(duration * movie_timescale / timescale), 0, 1, 0)]
    if mdhd_box is None:
        mdhd_box = mdhd(timescale, duration)
    hdlr = box(b"hdlr", bytes(8) + handler + bytes(12) + b"\0")
    entries = (entry or sample_entry()) * descriptions
    stsd = box(b"stsd", bytes(4) + struct.pack(">I", descriptions) + entries)
    stsz = box(
        b"stsz", bytes(4) + struct.pack(">II", 0, n_frames) + bytes(4 * n_frames)
    )
    stts = box(
        b"stts",
        bytes(4)
        + struct.pack(">I", len(runs))
        + b"".join(struct.pack(">II", *run) for run in runs),
    )
    tables = stsd + stts + stsz + (ctts_box or b"")
    if ctts is not None:
        tables += box(
            b"ctts",
            bytes(4)
            + struct.pack(">I", len(ctts))
            + b"".join(struct.pack(">II", 1, offset) for offset in ctts),
        )
    stbl = box(b"stbl", tables)
    edts = b"" if edits is None else box(b"edts", elst(edits))
    if edts_box is not None:
        edts = edts_box
    return box(b"trak", edts + box(b"mdia", hdlr + box(b"minf", stbl) + mdhd_box))


def minimal_mp4(
    n_frames,
    timescale=15360,
    delta=512,
    fragmented=False,
    tracks=1,
    moov_boxes=1,
    payload=bytes(16),
    mvhd_box=None,
    **kwargs,
):
    """An MP4 index of tracks video tracks (see track), then an mdat of payload."""
    if "mdhd" in kwargs:
        kwargs["mdhd_box"] = kwargs.pop("mdhd")
    movie_timescale = kwargs.get("movie_timescale", 1000)
    trak = track(n_frames, timescale, delta, **kwargs)
    if mvhd_box is None:
        mvhd_box = mvhd(movie_timescale)
    moov = box(b"moov", mvhd_box + trak * tracks)
    body = box(b"ftyp", b"isom" + bytes(4)) + moov * moov_boxes
    if fragmented:
        body += box(b"moof", b"")
    return body + box(b"mdat", payload)


def av1_video(n_frames, width, height, fps, payload):
    """The index ffmpeg with SVT-AV1 writes for n_frames frames at fps, around payload."""
    return minimal_mp4(
        n_frames,
        timescale=512 * fps,
        delta=512,
        entry=sample_entry(width=width, height=height),
        payload=payload,
    )
