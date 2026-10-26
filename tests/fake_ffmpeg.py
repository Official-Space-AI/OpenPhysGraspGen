# SPDX-FileCopyrightText: Copyright (c) 2026 SpaceAI
# SPDX-License-Identifier: Apache-2.0
"""A stand-in for ffmpeg with SVT-AV1, for writing RGB without an AV1 encoder.

It lists libsvtav1, answers -version, and for an encoding reads the raw RGB frames from
stdin and writes the index ffmpeg writes for them, around the frames' sha256, so that
two episodes' videos differ. No frame can be decoded from it; the format's checks read
the index only.
"""

import hashlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mp4_index import av1_video

args = sys.argv[1:]
if "-encoders" in args:
    print(" ------\n V....D libsvtav1            SVT-AV1 (codec av1)")
elif "-version" in args:
    print(os.environ.get("FAKE_FFMPEG_VERSION", "ffmpeg version 0.0-fake"))
else:
    width, height = (int(v) for v in args[args.index("-s") + 1].split("x"))
    fps = int(args[args.index("-r") + 1])
    frames = sys.stdin.buffer.read()
    n_frames = len(frames) // (width * height * 3)
    video = av1_video(n_frames, width, height, fps, hashlib.sha256(frames).digest())
    Path(args[-1]).write_bytes(video)
