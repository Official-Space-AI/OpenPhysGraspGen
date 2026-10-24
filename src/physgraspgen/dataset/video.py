# SPDX-FileCopyrightText: Copyright (c) 2026 SpaceAI
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .check_dataset import video_error, video_tracks


@dataclass(frozen=True)
class VideoSpec:
    """How RGB is encoded. AV1 in MP4, 8-bit 4:2:0 (yuv420p), is fixed by the format.

    crf, gop and preset are passed to ffmpeg as -crf, -g and -preset, within the ranges
    that its libsvtav1 encoder applies: it takes crf 0 and gop 1 as unset, and SVT-AV1
    4.2 runs presets 12 and 13 as 11.
    """

    encoder: str = "libsvtav1"
    crf: int = 30
    gop: int = 2
    preset: int = 8

    def __post_init__(self) -> None:
        if not isinstance(self.encoder, str) or not self.encoder:
            raise ValueError("encoder must be a nonempty string")
        if not 1 <= self.crf <= 63:
            raise ValueError("crf must be between 1 and 63")
        if self.gop < 2:
            raise ValueError("gop must be at least 2")
        if not 0 <= self.preset <= 11:
            raise ValueError("preset must be between 0 and 11")

    def record(self) -> dict[str, object]:
        """The encoder settings, which every episode of a dataset shares."""
        return {
            "codec": "av1",
            "pix_fmt": "yuv420p",
            "encoder": self.encoder,
            "crf": self.crf,
            "gop": self.gop,
            "preset": self.preset,
        }


def find_ffmpeg(executable: str = "ffmpeg", *, reading: bool = False) -> str:
    path = shutil.which(executable)
    if path is None:
        need = (
            "reading RGB needs an ffmpeg executable with an AV1 decoder"
            if reading
            else "writing RGB needs an ffmpeg executable with an AV1 encoder"
        )
        raise RuntimeError(f"{need}; {executable!r} was not found on the PATH")
    return path


def ffmpeg_version(ffmpeg: str) -> str:
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-version"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()[0].strip() if result.stdout else "unknown"


def check_encoder(ffmpeg: str, encoder: str) -> None:
    """Refuse an ffmpeg whose encoder list lacks encoder, or names it for another codec."""
    listing = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    # After the legend, each line is: flags, name, description.
    lines = listing.split("------", 1)[-1].splitlines()
    found = [line for line in lines if line.split()[1:2] == [encoder]]
    if not found:
        raise RuntimeError(
            f"{ffmpeg} has no {encoder} encoder (ffmpeg -encoders does not list it); "
            "writing RGB needs an ffmpeg built with it"
        )
    if not found[0].rstrip().endswith("(codec av1)"):
        raise RuntimeError(f"{encoder} is not an AV1 encoder: {found[0].strip()}")


class VideoEncoder:
    """One camera's RGB of one episode, piped frame by frame into ffmpeg."""

    def __init__(
        self,
        path: Path,
        size: tuple[int, int],
        fps: int,
        spec: VideoSpec,
        ffmpeg: str,
    ) -> None:
        self.path = path
        self.size = size
        self.fps = fps
        self.n_frames = 0
        height, width = size
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            spec.encoder,
            "-pix_fmt",
            "yuv420p",
            "-crf",
            str(spec.crf),
            "-g",
            str(spec.gop),
            "-preset",
            str(spec.preset),
            "-movflags",
            "+faststart",
            "-f",
            "mp4",
            "-y",
            str(path),
        ]
        # ffmpeg's messages go to a file: a full stderr pipe would block the encoder.
        self._stderr = tempfile.TemporaryFile()  # noqa: SIM115, closed in close and abort
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self._stderr,
            )
        except BaseException:
            self._stderr.close()
            raise

    def _message(self) -> str:
        self._stderr.seek(0)
        return self._stderr.read().decode(errors="replace").strip()

    def write(self, image: np.ndarray) -> None:
        stdin = self._process.stdin
        assert stdin is not None
        try:
            stdin.write(np.ascontiguousarray(image, dtype=np.uint8).tobytes())
        except BrokenPipeError:
            self._process.wait()
            raise RuntimeError(
                f"ffmpeg stopped while encoding {self.path}: {self._message()}"
            )
        self.n_frames += 1

    def close(self) -> None:
        stdin = self._process.stdin
        assert stdin is not None
        try:
            stdin.close()
        except BrokenPipeError:
            pass
        code = self._process.wait()
        message = self._message()
        self._stderr.close()
        if code != 0:
            raise RuntimeError(f"ffmpeg failed on {self.path} (exit {code}): {message}")
        tracks = video_tracks(self.path)
        if len(tracks) != 1:
            raise RuntimeError(f"{self.path} has {len(tracks)} video tracks, not one")
        problem = video_error(tracks[0], self.size, self.fps, self.n_frames)
        if problem:
            raise RuntimeError(f"{self.path} {problem}")

    def abort(self) -> None:
        if self._process.poll() is None:
            self._process.kill()
        self._process.wait()
        if self._process.stdin is not None:
            try:
                self._process.stdin.close()
            except BrokenPipeError:
                pass
        self._stderr.close()


def read_video(
    path: str | Path,
    size: tuple[int, int] | None = None,
    ffmpeg: str = "ffmpeg",
    *,
    n_frames: int | None = None,
) -> np.ndarray:
    """Decode an episode's video to an (N, H, W, 3) uint8 array.

    N, H and W come from the video's index; size (H, W) and n_frames, when given, must
    agree with it. Video frame i is the episode's frame i. Decoding needs an ffmpeg with
    an AV1 decoder, such as libdav1d.
    """
    tracks = video_tracks(path)
    if len(tracks) != 1:
        raise ValueError(f"{path} has {len(tracks)} video tracks, not one")
    track = tracks[0]
    height, width = track.height, track.width
    if size is not None and tuple(size) != (height, width):
        raise ValueError(
            f"{path} is {height} x {width} pixels, not {size[0]} x {size[1]}"
        )
    if n_frames is not None and n_frames != track.n_frames:
        raise ValueError(f"{path} holds {track.n_frames} frames, not {n_frames}")
    frames = np.empty((track.n_frames, height, width, 3), dtype=np.uint8)
    buffer = memoryview(frames.reshape(-1))
    command = [
        find_ffmpeg(ffmpeg, reading=True),
        "-hide_banner",
        "-loglevel",
        "error",
        "-xerror",
        "-i",
        str(path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    with tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=stderr, bufsize=0
        )
        stdout = process.stdout
        assert stdout is not None
        filled = 0
        try:
            while filled < len(buffer):
                got = stdout.readinto(buffer[filled:])
                if not got:
                    break
                filled += got
            more = filled == len(buffer) and bool(stdout.read(1))
        finally:
            stdout.close()
            code = process.wait()
        stderr.seek(0)
        message = stderr.read().decode(errors="replace").strip()
    if more:
        raise RuntimeError(
            f"{path} decoded to more frames than the {track.n_frames} of its index"
        )
    if code != 0:
        raise RuntimeError(f"ffmpeg could not decode {path} (exit {code}): {message}")
    if filled != len(buffer):
        decoded = filled // (height * width * 3)
        raise RuntimeError(
            f"{path} decoded to {decoded} frames, its index has {track.n_frames}"
        )
    return frames
