# SPDX-FileCopyrightText: Copyright (c) 2026 SpaceAI
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .check_dataset import video_tracks


@dataclass(frozen=True)
class VideoSpec:
    """How RGB is encoded. AV1 in MP4 with yuv420p is fixed by the format."""

    encoder: str = "libsvtav1"
    crf: int = 30
    gop: int = 2
    preset: int | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.crf <= 63:
            raise ValueError("crf must be between 0 and 63")
        if self.gop < 1:
            raise ValueError("gop must be at least 1")

    def record(self, ffmpeg_version: str) -> dict[str, object]:
        return {
            "codec": "av1",
            "pix_fmt": "yuv420p",
            "encoder": self.encoder,
            "crf": self.crf,
            "gop": self.gop,
            "preset": self.preset,
            "ffmpeg": ffmpeg_version,
        }


def find_ffmpeg(executable: str = "ffmpeg") -> str:
    path = shutil.which(executable)
    if path is None:
        raise RuntimeError(
            f"writing RGB needs an ffmpeg executable with an AV1 encoder; {executable!r} "
            "was not found on the PATH"
        )
    return path


def ffmpeg_version(ffmpeg: str) -> str:
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-version"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()[0].strip() if result.stdout else "unknown"


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
        ]
        if spec.preset is not None:
            command += ["-preset", str(spec.preset)]
        command += ["-movflags", "+faststart", "-f", "mp4", "-y", str(path)]
        # ffmpeg's messages go to a file: a full stderr pipe would block the encoder.
        self._stderr = tempfile.TemporaryFile()  # noqa: SIM115, closed in close and abort
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._stderr,
        )

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
        if tracks[0].n_frames != self.n_frames:
            raise RuntimeError(
                f"{self.path} holds {tracks[0].n_frames} frames, "
                f"{self.n_frames} were written"
            )
        if tracks[0].fps is None or abs(tracks[0].fps - self.fps) > 1e-9 * self.fps:
            raise RuntimeError(f"{self.path} is not at a constant {self.fps} fps")

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
    path: str | Path, size: tuple[int, int], ffmpeg: str = "ffmpeg"
) -> np.ndarray:
    """Decode an episode's video to an (N, H, W, 3) uint8 array; size is (H, W)."""
    result = subprocess.run(
        [
            find_ffmpeg(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"ffmpeg could not decode {path}: {message}")
    height, width = size
    frame_bytes = height * width * 3
    if len(result.stdout) % frame_bytes:
        raise RuntimeError(f"{path} did not decode to whole {width}x{height} frames")
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, height, width, 3)
