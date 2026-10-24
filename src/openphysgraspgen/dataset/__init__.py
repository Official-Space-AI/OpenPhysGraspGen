# SPDX-FileCopyrightText: Copyright (c) 2026 SpaceAI
# SPDX-License-Identifier: Apache-2.0
"""Write and read the public dataset format (``pip install "openphysgraspgen[dataset]"``).

The format is documented in ``docs/dataset/format.md``.
"""

from .check_dataset import (
    CAMERAS,
    CHANNELS,
    FACT_KINDS,
    FORMAT,
    FORMAT_VERSION,
    SEGMENTATION_CLASSES,
    VideoTrack,
    video_tracks,
)
from .finalize import DatasetError, card_front_matter, finalize_dataset
from .reader import episode_metadata, read_frames, read_metadata, read_trajectory
from .video import VideoSpec, read_video
from .writer import DatasetSpec, EpisodeFiles, EpisodeWriter, GraspDataRecorder

__all__ = [
    "CAMERAS",
    "CHANNELS",
    "FACT_KINDS",
    "FORMAT",
    "FORMAT_VERSION",
    "SEGMENTATION_CLASSES",
    "DatasetError",
    "DatasetSpec",
    "EpisodeFiles",
    "EpisodeWriter",
    "GraspDataRecorder",
    "VideoSpec",
    "VideoTrack",
    "card_front_matter",
    "episode_metadata",
    "finalize_dataset",
    "read_frames",
    "read_metadata",
    "read_trajectory",
    "read_video",
    "video_tracks",
]
