# The public dataset format

Status: draft for review (format `physgraspgen-public`, version `2.0`).

The project publishes a soft-body grasping dataset on the Hugging Face Hub under CC-BY-4.0, with a
Zenodo DOI. This page defines its files. The `openphysgraspgen.dataset` module writes and reads them,
and the dataset's own check script tests them, so the writer, the reader and the check share one
definition. Version 2.0 replaces an unpublished HDF5 draft (1.0) with Parquet and MP4.

## What the format has to serve

- **The dataset test.** An independent test lab checks four items on the published files: scale
  (at least 1,000,000 frames, every frame counted), metadata integrity (no mismatch between
  `metadata.yaml` and the files), four-layer consistency (RGB, depth, force and stress of every
  frame recorded at one simulation step, and contact points projected into the images landing on
  the object) and diversity (at least 10 object types, with distribution tables).
- **The specification.** Every episode carries six ground-truth channels (RGB, depth, force,
  stress, deformation gradient, segmentation) from two cameras (`front`, `wrist`).
- **The Hugging Face Hub.** Parquet that the Hub's tools read without a loading script, at most
  10,000 files in a folder, and files far below the Hub's size limit.
- **Partial reads.** A user can read only the columns and episodes they need, locally or over
  HTTP, without downloading the whole dataset.
- **Benchmark rescoring.** The PG-T1 score can be recomputed from the dataset alone, on a CPU.

## Layout

```
<root>/
  README.md                                          dataset card
  metadata.yaml                                      counts, object types, conventions, every file
  SHA256SUMS                                         sha256 of every data and tool file
  data/<object_type>/<episode_uid>.parquet           frames: every channel except RGB
  trajectories/<object_type>/<episode_uid>.parquet   robot and object state at the recorded rate
  videos/<camera>/<object_type>/<episode_uid>.mp4    RGB of one camera
  tools/check_dataset.py                             the check script
  tools/check_criteria.yaml                          the acceptance values it applies
```

One episode is one frames file, one trajectory file and one video per camera. Files are grouped
by object type, so each type can be exported, uploaded and downloaded on its own.

## Frames file

`data/<object_type>/<episode_uid>.parquet` has one row per published frame, in simulation order.
A frame is one simulation step: the images of both cameras and every ground-truth channel
recorded at that step. N is the frame count, T the object's tetrahedron count, H x W a camera's
image size.

Column names are `<group>.<field>` or `<group>.<camera>.<field>`. Every channel has its own
`sim_step` column, the step its data was captured at, so the consistency item can compare the
layers of each frame.

| Column | Type | Meaning |
|---|---|---|
| `episode_uid` | string | the episode, so tables of several files can be concatenated |
| `frame_index` | int32 | 0 to N - 1 |
| `sim_step` | int64 | simulation step, counted at the recorded rate from the episode start (the first recorded frame is step 1), strictly increasing |
| `sim_time_s` | float64 | simulation time [s], `sim_step / recorded_fps` |
| `source_frame_index` | int32 | the frame's row in the generator's own record |
| `rgb.<camera>.sim_step` | int64 | step of the video frame |
| `rgb.<camera>.video_frame` | int32 | index of this frame in `videos/<camera>/<object_type>/<episode_uid>.mp4` |
| `depth.<camera>.sim_step` | int64 | step of the depth image |
| `depth.<camera>.depth_m` | fixed list of H x W float32 | forward (z) depth [m], row-major, row 0 at the top |
| `segmentation.<camera>.sim_step` | int64 | step of the mask |
| `segmentation.<camera>.mask` | fixed list of H x W uint8 | per-pixel class, pixel-aligned with the image |
| `camera.<camera>.intrinsics` | fixed list of 9 float64 | K for the stored image, row-major 3 x 3 |
| `camera.<camera>.world_to_camera` | fixed list of 16 float64 | world-to-camera transform, OpenCV axes, row-major 4 x 4 |
| `force.sim_step` | int64 | step of the force layer |
| `force.n_contacts` | int32 | soft contacts on the object |
| `force.contacts` | list of struct | one entry per contact: `force` (3 float32, N, world frame), `point_world` (3 float32, m), `particle` (int32), `is_ground` (bool), `is_robot` (bool) |
| `force.gripper_net_force`, `force.ground_net_force` | fixed list of 3 float32 | net force on the object from the gripper and from the support [N] |
| `force.gripper_force_mag`, `force.ground_force_mag` | float64 | their magnitudes [N] |
| `stress.sim_step` | int64 | step of the stress layer |
| `stress.von_mises` | list of T float32 | von Mises stress per tetrahedron [Pa], NaN where not valid |
| `stress.valid` | list of T bool | where stress is defined |
| `stress.volume_ratio` | list of T float32 | det F |
| `deformation_gradient.sim_step` | int64 | step of the deformation gradient |
| `deformation_gradient.F` | list of 9T float32 | deformation gradient per tetrahedron, row-major `F[t, r, c]` |
| `robot.ee_pose_cmd`, `robot.ee_pose` | fixed list of 7 float64 | end-effector pose, commanded by the waypoint and reached: position [m] and quaternion (x, y, z, w), world frame |
| `robot.arm_q_setpoint`, `robot.arm_q` | fixed list of 6 float64 | arm joint positions [rad], this frame's setpoint and reached |
| `robot.gripper_width_cmd_m`, `robot.gripper_width_setpoint_m`, `robot.gripper_width_m` | float64 | jaw opening [m]: the waypoint's command, this frame's setpoint, measured |
| `robot.grip_contact_width_m` | float64 | jaw opening at which the pads first carried force, NaN before |
| `robot.waypoint` | int32 | index of the waypoint being flown |
| `robot.phase` | string | the waypoint's phase, such as `lift` or `hold` |
| `object.centroid`, `object.bbox_min`, `object.bbox_max` | fixed list of 3 float64 | centroid and axis-aligned bounds of the deformed object [m], world frame |
| `object.n_soft_contacts`, `object.n_rigid_contacts`, `object.n_robot_contacts` | int32 | contacts in the collision pipeline |

Image sizes are fixed per camera for the whole dataset and stated in `metadata.yaml`. The
tetrahedron count T is fixed per episode and stated in the episode metadata. Each fixed list
column also carries its shape in the Parquet field metadata (`shape`).

## Trajectory file

`trajectories/<object_type>/<episode_uid>.parquet` has one row per recorded simulation step, from
the first to the last, so it keeps the steps that the published frame rate skips. It holds what
benchmark scoring and control learning read, and no images.

| Column | Type | Meaning |
|---|---|---|
| `episode_uid` | string | the episode |
| `sim_step`, `sim_time_s` | int64, float64 | as in the frames file |
| `finite` | bool | the object's state is finite at this step |
| `robot.waypoint`, `robot.phase` | int32, string | as in the frames file |
| `robot.ee_pose_cmd`, `robot.ee_pose`, `robot.arm_q_setpoint`, `robot.arm_q` | as in the frames file | |
| `robot.gripper_width_cmd_m`, `robot.gripper_width_setpoint_m`, `robot.gripper_width_m`, `robot.grip_contact_width_m` | float64 | as in the frames file |
| `object.centroid`, `object.bbox_min`, `object.bbox_max` | fixed list of 3 float64 | as in the frames file, NaN where not finite |
| `object.n_robot_contacts` | int32 | as in the frames file |
| `force.gripper_force_mag`, `force.ground_force_mag` | float64 | as in the frames file |

## Episode metadata

Both Parquet files of an episode carry its metadata as JSON under the footer key `physgraspgen`:

| Key | Meaning |
|---|---|
| `format`, `format_version` | `physgraspgen-public`, `2.0` |
| `episode_uid`, `object_type`, `object_asset` | the episode, its object type (the value the diversity item counts) and the object's geometry asset |
| `n_frames`, `n_steps` | rows of the frames file and of the trajectory file |
| `fps`, `recorded_fps` | the published frame rate and the rate the simulator recorded at; `fps` divides `recorded_fps` |
| `channels`, `cameras` | the ground-truth channels and the cameras the episode carries |
| `n_tets` | T |
| `license` | `CC-BY-4.0` |
| `dataset_name`, `dataset_version` | the dataset the episode was written for |
| `label` | the episode's outcome: `success`, `fracture`, `fail` or `unknown` |
| `facts` | per-episode values the diversity tables read, nested by kind: `material`, `placement`, `lighting` and `camera` at least |
| `task` | what benchmark scoring reads besides the trajectory: the task name, the criterion, the planned waypoint phases, the frames each waypoint runs at the recorded rate, why the episode stopped, the placement target, the tolerances of the stage judgments, and the generator's own stage verdicts |
| `header`, `generation_provenance`, `export_provenance` | the generator's episode header and the code and data that generated and exported it |
| `source_n_frames`, `source_cut_at_row` | the generator's frame count and the row the export stopped at because a frame missed a layer (-1 when it did not stop early) |

## Videos

`videos/<camera>/<object_type>/<episode_uid>.mp4` holds the episode's RGB from one camera: one
video frame per published frame, in order, at a constant `fps`. Video frame i is the frame whose
`rgb.<camera>.video_frame` is i. RGB is the only lossy channel; every other channel is stored
exactly.

| Property | Value |
|---|---|
| Container | MP4 |
| Codec | AV1 |
| Pixel format | yuv420p |
| Frame rate | the dataset's `fps`, constant |
| Encoder settings | stated in `metadata.yaml` (`video`) |

## metadata.yaml

| Key | Meaning |
|---|---|
| `format`, `format_version`, `name`, `version` | format and dataset identity |
| `license`, `doi`, `download_url` | CC-BY-4.0, the Zenodo DOI, the download page |
| `created` | when the file was written (UTC) |
| `fps`, `recorded_fps`, `channels`, `cameras` | as in every episode |
| `image_size` | per camera: `height`, `width` |
| `video` | codec, pixel format and encoder settings |
| `counts` | `n_frames`, `n_episodes`, `n_files`, `n_object_types`, `n_object_assets`, `n_channels` |
| `object_types` | per type: `n_episodes`, `n_frames`, `assets` |
| `files` | per file: `path`, `kind` (`frames`, `trajectory` or `video`), `episode_uid`, `object_type`, `sha256`, `bytes`, and `n_frames` for frames files and videos or `n_steps` for trajectories |
| `tools` | per tool file: `path`, `sha256`, `bytes` |
| `conventions` | units, world frame, quaternion order, frame steps, camera model and axes, the deformation gradient's reference configuration, stress, and `segmentation_classes` |
| `provenance` | the distinct generation and export records the episodes carry |

## Conventions

- **Cameras.** `front` is fixed in the scene, `wrist` is mounted on the end effector. A pixel
  (u, v), u the column and v the row, sees the world point X where `[u, v, 1]^T ~ K (R X + t)`
  and `[R t]` is the top of `world_to_camera`: x right, y down, z forward.
- **Segmentation classes**, in value order: 0 `background`, 1 `object`, 2 `arm`, 3 `gripper`,
  4 `support`, 5 `prop`. `metadata.yaml` lists them, and readers look the object's value up by
  the name `object`.
- **Stress** is computed per tetrahedron from the deformed mesh through the object's material
  model. Invalid tetrahedra are NaN, not dropped.
- **Deformation gradient.** `F` is taken from the object's rest mesh turned by its placement yaw,
  so F is the identity at the unloaded start.

## The check

`tools/check_dataset.py` (issue #19) tests a downloaded copy against the four items and needs
Python 3.10 or newer with NumPy, PyArrow and PyYAML. It counts video frames from the MP4 index, so it needs no
video decoder. The acceptance values come from `tools/check_criteria.yaml`, and a run passes only
with the criteria file and the script whose sha256 `metadata.yaml` lists.

## Writing and reading

`pip install "openphysgraspgen[dataset]"` adds PyArrow and PyYAML. Writing RGB also needs an
`ffmpeg` executable with the SVT-AV1 encoder on the `PATH`; reading every other channel
needs only PyArrow.

```python
from openphysgraspgen.dataset import DatasetSpec, GraspDataRecorder, read_frames

spec = DatasetSpec(
    name="physgraspgen-dataset",
    version="1.0.0",
    fps=30,
    recorded_fps=60,
    image_size={"front": (480, 640), "wrist": (480, 640)},
)
recorder = GraspDataRecorder("dataset", spec)
with recorder.episode(
    episode_uid="apple-0001",
    object_type="apple",
    object_asset="apple_01",
    n_tets=n_tets,
    label="success",
    facts=facts,
    task=task,
) as episode:
    for frame in frames:  # published frames: a mapping from column name to value
        episode.add_frame(frame)
    for step in steps:  # every recorded step, for the trajectory file
        episode.add_step(step)
recorder.finalize()  # metadata.yaml and SHA256SUMS, once every episode is written

data = read_frames("dataset/data/apple/apple-0001.parquet", columns=["object.centroid"])
```

A frame gives every column of the frames file except those the writer fills in
(`episode_uid`, `frame_index`, `sim_time_s`, `force.n_contacts`, the video frame
indices and the per-channel steps, which default to the frame's `sim_step`). RGB is
given as `rgb.<camera>.image`, (H, W, 3) uint8, the contacts as
`force.contact_force`, `force.contact_point_world`, `force.contact_particle`,
`force.contact_is_ground` and `force.contact_is_robot`, one row per contact, and the
camera matrices as 3 x 3 and 4 x 4 arrays. A frame or a step that does not follow the
format is refused before anything of it is written, and an episode is written to
`.part` files that are renamed only when every file of the episode is complete.
