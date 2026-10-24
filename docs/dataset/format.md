# The public dataset format

Status: draft for review (format `physgraspgen-public`, version `2.0`).

The project publishes a soft-body grasping dataset on the Hugging Face Hub under CC-BY-4.0, with a
Zenodo DOI. This page defines its files. The `openphysgraspgen.dataset` module writes and reads
them, and the dataset's own check script tests them, so the writer, the reader and the check share
one definition. Version 2.0 replaces an unpublished HDF5 draft (1.0) with Parquet and MP4.

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
  README.md                                          dataset card, with the front matter below
  .gitattributes                                     keeps the hashed files byte for byte in git
  metadata.yaml                                      counts, object types, conventions, every file
  SHA256SUMS                                         sha256 of every data and tool file
  data/<object_type>/<episode_uid>.parquet           frames: every channel except RGB
  trajectories/<object_type>/<episode_uid>.parquet   robot and object state at the recorded rate
  videos/<camera>/<object_type>/<episode_uid>.mp4    RGB of one camera
  tools/check_dataset.py                             the check script (issue #19)
  tools/check_criteria.yaml                          the acceptance values it applies (issue #19)
```

One episode is one frames file, one trajectory file and one video per camera. Files are grouped
by object type, so each type can be exported, uploaded and downloaded on its own. Object types,
episode ids and camera names match `[A-Za-z0-9][A-Za-z0-9._-]*`.

### Dataset card

The frames files and the trajectory files have different columns, and the Hub reads one table per
config, so `README.md` starts with this front matter:

```yaml
---
license: cc-by-4.0
configs:
- config_name: frames
  data_files: data/*/*.parquet
  default: true
- config_name: trajectories
  data_files: trajectories/*/*.parquet
---
```

`openphysgraspgen.dataset.card_front_matter()` returns this text for an exporter to put at the top
of its card. Finalize does not write `README.md`, and does not hash it.

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
| `sim_step` | int64 | simulation step, counted at the recorded rate from the episode start (the first recorded step is 1). Frames are consecutive on the published grid: the first frame's step is a positive multiple of `recorded_fps / fps`, and each later frame's step is the previous one's plus `recorded_fps / fps` |
| `sim_time_s` | float64 | simulation time [s], `sim_step / recorded_fps` |
| `source_frame_index` | int32 | the frame's row in the generator's own record |
| `rgb.<camera>.sim_step` | int64 | step of the video frame |
| `rgb.<camera>.video_frame` | int32 | index of this frame in `videos/<camera>/<object_type>/<episode_uid>.mp4`, equal to `frame_index` |
| `depth.<camera>.sim_step` | int64 | step of the depth image |
| `depth.<camera>.depth_m` | fixed list of H x W float32 | forward (z) depth [m], row-major, row 0 at the top |
| `segmentation.<camera>.sim_step` | int64 | step of the mask |
| `segmentation.<camera>.mask` | fixed list of H x W uint8 | per-pixel class, pixel-aligned with the image |
| `camera.<camera>.intrinsics` | fixed list of 9 float64 | K for the stored image, a 3 x 3 matrix stored row-major |
| `camera.<camera>.world_to_camera` | fixed list of 16 float64 | world-to-camera transform, OpenCV axes, a 4 x 4 matrix stored row-major |
| `force.sim_step` | int64 | step of the force layer |
| `force.n_contacts` | int32 | soft contacts on the object |
| `force.contacts` | list of struct | one entry per contact: `force` (3 float32, N, world frame), `point_world` (3 float32, m), `particle` (int32), `is_ground` (bool), `is_robot` (bool) |
| `force.gripper_net_force`, `force.ground_net_force` | fixed list of 3 float32 | net force on the object from the gripper and from the support [N] |
| `force.gripper_force_mag`, `force.ground_force_mag` | float64 | their magnitudes [N] |
| `stress.sim_step` | int64 | step of the stress layer |
| `stress.von_mises` | list of T float32 | von Mises stress per tetrahedron [Pa], NaN exactly where `stress.valid` is false |
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

A frames file holds exactly the columns that its episode's `channels` and `cameras` call for, and
no other. `episode_uid`, `frame_index`, `sim_step`, `sim_time_s`, `source_frame_index` and every
`robot` and `object` column are always there. A channel's columns are there exactly when `channels`
lists it, a column named with `<camera>` once for each camera of `cameras`. `cameras` lists the
cameras given an image size when the episode carries RGB, depth or segmentation, and is empty
otherwise; the `camera.<camera>` matrices are there for each of them. A trajectory file always has
exactly the columns of its table. `check_dataset.frame_schema(channels, cameras, image_size)` and
`check_dataset.trajectory_schema()`, in `openphysgraspgen.dataset`, build the two Arrow schemas,
column order included.

Image sizes are fixed per camera for the whole dataset and stated in `metadata.yaml` and in every
episode's metadata. The tetrahedron count T is fixed per episode and stated in the episode
metadata. No column holds nulls.

Each fixed list and per-tetrahedron column carries the shape of one row's value in its Arrow field
metadata (`shape`): `[H, W]` for images, `[3, 3]` and `[4, 4]` for the camera matrices, `[n]` for
the other fixed lists, `["n_tets"]` for the per-tetrahedron values and `["n_tets", 3, 3]` for the
deformation gradient. Arrow field metadata lives in the file's `ARROW:schema` footer entry, which
Arrow-based readers decode; other readers see the flat lists. `read_frames` restores the shapes:
images (N, H, W), camera matrices (N, 3, 3) and (N, 4, 4), per-tetrahedron values (N, T) and the
deformation gradient (N, T, 3, 3).

### Values

- Every `<layer>.sim_step` equals the frame's `sim_step`: all the layers of a frame are captured at
  one step.
- Every float value of the `robot`, `object`, `camera`, `force` and `deformation_gradient` columns,
  contact forces and points included, is finite, except `robot.grip_contact_width_m`, which is
  finite or NaN.
- `stress.von_mises` is NaN exactly where `stress.valid` is false, and finite where it is true.
- `camera.<camera>.intrinsics` is a pinhole K: `K[0][1]` (skew) and `K[1][0]` are 0, the last row
  is `0 0 1` and both focal lengths are positive.
- `camera.<camera>.world_to_camera` is a rigid transform: its last row is `0 0 0 1`, and its
  rotation R satisfies `R R^T = I` and `det R = +1` to within 1e-6. A rotation computed in
  float32, or composed of float32 rotations, can miss that, so compute R in float64, for example
  from a normalized quaternion.
- A mask value is the index of a segmentation class, 0 to 5. Depth and `stress.volume_ratio` have
  no rule beyond their type.

## Trajectory file

`trajectories/<object_type>/<episode_uid>.parquet` has one row per recorded simulation step, from
step 1 to the last, consecutive, so it keeps the steps that the published frame rate skips. It
covers every frame's step. It holds what benchmark scoring and control learning read, and no
images.

| Column | Type | Meaning |
|---|---|---|
| `episode_uid` | string | the episode |
| `sim_step`, `sim_time_s` | int64, float64 | as in the frames file |
| `finite` | bool | the simulation's state, the object's and the robot's, is finite at this step |
| `robot.waypoint`, `robot.phase` | int32, string | as in the frames file |
| `robot.ee_pose_cmd`, `robot.ee_pose`, `robot.arm_q_setpoint`, `robot.arm_q` | as in the frames file | |
| `robot.gripper_width_cmd_m`, `robot.gripper_width_setpoint_m`, `robot.gripper_width_m`, `robot.grip_contact_width_m` | float64 | as in the frames file |
| `object.centroid`, `object.bbox_min`, `object.bbox_max` | fixed list of 3 float64 | as in the frames file, NaN exactly where `finite` is false |
| `object.n_robot_contacts` | int32 | as in the frames file |
| `force.gripper_force_mag`, `force.ground_force_mag` | float64 | as in the frames file, both NaN on a step without a force layer |

- `object.centroid`, `object.bbox_min` and `object.bbox_max` are NaN in every element exactly
  where `finite` is false, and finite where it is true. Where `finite` is false, the robot's float
  values may be NaN too.
- `force.gripper_force_mag` and `force.ground_force_mag` are both NaN on a step without a force
  layer, such as the step at which the simulation blew up, and both finite elsewhere.
- `robot.grip_contact_width_m` is finite or NaN, and every other float value is finite.

So the trajectory keeps the last step of a recording that blew up, and NaN marks the values the
simulation did not give there.

## Episode metadata

Both Parquet files of an episode carry the same metadata as JSON under the footer key
`physgraspgen`, with these keys:

| Key | Meaning |
|---|---|
| `format`, `format_version` | `physgraspgen-public`, `2.0` |
| `dataset_name`, `dataset_version` | the dataset the episode was written for |
| `license` | `CC-BY-4.0` |
| `episode_uid`, `object_type`, `object_asset` | the episode, its object type (the value the diversity item counts) and the object's geometry asset |
| `label` | the episode's outcome: `success`, `fracture`, `fail` or `unknown` |
| `n_frames`, `n_steps` | rows of the frames file and of the trajectory file |
| `n_tets` | T |
| `fps`, `recorded_fps` | the published frame rate and the rate the simulator recorded at; `fps` divides `recorded_fps` |
| `channels`, `cameras` | the ground-truth channels and the cameras the episode carries |
| `image_size` | per camera of `cameras`: `height`, `width` |
| `segmentation_classes` | the mask classes in value order, or null without segmentation |
| `video` | the encoder settings (see Videos), or null without RGB |
| `ffmpeg_version` | the first line of `ffmpeg -version` of the build that encoded the videos, or null without RGB |
| `facts` | per-episode values the diversity tables read, nested by kind. `material`, `placement`, `lighting` and `camera` are each a mapping holding at least one value that is not null; other kinds may be added |
| `task` | what benchmark scoring reads besides the trajectory. Its keys are defined by the PG-T1 scorer (issue #20); until then it is free-form, or null |
| `header`, `generation_provenance`, `export_provenance` | the generator's episode header and the records of the code and data that generated and exported it, or null |
| `source_n_frames`, `source_cut_at_row` | the generator's frame count, which `n_steps` equals when the trajectory holds every step the generator recorded, or null when not given; and the row the export stopped at because a frame missed a layer (-1 when it did not stop early) |

Every episode of a dataset must state the same `format`, `format_version`, `dataset_name`,
`dataset_version`, `license`, `fps`, `recorded_fps`, `channels`, `cameras`, `image_size`,
`segmentation_classes` and `video`, which `metadata.yaml` states once. `ffmpeg_version` may differ
between episodes.

## Videos

`videos/<camera>/<object_type>/<episode_uid>.mp4` holds the episode's RGB from one camera: one
video frame per published frame, in order, at a constant `fps`. Video frame i is the episode's
frame i, the row whose `frame_index` and `rgb.<camera>.video_frame` are i. The video's own
timestamps are i / fps from 0, not `sim_time_s`, so a reader looks a frame up by its index, not by
its simulation time. RGB is the only lossy channel; every other channel is stored exactly.

| Property | Value |
|---|---|
| Container | MP4, not fragmented, one video track |
| Codec | AV1, an `av01` sample entry |
| Pixel format | 8-bit 4:2:0 (yuv420p), as the sample entry's `av1C` box states |
| Frame size | the camera's `image_size`, whose height and width are even when the dataset has RGB (4:2:0 halves both for colour) |
| Frame rate | the dataset's `fps`, constant |
| Frames | the episode's `n_frames` |
| Encoder settings | stated in `metadata.yaml` (`video`) |

All of these are read from the file's index, without decoding. The encoder settings are `codec`
(`av1`), `pix_fmt` (`yuv420p`), `encoder` (`libsvtav1`, ffmpeg's SVT-AV1 encoder, the only one
the writer runs), `crf` (1 to 63, 30 by default), `gop` (at least 2, 2 by default) and `preset`
(0 to 11, 8 by default). The writer passes each of them to ffmpeg, and the ranges are those in
which SVT-AV1 uses the value given rather than a default of its own, so `metadata.yaml` states
the settings the videos were encoded with. Other AV1 encoders read these options on other scales
or ignore them.

## metadata.yaml

| Key | Meaning |
|---|---|
| `format`, `format_version`, `name`, `version` | format and dataset identity |
| `license`, `doi`, `download_url` | CC-BY-4.0; the Zenodo DOI as a bare DOI, `10.<registrant>/<suffix>` without `https://doi.org/`; the download page, an `https://` URL. `doi` and `download_url` are null until given |
| `created` | when the file was written (UTC) |
| `fps`, `recorded_fps`, `channels`, `cameras` | as in every episode |
| `image_size` | per camera: `height`, `width` |
| `video` | the encoder settings every episode shares, and `ffmpeg_versions`, the distinct `ffmpeg_version` of the episodes; null without RGB |
| `counts` | `n_frames`, `n_episodes`, `n_files` (the entries of `files`: frames, trajectory and video files, not tool files), `n_object_types`, `n_object_assets`, `n_channels` |
| `object_types` | per type: `n_episodes`, `n_frames`, `assets` |
| `files` | per file: `path`, `kind` (`frames`, `trajectory` or `video`), `episode_uid`, `object_type`, `n_frames` for frames files and videos or `n_steps` for trajectories, `sha256`, `bytes` |
| `tools` | per file under `tools/`: `path`, `sha256`, `bytes` |
| `conventions` | units, world frame, quaternion order, frame steps, the trajectory's rows, the video clock, camera model and axes, images, depth, stress, the deformation gradient's reference configuration, and `segmentation_classes` (null without segmentation) |
| `provenance` | `generation` and `export`: the distinct provenance records the episodes carry |

`SHA256SUMS` lists every entry of `files` and `tools` as `<sha256>  <path>`, in path order, as
`sha256sum -c` reads it.

## Finalizing

`finalize_dataset(root)`, or `GraspDataRecorder.finalize()`, runs once every episode is written.
It writes `metadata.yaml`, `SHA256SUMS` and `.gitattributes`, and writes none of them when it
refuses the dataset. It refuses it when:

- a file under the root is not part of the layout (an interrupted episode leaves `.part` files),
  or an episode's file is missing, or a file belongs to no episode;
- an episode's metadata is not a mapping, is not `physgraspgen-public` 2.0, lacks a key, names
  another episode or object type than its path, has an `object_asset` that is not a nonempty
  string, does not describe its columns, or lacks a kind of `facts`;
- one episode id is under two object types;
- episodes disagree on a key that every episode must state alike (see Episode metadata);
- two object types are one after lower-casing and stripping (`apple`, `Apple`), or one
  `object_asset` is under two object types;
- a Parquet file cannot be read in full: every column of every row group is decoded and every
  page's checksum verified;
- a Parquet file's columns differ from the format's (its Arrow schema, with the field metadata
  that holds each column's shape, the file's own schema metadata aside), its row count differs
  from its metadata, or a trajectory file's metadata differs from its frames file's;
- a video differs from the Videos table, read from its index;
- `doi` or `download_url` is not of the form given in the metadata.yaml table.

It reads each Parquet file in full and parses each video's index, then hashes the file, in
`workers` threads. The `.part` files that a stopped finalize leaves at the root are replaced, not
refused. `.gitattributes` marks `metadata.yaml`, `SHA256SUMS` and `tools/**` as `-text`, so that a
git checkout keeps the bytes that are hashed whatever its line-ending settings, and sends
`*.parquet` and `*.mp4` through Git LFS, as the Hub stores them.

## Conventions

- **World frame.** Right-handed, z up, so gravity points along -z. Positions, poses, forces and
  contact points are in the world frame unless their column says otherwise, and every quantity is
  in SI units: metres, seconds, newtons, pascals and radians.
- **Cameras.** `front` is fixed in the scene, `wrist` is mounted on the end effector. A pixel
  (u, v), u the column and v the row, sees the world point X where `[u, v, 1]^T ~ K (R X + t)`
  and `[R t]` is the top of `world_to_camera`: x right, y down, z forward. Pixel centres are at
  integer coordinates: the top-left pixel's centre is (0, 0).
- **Segmentation classes**, in value order: 0 `background`, 1 `object`, 2 `arm`, 3 `gripper`,
  4 `support`, 5 `prop`. `metadata.yaml` and every episode's metadata list them, and readers look
  the object's value up by the name `object`.
- **Stress** is computed per tetrahedron from the deformed mesh through the object's material
  model. Invalid tetrahedra are NaN, not dropped.
- **Deformation gradient.** `F` is taken from the object's rest mesh turned by its placement yaw,
  so F is the identity at the unloaded start.

## The check

`tools/check_dataset.py` (issue #19) tests a downloaded copy against the four items and needs
Python 3.10 or newer with NumPy, PyArrow and PyYAML. It counts video frames from the MP4 index, so
it needs no video decoder. The acceptance values come from `tools/check_criteria.yaml`, and a run
passes only with the criteria file and the script whose sha256 `metadata.yaml` lists.

## Writing and reading

`pip install "openphysgraspgen[dataset]"` adds PyArrow and PyYAML, which writing and reading both
need. Writing RGB also needs an `ffmpeg` executable on the `PATH` whose `ffmpeg -encoders` lists
`libsvtav1` (SVT-AV1); the recorder checks this when it is created. Reading RGB with `read_video`
needs an ffmpeg with an AV1 decoder, such as libdav1d; reading every other channel needs no ffmpeg.

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
    facts=facts,  # material, placement, lighting and camera, each a mapping
    task=task,
) as episode:
    for frame in frames:  # published frames: a mapping from column name to value
        episode.add_frame(frame)
    for step in steps:  # every recorded step from step 1, for the trajectory file
        episode.add_step(step)
recorder.finalize()  # once every episode is written

data = read_frames("dataset/data/apple/apple-0001.parquet", columns=["object.centroid"])
```

A frame gives every column of the frames file except those the writer fills in (`episode_uid`,
`frame_index`, `sim_time_s`, `force.n_contacts`, the video frame indices and the per-channel steps,
which may be given but must equal the frame's `sim_step`); `source_frame_index` is required. RGB is
given as `rgb.<camera>.image`, (H, W, 3) uint8, the contacts as `force.contact_force`,
`force.contact_point_world`, `force.contact_particle`, `force.contact_is_ground` and
`force.contact_is_robot`, one row per contact, and the camera matrices as 3 x 3 and 4 x 4 arrays. A
step gives every column of the trajectory file except `episode_uid` and `sim_time_s`. Where a value
may be NaN, it may also be given as None. The writer copies what it keeps, so a caller may refill
one set of arrays for every frame and step. A frame or a step that does not follow the format, its
value rules included, is refused before anything of it is written; so is a float64 value too large
for a float32 column.

An episode is written to `.part` files that are renamed only when every file of the episode is
complete, the frames file last. The writer claims the episode by creating its frames `.part` file,
which only one writer can create, so a second writer of the episode is refused, and so is a writer
after one that stopped without removing its `.part` files. `overwrite=True` replaces a written
episode or a leftover `.part` file; a writer whose episode another writer took over that way
raises an error, and neither removes nor publishes the other writer's files. Every Parquet page is
written with a checksum.
