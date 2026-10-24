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
  tools/check_dataset.py                             the check script (see The check)
  tools/check_criteria.yaml                          the acceptance values it applies
```

One episode is one frames file, one trajectory file and one video per camera. Files are grouped
by object type, so each type can be exported, uploaded and downloaded on its own. Object types
and episode ids become path components, so that every file system a copy lands on can hold them:
each matches `[A-Za-z0-9][A-Za-z0-9._-]*`, has at most 100 characters, does not end in a dot, and
is no device name that Windows reserves (`CON`, `PRN`, `AUX`, `NUL`, `COM1` to `COM9`, `LPT1` to
`LPT9`, with or without a suffix). No two paths of a dataset differ only in case. The cameras are
`front` and `wrist`.

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
episode's metadata; an image holds at most 2,147,483,647 pixels, the size of an Arrow list. The
tetrahedron count T is fixed per episode and stated in the episode metadata. No column holds a
null, and neither does any list in a column.

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
- `robot.phase` is a nonempty string, here and in the trajectory.
- A mask value is the index of a segmentation class, 0 to 5. Depth and `stress.volume_ratio` have
  no rule beyond their type.
- A frame's `robot` and `object` columns, and its two force magnitudes, hold the trajectory's
  values at the frame's step, NaN where the trajectory's is NaN. A frame therefore lies on a step
  whose object state is finite.

## Trajectory file

`trajectories/<object_type>/<episode_uid>.parquet` has one row per recorded simulation step,
consecutive from step 1, so it keeps the steps that the published frame rate skips. It covers every
frame's step. It holds what benchmark scoring and control learning read, and no images.

| Column | Type | Meaning |
|---|---|---|
| `episode_uid` | string | the episode |
| `sim_step`, `sim_time_s` | int64, float64 | as in the frames file |
| `finite` | bool | the object's state is finite at this step |
| `robot.waypoint`, `robot.phase` | int32, string | as in the frames file |
| `robot.ee_pose_cmd`, `robot.ee_pose`, `robot.arm_q_setpoint`, `robot.arm_q` | as in the frames file | |
| `robot.gripper_width_cmd_m`, `robot.gripper_width_setpoint_m`, `robot.gripper_width_m`, `robot.grip_contact_width_m` | float64 | as in the frames file |
| `object.centroid`, `object.bbox_min`, `object.bbox_max` | fixed list of 3 float64 | as in the frames file, NaN exactly where `finite` is false |
| `object.n_robot_contacts` | int32 | as in the frames file |
| `force.gripper_force_mag`, `force.ground_force_mag` | float64 | as in the frames file |

- `object.centroid`, `object.bbox_min` and `object.bbox_max` are NaN in every element exactly
  where `finite` is false, and finite where it is true.
- Where `finite` is false, the reached robot state (`robot.ee_pose`, `robot.arm_q`,
  `robot.gripper_width_m`) and the two force magnitudes may also be NaN, because a simulation can
  break down before they are recorded. Commands and setpoints stay finite.
- `robot.grip_contact_width_m` is finite or NaN, and every other float value is finite.

So the step at which a simulation blew up can be written, and the trajectory keeps every
recorded step: `n_steps` equals `source_n_frames` unless the generator lost steps. A step that
breaks these rules cannot be written.

## Episode metadata

Both Parquet files of an episode carry the same metadata, byte for byte, as JSON under the footer
key `physgraspgen`. The JSON is strict: UTF-8, one object, no key repeated in an object, floats
that are finite, integers that fit int64, and objects and arrays nested at most 32 levels deep,
the metadata's own object the first. It holds exactly these keys:

| Key | Meaning |
|---|---|
| `format`, `format_version` | `physgraspgen-public`, `2.0` |
| `dataset_name`, `dataset_version` | the dataset the episode was written for, nonempty strings |
| `license` | `CC-BY-4.0` |
| `episode_uid`, `object_type`, `object_asset` | the episode and its object type, those of its path (the type is the value the diversity item counts), and the object's geometry asset, a nonempty string |
| `label` | the episode's outcome: `success`, `fracture`, `fail` or `unknown` |
| `n_frames`, `n_steps` | rows of the frames file and of the trajectory file, at least 1 |
| `n_tets` | T |
| `fps`, `recorded_fps` | the published frame rate and the rate the simulator recorded at; `fps` divides `recorded_fps` |
| `channels`, `cameras` | the ground-truth channels and the cameras the episode carries, each in the order of the Layout (`rgb`, `depth`, `force`, `stress`, `deformation_gradient`, `segmentation`; `front`, `wrist`); `cameras` is empty exactly when no camera channel is listed |
| `image_size` | per camera of `cameras`: `height`, `width` |
| `segmentation_classes` | the six mask classes in value order (see Conventions), or null without segmentation |
| `video` | the encoder settings (see Videos), or null without RGB |
| `ffmpeg_version` | the first line of `ffmpeg -version` of the build that encoded the videos, or null without RGB |
| `video_sha256` | per camera of `cameras`, the sha256 of its video in lower-case hex, or null without RGB. It ties each video to its episode by content |
| `facts` | per-episode values the diversity tables read, nested by kind. `material`, `placement`, `lighting` and `camera` are each a mapping holding at least one value that is not null; other kinds may be added |
| `task` | what benchmark scoring reads besides the trajectory, a mapping or null. Its keys are defined by the PG-T1 scorer (issue #20); until then it is free-form |
| `header`, `generation_provenance`, `export_provenance` | the generator's episode header and the records of the code and data that generated and exported it, each a mapping or null |
| `source_n_frames`, `source_cut_at_row` | the generator's frame count, at least `n_steps`, which `n_steps` equals when the trajectory holds every step the generator recorded, or null when it is unknown; and the row the export stopped at because a frame missed a layer (-1 when it did not stop early) |

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
| Timing | every frame shown once, in order: the media duration (`mdhd`) is the frames' total, no composition offset (`ctts`) reorders them, and an edit list, when there is one, holds one entry at media time 0 and rate 1 that lasts as long as the frames |
| Content | the sha256 its episode's metadata states (`video_sha256`) |
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
| `license`, `doi`, `download_url` | CC-BY-4.0; the Zenodo DOI as a bare DOI, `10.<registrant>/<suffix>` with a registrant code of 4 to 9 digits, without `https://doi.org/`; the download page, an `https://` URL. `doi` and `download_url` are null until given |
| `created` | when the file was written (UTC) |
| `fps`, `recorded_fps`, `channels`, `cameras` | as in every episode |
| `image_size` | per camera: `height`, `width` |
| `video` | the encoder settings every episode shares, and `ffmpeg_versions`, the distinct `ffmpeg_version` of the episodes; null without RGB |
| `counts` | `n_frames`, `n_episodes`, `n_files` (the entries of `files`: frames, trajectory and video files, not tool files), `n_object_types`, `n_object_assets`, `n_channels` |
| `object_types` | per type: `n_episodes`, `n_frames`, `assets` |
| `files` | per file: `path`, `kind` (`frames`, `trajectory` or `video`), `episode_uid`, `object_type`, `n_frames` for frames files and videos or `n_steps` for trajectories, `sha256`, `bytes` |
| `tools` | `tools/check_dataset.py` and `tools/check_criteria.yaml`: `path`, `sha256`, `bytes` |
| `conventions` | units, world frame, quaternion order, frame steps, the trajectory's rows, the video clock, camera model and axes, images, depth, stress, the deformation gradient's reference configuration, and `segmentation_classes` (null without segmentation) |
| `provenance` | `generation` and `export`: the distinct provenance records the episodes carry |

`SHA256SUMS` lists every entry of `files` and `tools` as `<sha256>  <path>`, in path order, as
`sha256sum -c` reads it.

## Finalizing

`finalize_dataset(root)`, or `GraspDataRecorder.finalize()`, runs once every episode is written.
It writes `tools/check_dataset.py`, `tools/check_criteria.yaml`, `.gitattributes`,
`metadata.yaml` and `SHA256SUMS`, in that order, and writes none of them when it refuses the
dataset. `tools/check_dataset.py` is this package's `check_dataset.py` byte for byte, and
`tools/check_criteria.yaml` the packaged acceptance values, or the file `criteria=` names.

It reads every file with the check's own rules (see The check), in `workers` threads: every
Parquet file in full, with every page's checksum verified and every value rule applied, and every
video's index. Then it hashes each file. It refuses the dataset when:

- the root holds no episode, or a file the layout does not define (an interrupted episode leaves
  `.part` files), or `tools/` holds a file other than the two tool files;
- a file breaks a rule of the check: an episode's metadata (its keys, their values, the identity
  of its path), a file's columns (names, order, types, nullability, field metadata), a value of a
  frame or of a trajectory step, a video's index or its sha256, a file of an episode that is
  missing, a file of no episode, one episode id under two object types, or two paths that differ
  only in case;
- episodes disagree on a key that every episode must state alike (see Episode metadata);
- two object types are one after lower-casing and stripping (`apple`, `Apple`), or one
  `object_asset` is under two object types;
- an episode's `facts` lack a kind the criteria require;
- a folder holds more than the 10,000 files the Hugging Face Hub allows in one;
- the criteria file is not one the check can apply;
- `doi` or `download_url` is not of the form given in the metadata.yaml table.

The tool files and the `.part` files that a stopped finalize leaves are replaced, not refused,
and the `.git/` and `.cache/` folders at the root, which a clone of the Hub's repository and
`hf download --local-dir` create, are skipped. `.gitattributes` marks `metadata.yaml`,
`SHA256SUMS` and `tools/**` as `-text`, so that a git checkout keeps the bytes that are hashed
whatever its line-ending settings, and sends `*.parquet` and `*.mp4` through Git LFS, as the Hub
stores them.

With `check_seed=`, finalize then runs the `tools/check_dataset.py` it wrote on the dataset, with
the running interpreter and that seed, and the metadata it returns holds `check`: the check's exit
status and its `RESULT` payload. A failed check is returned as such, and the check fails until
`doi` and `download_url` are given; a check that could not run raises `DatasetError`. That the
script needs nothing but NumPy, PyArrow and PyYAML is shown by the tests, which run it with
nothing else importable.

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

`tools/check_dataset.py` tests a dataset against the four items of the dataset test, with the
acceptance values of `tools/check_criteria.yaml`. It is this package's
`openphysgraspgen/dataset/check_dataset.py`, the same file that defines the format for the writer
and finalize, and it needs only Python 3.10 or newer with NumPy 1.26, PyArrow 17 and PyYAML 6 or
newer: nothing from this package, and no video decoder.

### Running it

From the dataset's root, the folder that holds `metadata.yaml`:

```
python -I tools/check_dataset.py . --seed <the tester's seed>
python -I tools/check_dataset.py . --seed 20261116 --json report.json --workers 8
```

`-I` keeps the interpreter from importing from the working directory or `PYTHONPATH`, and the
script drops its own folder from the path, so no file beside it in `tools/` can stand in for a
module it imports. `-I` also leaves out packages installed with `pip install --user`: install
NumPy, PyArrow and PyYAML in a virtual environment or in the interpreter's own site-packages, or
run the script without `-I`, since it drops its own folder from the path either way.

| Option | Meaning |
|---|---|
| `--seed N` | the tester's seed, which draws the consistency samples; required when that item runs |
| `--json PATH` | also write the whole report as JSON |
| `--workers N` | threads that hash and read the files (default: the CPU count, at most 8) |
| `--only ITEMS` | comma-separated items to run, from `scale,metadata,consistency,diversity` |
| `--criteria PATH` | other acceptance values; a run passes only with the published file |
| `--sample N` | frames in each consistency sample; one smaller than the criteria's passes only when it holds every candidate frame |
| `--no-checksums` | read every file without comparing its sha256; the metadata item cannot pass |

### What it checks

Every file under `data/`, `trajectories/` and `videos/` is hashed before any is opened, and only
a file whose sha256 is the one `metadata.yaml` states is read: a damaged or partial download is
reported as such, never read as if it were the published file. A file read is read in full:
every page of a Parquet file is decoded and its checksum verified, and every rule on this page is
applied to its footer, its columns and every value of every frame and step. A video is read from
its MP4 index alone. A file that changes while the check runs is reported. Files of another
suffix or with a hidden name in the data folders, and files in `tools/` other than the two tool
files, are ignored and listed in the report.

| Item | Passes when | Published values |
|---|---|---|
| scale | no data file breaks a rule, and the frames of the episodes whose files are all clean reach the minimum | `min_frames` 1,000,000 |
| metadata | nothing `metadata.yaml` states disagrees with the files: counts, object types, channels, conventions, each episode's shared values, each file's kind, size, count, identity and sha256, the tools and `SHA256SUMS`. And the specification holds: the required channels and cameras in every episode, the licence, a DOI and an `https://` download URL | `max_mismatches` 0; the six channels, `front` and `wrist`, `CC-BY-4.0` |
| consistency | every frame's layers carry the frame's own step and its videos hold it at its index (the step part), and on a sample drawn with the seed from the frames whose contacts press on the robot, those contact points projected through each camera land within a tolerance of the object's mask (the projection) | `min_pass_fraction` 1.0 for both parts, of the layers `rgb`, `depth`, `force`, `stress`; a sample of 1,000 from at least 1,000 such frames; 4 px |
| diversity | the object types reach the minimum, and no frames file is unread or untyped, no two types are one in lower case, no asset or episode is under two types, and every episode's `facts` hold the required kinds; a table per variable of the facts, the types and the labels is reported | `min_object_types` 10; `material`, `placement`, `lighting`, `camera` |

The samples: a frame's key is the first 8 bytes of the sha256 of `<seed> NUL <purpose> NUL
<path> NUL <frame>`, and the frames of the smallest keys are the sample, so one seed draws the
same frames on any machine, whatever order the files are read in. The projection takes each
sampled frame's contacts that press on the robot (`is_robot`, and a force that is not 0), its
camera matrices and its masks, read by the class name `object`: a camera passes when its points
in view lie within `projection_radius_px` of an object pixel, a frame when every camera that sees
a point passes, and every camera must see a point in some sampled frame. The 4 px tolerance is a
placeholder until it is measured on format 2.0 data and agreed with the test lab; the criteria
file states its provenance, and the report repeats it.

A run passes only with the script and the criteria file whose sha256 `metadata.yaml` lists: the
digests of the running script and of the criteria bytes the run applied are compared, so weaker
criteria or an edited script fail the run even when every item passes. `metadata.yaml` itself is
not hashed by the dataset; the report prints its sha256 to compare with the one a DOI record
publishes. The specification fails until the dataset is published, as finalize writes `doi` and
`download_url` as null until they are given.

### What it prints

A table, the listings of what failed (the first 50 of each; the JSON report holds them all), the
diversity tables, and a last line that a program reads:

```
physgraspgen-public 2.0 dataset check
root: /data/physgraspgen-dataset
metadata.yaml: sha256 8ed4b2834fb12ee20394237981321dded26c02b3c05df04276403835cb553879
criteria: /data/physgraspgen-dataset/tools/check_criteria.yaml (sha256 693e6253e768b0c8; the published one: yes)
script: /data/physgraspgen-dataset/tools/check_dataset.py (sha256 c66e0e133da35f5b; the published one: yes)
items run: scale, metadata, consistency (not run: diversity)

Item                  Measured                                                             Criterion                      Result
scale                 10 frames in 4 episodes (16 files)                                   >= 10 frames                   PASS
metadata              0 mismatches in 184 comparisons                                      <= 0 mismatches, and spec      FAIL
  spec (in metadata)  6 of 6 GT channels, cameras front, wrist; failed: doi, download_url  6 channels, front, wrist, ...  FAIL
consistency           steps 10/10 frames (100.00 %), sample 5/5; projection 5/5 of ...     100 % of frames; ...           PASS

Overall: FAIL for the items run; not run: diversity
RESULT {"pass": false, "items_run": ["scale", "metadata", "consistency"], "criteria": {...}, ...}
```

The `RESULT` line holds the verdict, the items run, the criteria's and the script's sha256 and
whether each is the published one, the sha256 of `metadata.yaml`, and each item's headline
numbers. Progress lines (`hashing:`, `reading:`, `projecting:`) and warnings go to standard error.
The JSON report is strict JSON: a value JSON cannot hold, such as a NaN in `metadata.yaml`, is
written as a string.

| Exit status | Meaning |
|---|---|
| 0 | every item run passed, with the published criteria and script |
| 1 | the run completed and failed: an item failed, the criteria or the script are not the published ones, or `--no-checksums` was given with the metadata item |
| 2 | the check could not run, or its report could not be written: a usage error, a missing or invalid criteria file or `metadata.yaml`, criteria values an item cannot apply, a dependency or Python too old, an interrupt, or a machine out of memory or file descriptors. The `RESULT` line gives the reason |

Hashing reads every byte once. Reading decodes every column, about 1.6 ms a frame at 480 x 640
with two cameras and 3,000 tetrahedra on one thread, or half an hour per million frames, which
`--workers` threads share. A file is decoded about 16 MiB of values at a time, whatever its row
groups, so at that size a run holds about 0.5 GB with one thread and 2 GB with eight.

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
    source_n_frames=n_recorded,  # the generator's frame count
    task=task,
) as episode:
    for frame in frames:  # published frames: a mapping from column name to value
        episode.add_frame(frame)
    for step in steps:  # every recorded step from step 1, for the trajectory file
        episode.add_step(step)
recorder.finalize()  # once every episode is written: tools/, metadata.yaml, SHA256SUMS

data = read_frames("dataset/data/apple/apple-0001.parquet", columns=["object.centroid"])
```

A frame gives every column of the frames file except those the writer fills in (`episode_uid`,
`frame_index`, `sim_time_s`, `force.n_contacts`, the video frame indices and the per-channel steps,
which may be given but must equal the frame's `sim_step`); `source_frame_index` is required. So is
an episode's `source_n_frames`, which is None only when the generator's frame count is unknown. RGB
is given as `rgb.<camera>.image`, (H, W, 3) uint8, the contacts as `force.contact_force`,
`force.contact_point_world`, `force.contact_particle`, `force.contact_is_ground` and
`force.contact_is_robot`, one row per contact, and the camera matrices as 3 x 3 and 4 x 4 arrays. A
step gives every column of the trajectory file except `episode_uid` and `sim_time_s`. In the `robot`
and `object` columns, a value that may be NaN may also be given as None. The writer copies what it
keeps, so a caller may refill one set of arrays for every frame and step. A frame or a step that
does not follow the format, its value rules included, is refused before anything of it is written;
so is a float64 value too large for a float32 column.

An episode is written to `.part` files that are renamed only when every file of the episode is
complete, the frames file last. The writer claims the episode by creating its frames `.part` file,
which only one writer can create, so a second writer of the episode is refused, and so is a writer
after one that stopped without removing its `.part` files. `overwrite=True` replaces a written
episode or a leftover `.part` file; a writer whose episode another writer took over that way
raises an error, and neither removes nor publishes the other writer's files. Every Parquet page is
written with a checksum.
