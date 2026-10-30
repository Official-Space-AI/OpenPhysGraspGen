# Open source scope and commercial offerings

OpenPhysGraspGen is the open part of SpaceAI's PhysGraspGen project, which is
developed as open core. This page says what is open, what SpaceAI offers
commercially, how we decide on which side a new part belongs, and what we commit to
for the open side. [GOVERNANCE.md](../../GOVERNANCE.md) says who approves a change to
this scope.

## What is open

| Part | Contents | License |
|---|---|---|
| The `openphysgraspgen` package (this repository) | Constitutive laws, contact sensors, the grasp data recorder and reader, the dataset check and the PG-T1 scorer (with `1.0.0rc1`), examples and documentation | Apache-2.0 |
| The grasp dataset ([Hugging Face Hub](https://huggingface.co/datasets/Official-Space-AI/physgraspgen-dataset), [DOI 10.5281/zenodo.23787395](https://doi.org/10.5281/zenodo.23787395)) | Deformable-object grasps with physical ground truth, described in its dataset card. Release `1.0.0rc1` holds 933,896 frames of five object types; later releases add episodes and object types | CC-BY-NC-4.0 |
| Benchmarks | PG-T1 with `1.0.0rc1`, PG-T2 and PG-T3 later | Apache-2.0 |
| The container image (with each release) | The package on the official Python image | Apache-2.0. The base image's packages keep their own licenses |

With these alone, anyone can read and check the dataset and reproduce the published
benchmark numbers from it.

## What is commercial

SpaceAI develops the following outside this repository, to be offered under
commercial terms.

- A high-precision solver for production use, under a commercial license.
- Simulation assets with industry-specific material data calibrated against
  measurements.
- Robot learning data generated for a customer's own objects.

Commercial support for these offerings is planned. [SUPPORT.md](../../SUPPORT.md)
describes it next to the community support for the open parts, which
[COMMUNITY.md](../../COMMUNITY.md) details.

## Where the line is

1. Everything needed to write, read and check the public dataset and to reproduce the
   public benchmarks is open.
2. Accuracy for production use, calibrated industrial material data, customer-specific
   data and integration, and hosted services are commercial.
3. When a new part could go either way, the project lead decides, as
   [GOVERNANCE.md](../../GOVERNANCE.md) describes. As a guide, we ask who needs it
   most: individual researchers and developers point to open, a company running a
   production line points to commercial.

## What we commit to

1. Each dataset release comes with the tools needed to write, read and check it and to
   reproduce its published benchmark numbers, under Apache-2.0.
2. Released versions keep the licenses they were released under.
3. This page lists the commercial offerings, and the release notes announce any
   change to the line between open and commercial.

If rights, privacy, legal or security problems force us to take something down or to
withdraw a release, we remove only what is affected and say so in this repository.

This page is kept up to date as the offerings and the line change. Changes to it go
through a pull request, as [GOVERNANCE.md](../../GOVERNANCE.md) describes.

Contributions are accepted under the Developer Certificate of Origin and the
project's license, as [CONTRIBUTING.md](../../CONTRIBUTING.md) describes.
