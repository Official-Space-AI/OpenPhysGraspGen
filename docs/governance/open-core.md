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
| The grasp dataset (Hugging Face Hub, with `1.0.0rc1`) | One million frames of deformable-object grasps with physical ground truth, described in its dataset card | CC-BY-4.0 |
| Benchmarks | PG-T1 with `1.0.0rc1`, PG-T2 and PG-T3 later | Apache-2.0 |
| The container image (with each release) | The package on the official Python image | Apache-2.0. The base image's packages keep their own licenses |

With these alone, anyone can read and check the dataset and reproduce the published
benchmark numbers from it.

## What is commercial

SpaceAI offers the following under commercial terms. They are developed outside this
repository.

- A high-precision solver for production use, under a commercial license.
- Simulation assets with industry-specific material data calibrated against
  measurements.
- Robot learning data generated for a customer's own objects.
- A hosted robot learning platform (planned).

Customers of these offerings get commercial support. Community support for the open
parts is described in [COMMUNITY.md](../../COMMUNITY.md).

## Where the line is

1. Everything needed to write, read and check the public dataset and to reproduce the
   public benchmarks is open.
2. Accuracy for production use, calibrated industrial material data, customer-specific
   data and integration, and hosted services are commercial.
3. When a new part could go either way, we ask who needs it most. If it is
   individual researchers and developers, it is open; if it is a company running a
   production line, it is commercial.

## What we commit to

1. What is needed to write, read and check the public dataset and to reproduce the
   public benchmarks stays open source, with no limits on use, time or scale.
2. Released versions keep their licenses: Apache-2.0 for the code and CC-BY-4.0 for
   the dataset.
3. This page lists the commercial offerings, and the release notes announce any
   change to the line between open and commercial.

If rights, privacy, legal or security problems force us to take something down or to
withdraw a release, we remove only what is affected and say so in this repository.

Contributions are accepted under the Developer Certificate of Origin and the
project's license, as [CONTRIBUTING.md](../../CONTRIBUTING.md) describes.
