# Governance

OpenPhysGraspGen is maintained by SpaceAI (스페이스에이아이) and developed in the
open under the Apache License 2.0. This document describes who makes decisions,
how, and how that will change as the community grows.

## Roles

| Role | Who | Rights and responsibilities |
|---|---|---|
| User | Anyone | Uses the project, asks questions, reports bugs |
| Contributor | Anyone with a merged pull request, triaged issue, or answered discussion | Proposes changes through issues and pull requests |
| Maintainer | Listed below | Reviews and merges pull requests, triages issues and discussions, moderates under the [Code of Conduct](CODE_OF_CONDUCT.md), prepares releases |
| Project lead | Listed below | Final say when maintainers cannot agree, approves each public release and changes to the open-release scope |

### Current maintainers

| Name | GitHub | Area |
|---|---|---|
| 신동현 (Donghyun Shin) | [@oollccddss](https://github.com/oollccddss) | Project lead; releases, packaging, SBOM, dataset |
| 강민성 | [@kms8720](https://github.com/kms8720) | Repository operations, documentation, CI, community |
| Jan Wissing | [@wissingj](https://github.com/wissingj) | Code review |
| 송규정 | [@songgyujeong](https://github.com/songgyujeong) | Code review |

The project lead is **신동현 (@oollccddss)**.

Maintainers can be reached privately at **oss.spaceai@gmail.com** for conduct
reports and security issues.

## Decision making

- **Everyday changes** (bug fixes, docs, tests, small features) are decided in the
  pull request: the required checks pass and one maintainer other than the author
  approves. The `main` branch enforces both, for administrators too.
- **Larger changes** (new modules, public API changes, new runtime dependencies,
  dropping a supported Python version) start as an issue labelled `proposal`.
  The proposal stays open for at least 5 business days. If no maintainer objects,
  it is accepted (lazy consensus). If there is an objection that cannot be
  resolved, the project lead decides and records the reason in the issue. For a
  time-critical change the project lead can close the proposal period early and
  records why.
- Technical decisions about this repository that are made elsewhere (meetings,
  chat) are summarized in the relevant issue or discussion; one that is not
  recorded there is not final. Business, customer and commercial-product
  decisions are outside the scope of this document.

## What is published

PhysGraspGen is developed as open core. The boundary is:

| Published here (Apache-2.0) | Published on Hugging Face (CC-BY-NC-4.0) | Not published |
|---|---|---|
| Constitutive laws, contact sensors, data recorder, calibration and transfer pipeline, benchmarks, examples, documentation | Grasp dataset and its dataset card | High-precision solver core; raw measurement data, material cards and tuning parameters; the production data pipeline; customer data and customer-specific code; internal infrastructure |

Before any code, data or model leaves the private repository, a maintainer fills
in the **OSS evaluation** issue form (gate: *publish our OSS*) and confirms that:

1. SpaceAI owns, or has permission to publish, everything in the change.
2. It contains no secrets, personal data, customer information or trade secrets.
3. `LICENSE` and any required `NOTICE` are present and correct.
4. There are no known critical vulnerabilities or unsafe defaults.
5. Nothing in the change is the subject of a planned patent application that has
   not been filed yet, unless the project lead approves publishing it; the
   publication date is recorded in the issue.
6. The project lead has approved the publication.

## Releases

1. A maintainer opens a pull request that sets the version and updates the
   roadmap, and the dataset card if the dataset changes. It is reviewed like any
   other change.
2. After it is merged, a maintainer tags the merge commit (`v1.0.0rc1`, …). The
   release workflow builds the sdist and wheel, generates SPDX and CycloneDX SBOMs,
   scans them for known vulnerabilities, and waits for approval of the `pypi`
   environment.
3. The project lead approves that deployment. The workflow then publishes to PyPI
   and creates the GitHub release with the SBOMs, SHA-256 checksums and build
   provenance attestations.
4. The release is announced in Discussions → Announcements.

Step-by-step instructions are in [RELEASING.md](RELEASING.md).

Versioning follows [Semantic Versioning](https://semver.org/) with PEP 440
pre-release suffixes (`0.1.0b0`, `1.0.0rc1`).

## Dependencies

Every pull request that adds or changes a dependency runs the dependency review
check. It passes only for licenses in the allowed class listed in
[`.github/dependency-review-config.yml`](.github/dependency-review-config.yml) and
fails on high or critical vulnerabilities. A dependency under any other license
needs an approved **OSS evaluation** issue (license, maintenance, security record,
alternatives) and is then listed there as an exception.

## Becoming a maintainer

A contributor with sustained, high-quality contributions over at least three
months (code, reviews, triage or documentation) can be nominated by a maintainer.
Nomination is an issue; existing maintainers have 5 business days to comment, and
the project lead then decides on the appointment. Maintainers who are inactive
for six months move to emeritus status and can return on request.

## How governance will evolve

The project may move through these stages; the project lead decides when a move
is made, through a change to this document.

| Stage | When | Model |
|---|---|---|
| 1. Project lead | Now | SpaceAI maintainers; project lead has the final say |
| 2. Core team | After `1.0.0`, once there are regular outside contributors | Outside maintainers may join; decisions by maintainer vote |
| 3. Steering group | When multiple organizations depend on the project | A steering group with members from several organizations may set the direction of this repository |

At every stage, governance covers this repository. SpaceAI keeps the project's
names, the commercial solver core and its own products under its own decisions.

## Trademarks

SpaceAI, PhysGraspGen and OpenPhysGraspGen are names of SpaceAI. The Apache
License does not grant permission to use them (section 6). Forks and derived
distributions should use a different name.

## Changing this document

Changes to this document follow the larger-change process above.
