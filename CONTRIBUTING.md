# Contributing to OpenPhysGraspGen

Thank you for your interest. This guide explains how a change goes from an idea to
a release. It applies to maintainers as well as outside contributors.

## Before you start

- **Questions** go to [Discussions → Q&A](https://github.com/Official-Space-AI/OpenPhysGraspGen/discussions/categories/q-a),
  not to issues.
- **Bugs** go to [Issues](https://github.com/Official-Space-AI/OpenPhysGraspGen/issues/new/choose)
  with the bug report form. Include the version, Python version, OS, and a minimal
  reproduction.
- **New features or larger changes** start as an issue (or an Ideas discussion)
  so that the scope is agreed before code is written. Unlabelled large pull
  requests may be closed with a request to open an issue first.
- **Security problems** must not be reported publicly. Follow [SECURITY.md](SECURITY.md).

Good places to start are issues labelled
[`good first issue`](https://github.com/Official-Space-AI/OpenPhysGraspGen/labels/good%20first%20issue)
and [`help wanted`](https://github.com/Official-Space-AI/OpenPhysGraspGen/labels/help%20wanted).
Comment on the issue before you start so that work is not duplicated. If an
assigned issue has no update for 7 days, a maintainer may unassign it.

## Workflow

1. Fork the repository and create a branch from `main`.
   Name it `<type>/<short-topic>`, for example `fix/ogden-zero-alpha`.
2. Set up a development environment:

   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -e ".[dev]"
   ```

3. Make your change with tests. Keep a pull request to one topic; under about
   400 changed lines is easiest to review.
4. Run the same checks as CI:

   ```bash
   ruff check .
   ruff format --check .
   pytest
   ```

5. Open a pull request against `main` and fill in the template. Link the issue
   with `Closes #<number>`.

## Commit messages and pull request titles

Use [Conventional Commits](https://www.conventionalcommits.org/):
`feat:`, `fix:`, `docs:`, `test:`, `ci:`, `chore:`, `refactor:`.
The pull request title becomes the merge commit subject and is used to write
release notes.

## Review

- Every pull request needs the required checks to pass (lint, tests on Python
  3.10–3.14, dependency review, the wheel build, CodeQL for Python and for the
  workflows) and one approving review from a maintainer other than the author (see
  [GOVERNANCE.md](GOVERNANCE.md)). Branch protection on `main` enforces both.
- Reviews are requested automatically from the maintainers who look after the files
  a pull request changes, as listed in [`.github/CODEOWNERS`](.github/CODEOWNERS).
- Maintainers aim to give a first response within **48 hours**.
  This is a goal, not a guarantee.
- Reviewers focus on correctness, tests, physical units and numerical edge cases,
  and public API changes. Style is left to `ruff`.
- Pull requests are merged with a merge commit so that the history of a change
  stays visible.

## Adding a dependency

New dependencies are reviewed before they are added. Open an issue with the
**OSS evaluation** form and describe the license, maintenance status and why the
dependency is needed. The dependency review check on pull requests passes only
for licenses in the allowed class listed in
[`.github/dependency-review-config.yml`](.github/dependency-review-config.yml); a
dependency under any other license is added there as an exception once its
evaluation is approved. The license classes are in
[docs/oss/license-policy.md](docs/oss/license-policy.md), the evaluation steps in
[docs/oss/evaluation.md](docs/oss/evaluation.md), and what we use today in
[docs/oss/inventory.md](docs/oss/inventory.md).

## License and sign-off

By submitting a pull request you agree that your contribution is licensed under
the [Apache License 2.0](LICENSE) (section 5, "Submission of Contributions").

Sign off every commit with `git commit -s`. The `Signed-off-by` line certifies the
[Developer Certificate of Origin 1.1](https://developercertificate.org/): that you
wrote the change or otherwise have the right to submit it under the project's
license. Pull requests from outside contributors are merged only when every
commit is signed off.
