# Releasing

Releases are cut by pushing a version tag. The `Release` workflow
(`.github/workflows/release.yml`) then:

1. checks that the tag matches `version` in `pyproject.toml`,
2. builds the sdist and wheel and checks their metadata,
3. installs the wheel into a clean environment and generates SPDX and CycloneDX SBOMs of
   that environment, so the SBOMs list the package as a user installs it together with the
   dependencies it resolves,
4. fails if the SBOMs miss the package or a dependency, or if a known vulnerability of
   severity high or above is found,
5. publishes the sdist and wheel to PyPI as `openphysgraspgen`,
6. attests the build provenance and the SBOM, and creates a GitHub release with the sdist,
   wheel, both SBOMs and `SHA256SUMS` attached. Tags ending in `aN`, `bN` or `rcN` are marked
   as pre-releases.

The PyPI project is `openphysgraspgen`, the repository's name. It is not `physgraspgen`,
because that is the import name of the project's internal solver package (#28).

## Cutting a release

1. Open a pull request that sets `version` in `pyproject.toml` (for example `1.0.0rc1`) and
   updates the roadmap in the README if needed. Merge it after review.
2. Tag the merge commit and push the tag:

   ```sh
   git switch main && git pull
   git tag v1.0.0rc1
   git push origin v1.0.0rc1
   ```

3. Approve the `pypi` environment deployment when the run asks for it, and check that the
   release page has the five assets.

## Verifying a release

```sh
gh attestation verify physgraspgen-1.0.0rc1-py3-none-any.whl --repo Official-Space-AI/OpenPhysGraspGen
```

## Keeping the workflow current

Third-party actions are pinned to full commit SHAs, with the release tag in a comment.
Dependabot (`.github/dependabot.yml`) opens a pull request each week when an action or a
Python dependency has a new release; review it like any other change.
`pypa/gh-action-pypi-publish` must stay at v1.14.3 or later (GHSA-5xjx-3w67-w9x9).

## One-time setup

- On PyPI, add a trusted publisher for the project `openphysgraspgen`: owner
  `Official-Space-AI`, repository `OpenPhysGraspGen`, workflow `release.yml`, environment
  `pypi`. Before the first upload this is a "pending publisher" under the account's
  publishing settings.
- The repository has an environment `pypi` that only `v*` tags can deploy to, with a required
  reviewer, so every upload waits for a maintainer's approval.
