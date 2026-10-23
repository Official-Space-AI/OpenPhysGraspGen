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
5. publishes the sdist and wheel to PyPI,
6. attests the build provenance and the SBOM, and creates a GitHub release with the sdist,
   wheel, both SBOMs and `SHA256SUMS` attached. Tags ending in `aN`, `bN` or `rcN` are marked
   as pre-releases.

## Cutting a release

1. Open a pull request that sets `version` in `pyproject.toml` (for example `1.0.0rc1`) and
   updates the roadmap in the README if needed. Merge it after review.
2. Tag the merge commit and push the tag:

   ```sh
   git switch main && git pull
   git tag v1.0.0rc1
   git push origin v1.0.0rc1
   ```

3. Approve the `pypi` environment deployment if the run asks for it, and check that the
   release page has the five assets.

## Verifying a release

```sh
gh attestation verify openphysgraspgen-1.0.0rc1-py3-none-any.whl --repo Official-Space-AI/OpenPhysGraspGen
```

## One-time setup

- On PyPI, add a trusted publisher for the project `openphysgraspgen`: owner
  `Official-Space-AI`, repository `OpenPhysGraspGen`, workflow `release.yml`, environment
  `pypi`. Before the first upload this is a "pending publisher" under the account's
  publishing settings.
- In the repository settings, create the environment `pypi`. Adding required reviewers to it
  makes every upload wait for a maintainer's approval.
