# Release process

`makewfs` publishes distributions with PyPI Trusted Publishing. The repository
workflow is `.github/workflows/release.yml`, and its production job uses the
GitHub environment named `pypi`.

## Prepare

1. Set the release version in `src/makewfs/__about__.py`, `CITATION.cff`, and the
   changelog.
2. Run the complete quality gate documented in `CONTRIBUTING.md`.
3. Build the sdist and wheel, run `twine check dist/*`, and install the wheel in
   a clean environment.
4. Commit and push the release-ready source to `main`.

## Publish

Create a GitHub release with a semantic-version tag such as `1.0.0`. Publishing
the release starts the `Release` workflow, which rebuilds the distributions,
checks their metadata, uploads them as a workflow artifact, and publishes them
to PyPI through the `pypi` environment.

A manual `workflow_dispatch` publishes to TestPyPI through the separate
`testpypi` environment. It never publishes to production PyPI.

After publication, verify the PyPI project page and install the exact version
from PyPI in a new environment. Confirm `makewfs --version` and run one minimal
configuration through `WavefrontSensor.expose`.
