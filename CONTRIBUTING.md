# Contributing to makewfs

`makewfs` is a numerical optics package. A contribution is complete only when
its behavior, units, tests, documentation, and performance implications are
clear.

Read [AGENTS.md](AGENTS.md) and the relevant section of [ROADMAP.md](ROADMAP.md)
before starting. Keep atmosphere physics in `pyturb` and detector physics in
`getframes`.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,examples]"
```

For local sibling development, install the checked-out packages separately:

```bash
python -m pip install -e ../getframes
python -m pip install -e ../pyturb
```

## Quality gate

```bash
ruff check .
ruff format --check .
mypy
python -m pytest -q --cov=makewfs --cov-branch --cov-report=term-missing
mkdocs build --strict
python -m build
```

Physics changes require an analytic or independent-reference assertion. Keep
randomness on explicit seeded generators, state all array units, and update the
configuration reference and changelog for public changes.

## Pull requests

Keep changes focused. Describe the numerical model, source/reference used for
validation, tests run, benchmark impact, and any conditional upstream work.
