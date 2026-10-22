# AGENTS.md

Guidance for agents working in **pyxa_scverse_demo**: marimo notebooks for the Stellaromics workshop (Pyxa, SpatialData, Milume). Rules that can be checked run in CI ([`.github/workflows/checks.yml`](.github/workflows/checks.yml)); the rest are below.

| Rule | Enforced by |
| ---- | ----------- |
| milume is pinned to an exact release (`"milume==X.Y.Z"`) in every notebook header and `pyproject.toml`; no ranges, bare `"milume"`, or preview wheels (`milume @ <url>`) | `scripts/check-milume-pins.sh` (`milume-pins` job) |
| Every notebook parses as a marimo app | `uvx marimo@0.25.0 check *.py` (`marimo-check` job) |
| `3_colon_3d_milume.py` runs every cell, with and without drawn landmarks | `tests/smoke_colon_a2.py` on synthetic data from `annotations/` (`smoke-colon` job) |

## Notebooks

| Notebook | What it is | Data |
| --- | --- | --- |
| `1_intro_marimo.py` | marimo basics: reactive cells, UI elements, markdown | none |
| `2_pyxa_spatialdata.py` | Read Pyxa output into SpatialData with the pyxa reader and explore it in Milume | Pyxa `small` crop (mouse brain) from HF `Stellaromics/demo` |
| `3_colon_3d_milume.py` | 3D spatial analysis of colorectal cancer; vignettes read landmarks drawn in Milume | Glasgow colon A2 store built by `build_colon_a2.py` (not committed) |

## Data

- Colon A2 and Pyxa `small`/`xsmall` are different tissues (human colon vs mouse brain). Never join cell IDs, annotations or coordinates across them (ckmah/milume#97).
- Don't build the full colon store on cloud VMs: it downloads ~22.5 GB and peaks at ~30 GB of memory. Verify `3_colon_3d_milume.py` with `tests/smoke_colon_a2.py` instead, which CI also runs:
  `uv export --script 3_colon_3d_milume.py --no-hashes > /tmp/reqs.txt && uv run --no-project --with-requirements /tmp/reqs.txt python tests/smoke_colon_a2.py`
- Reuse data you already downloaded; don't re-download.

## Before opening a PR

- Check for an open PR on the same issue or files first: `gh pr list -R stellaromics/pyxa_scverse_demo --state open --search "<issue number or topic>"`. Two bots opened the same change before (#6 / #7, #22). With a milume checkout, `node <milume>/frontend/scripts/check-pr-hygiene.mjs overlap --repo stellaromics/pyxa_scverse_demo --branch <branch>` lists overlapping open PRs.
- Branch on the `ckmah/pyxa_scverse_demo` fork and open the PR into `stellaromics/pyxa_scverse_demo` `main`. PRs land as merge commits.
- A notebook change isn't done until it runs on molab. Open `https://molab.marimo.io/github/ckmah/pyxa_scverse_demo/blob/<branch>/<notebook>.py` in a fresh session, use **Run on Server**, and confirm every cell runs and the Milume widget renders. Put the molab link and the result in the PR body.
- A milume version bump changes every header and `pyproject.toml` together, and updates `uv.lock`.
