#!/usr/bin/env python3
"""Upload a local ``colon_a2.sdata.zarr`` folder to ``Stellaromics/demo`` on the Hub.

Uses ``HfApi.upload_large_folder`` when available (see issue milume#134); on
``huggingface_hub`` 2.x the same workflow is ``upload_folder`` / ``hf upload``.
Run on Clarence's Mac after ``build_colon_a2.py --reshard-in-place``.

    uv run python scripts/upload_colon_a2_hf.py \\
        ~/scverse_conference_2026/pyxa_scverse_demo/data/colon_a2.sdata.zarr
"""

from __future__ import annotations

import argparse
from pathlib import Path

HF_REPO = "Stellaromics/demo"
DEFAULT_PATH_IN_REPO = "colon_a2.sdata.zarr"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="local colon_a2.sdata.zarr directory")
    parser.add_argument("--repo-id", default=HF_REPO)
    parser.add_argument("--path-in-repo", default=DEFAULT_PATH_IN_REPO)
    parser.add_argument("--repo-type", default="dataset")
    args = parser.parse_args()
    folder = args.folder.expanduser().resolve()
    if not folder.is_dir():
        raise SystemExit(f"not a directory: {folder}")

    from huggingface_hub import HfApi

    api = HfApi()
    upload = getattr(api, "upload_large_folder", None) or api.upload_folder
    upload(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        folder_path=folder,
        path_in_repo=args.path_in_repo,
    )
    print(f"uploaded {folder} -> {args.repo_id} ({args.path_in_repo})")


if __name__ == "__main__":
    main()
