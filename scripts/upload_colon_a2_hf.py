#!/usr/bin/env python3
"""Upload a local ``colon_a2.sdata.zarr`` folder to ``Stellaromics/demo`` on the Hub.

``huggingface_hub`` 2.x exposes :meth:`huggingface_hub.HfApi.upload_folder` (and ``hf
upload``) with ``path_in_repo``. Older releases had ``upload_large_folder`` / ``hf
upload-large-folder`` without ``path_in_repo``; this script stages under a temp parent when
needed.

    uv run python scripts/upload_colon_a2_hf.py \\
        ~/scverse_conference_2026/pyxa_scverse_demo/data/colon_a2.sdata.zarr
"""

from __future__ import annotations

import argparse
import inspect
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

HF_REPO = "Stellaromics/demo"
DEFAULT_PATH_IN_REPO = "colon_a2.sdata.zarr"


@dataclass(frozen=True)
class UploadPlan:
    """Resolved upload callable and kwargs for the installed ``huggingface_hub``."""

    method: str
    folder_path: Path
    kwargs: dict[str, Any]
    staged: bool = False


def plan_hub_upload(
    folder: Path,
    *,
    repo_id: str = HF_REPO,
    path_in_repo: str = DEFAULT_PATH_IN_REPO,
    repo_type: str = "dataset",
) -> UploadPlan:
    from huggingface_hub import HfApi

    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        raise ValueError(f"not a directory: {folder}")

    api = HfApi()
    upload_large = getattr(api, "upload_large_folder", None)
    if upload_large is not None:
        params = inspect.signature(upload_large).parameters
        if "path_in_repo" in params:
            return UploadPlan(
                method="upload_large_folder",
                folder_path=folder,
                kwargs={
                    "repo_id": repo_id,
                    "repo_type": repo_type,
                    "folder_path": folder,
                    "path_in_repo": path_in_repo,
                },
            )
        # Legacy upload_large_folder: upload folder_path as repo root; stage path_in_repo.
        staging = Path(tempfile.mkdtemp(prefix="hf-upload-stage-"))
        staged_dest = staging / path_in_repo
        shutil.copytree(folder, staged_dest)
        return UploadPlan(
            method="upload_large_folder",
            folder_path=staging,
            staged=True,
            kwargs={
                "repo_id": repo_id,
                "repo_type": repo_type,
                "folder_path": staging,
            },
        )

    return UploadPlan(
        method="upload_folder",
        folder_path=folder,
        kwargs={
            "repo_id": repo_id,
            "repo_type": repo_type,
            "folder_path": folder,
            "path_in_repo": path_in_repo,
        },
    )


def run_upload(plan: UploadPlan) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    upload: Callable[..., Any] = getattr(api, plan.method)
    upload(**plan.kwargs)
    if plan.staged:
        shutil.rmtree(plan.folder_path, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="local colon_a2.sdata.zarr directory")
    parser.add_argument("--repo-id", default=HF_REPO)
    parser.add_argument("--path-in-repo", default=DEFAULT_PATH_IN_REPO)
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the upload plan for this huggingface_hub install (no upload)",
    )
    args = parser.parse_args()

    plan = plan_hub_upload(
        args.folder,
        repo_id=args.repo_id,
        path_in_repo=args.path_in_repo,
        repo_type=args.repo_type,
    )
    if args.dry_run:
        import json

        print(
            json.dumps(
                {
                    "method": plan.method,
                    "folder_path": str(plan.folder_path),
                    "staged": plan.staged,
                    "kwargs": {k: str(v) if k == "folder_path" else v for k, v in plan.kwargs.items()},
                },
                indent=2,
            )
        )
        if plan.staged and plan.folder_path.exists():
            shutil.rmtree(plan.folder_path, ignore_errors=True)
        return

    run_upload(plan)
    print(f"uploaded {args.folder} -> {args.repo_id} ({args.path_in_repo}) via {plan.method}")


if __name__ == "__main__":
    main()
