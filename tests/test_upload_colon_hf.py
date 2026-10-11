"""Upload helper matches the installed huggingface_hub API (no network)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def main() -> None:
    import tempfile

    import huggingface_hub as hf
    from upload_colon_a2_hf import plan_hub_upload

    with tempfile.TemporaryDirectory() as tmp:
        store = Path(tmp) / "colon_a2.sdata.zarr"
        store.mkdir()
        plan = plan_hub_upload(store)
        if hasattr(hf.HfApi(), "upload_large_folder"):
            assert plan.method == "upload_large_folder"
        else:
            assert plan.method == "upload_folder"
            assert plan.kwargs["path_in_repo"] == "colon_a2.sdata.zarr"
            assert plan.kwargs["repo_id"] == "Stellaromics/demo"
            assert plan.kwargs["repo_type"] == "dataset"
        print(f"upload plan OK ({hf.__version__}: {plan.method})")


if __name__ == "__main__":
    main()
