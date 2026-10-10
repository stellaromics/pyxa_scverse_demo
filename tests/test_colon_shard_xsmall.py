"""Round-trip and Hugging Face layout checks for sharded colon/xsmall SpatialData.

Uses the ``xsmall/`` crop from `Stellaromics/demo` (no full colon download).

    uv run python tests/test_colon_shard_xsmall.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Hugging Face Hub repository recommendations (hard per-folder cap on push; repo cap is guidance).
# https://huggingface.co/docs/hub/repositories-recommendations
HF_MAX_FILES_PER_REPO = 100_000
HF_MAX_ENTRIES_PER_FOLDER = 10_000


def _download_xsmall_raw(dest: Path) -> Path:
    from huggingface_hub import snapshot_download

    snapshot_download(
        "Stellaromics/demo",
        repo_type="dataset",
        allow_patterns=["xsmall/*"],
        local_dir=dest,
    )
    return dest / "xsmall"


def _assert_raster_equal(a, b, label: str) -> None:
    assert list(a.children) == list(b.children), label
    for scale in a.children:
        oa = a[scale].values().__iter__().__next__()
        ba = b[scale].values().__iter__().__next__()
        assert oa.shape == ba.shape, f"{label}/{scale} shape"
        assert np.array_equal(np.asarray(oa), np.asarray(ba)), f"{label}/{scale} pixels"


def main() -> None:
    from spatialdata import read_zarr
    from spatialdata_io.experimental import pyxa

    from colon_a2_sharding import assert_hf_layout_limits, measure_store_layout, write_sharded_spatialdata

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        raw = _download_xsmall_raw(tmp_path / "hf")
        reference = pyxa(raw, labels=True)
        plain = tmp_path / "xsmall_plain.sdata.zarr"
        sharded = tmp_path / "xsmall_sharded.sdata.zarr"
        reference.write(plain, overwrite=True)
        write_sharded_spatialdata(reference, sharded, overwrite=True)

        plain_s = read_zarr(plain)
        shard_s = read_zarr(sharded)

        assert set(plain_s.images) == set(shard_s.images)
        assert set(plain_s.labels) == set(shard_s.labels)
        assert set(plain_s.tables) == set(shard_s.tables)
        assert plain_s.tables["rna"].n_obs == shard_s.tables["rna"].n_obs

        for name in plain_s.images:
            _assert_raster_equal(plain_s.images[name], shard_s.images[name], f"image:{name}")
        for name in plain_s.labels:
            _assert_raster_equal(plain_s.labels[name], shard_s.labels[name], f"labels:{name}")

        assert shard_s.has_consolidated_metadata()

        stats = measure_store_layout(sharded)
        assert_hf_layout_limits(
            stats,
            max_repo_files=HF_MAX_FILES_PER_REPO,
            max_folder_entries=HF_MAX_ENTRIES_PER_FOLDER,
        )

        print("colon shard xsmall: OK")
        print(
            f"  files={stats.total_files} (HF repo guidance <={HF_MAX_FILES_PER_REPO:,})"
        )
        print(
            f"  max_folder_entries={stats.max_entries_per_folder} in {stats.worst_folder!r} "
            f"(HF hard limit <={HF_MAX_ENTRIES_PER_FOLDER:,} per directory)"
        )
        blob = stats.as_dict()["shard_byte_sizes"]
        print(
            f"  shard_blobs={blob['count']} size_bytes min={blob['min']} median={blob['median']} max={blob['max']}"
        )


if __name__ == "__main__":
    main()
