"""Zarr v3 sharded writes for Glasgow colon A2 SpatialData stores.

Images and labels are written with inner chunks of 256² (2D) or 64³ (3D) and
shards sized to land in the 16–64 MB range. Other elements use SpatialData's
normal writers; metadata is consolidated at the root.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import zarr
from ome_zarr.types import JSONDict
from spatialdata import SpatialData
from spatialdata._io._utils import _resolve_zarr_store
from spatialdata._io.format import CurrentRasterFormat, _parse_formats
from spatialdata._io.io_raster import _write_raster
from spatialdata._io.io_zarr import _get_groups_for_element
from spatialdata._io.io_points import write_points
from spatialdata._io.io_shapes import write_shapes
from spatialdata._io.io_table import write_table

# Pin must match pyproject.toml. We call _write_raster (not public write_image) so we can pass
# ome-zarr ``storage_options`` with ``shards`` without waiting on a spatialdata release that
# exposes the same knobs on SpatialData.write().
SPATIALDATA_VERSION_PIN = "0.8.0"

TARGET_SHARD_MIN_BYTES = 16 * 1024**2
TARGET_SHARD_MAX_BYTES = 64 * 1024**2
TARGET_SHARD_BYTES = 32 * 1024**2

INNER_CHUNK_2D = 256
INNER_CHUNK_3D = 64


@dataclass(frozen=True)
class StoreLayoutStats:
    total_files: int
    max_entries_per_folder: int
    worst_folder: str
    shard_byte_sizes: list[int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_files": self.total_files,
            "max_entries_per_folder": self.max_entries_per_folder,
            "worst_folder": self.worst_folder,
            "shard_byte_sizes": {
                "count": len(self.shard_byte_sizes),
                "min": min(self.shard_byte_sizes) if self.shard_byte_sizes else 0,
                "max": max(self.shard_byte_sizes) if self.shard_byte_sizes else 0,
                "median": int(np.median(self.shard_byte_sizes)) if self.shard_byte_sizes else 0,
            },
        }


def _volume_is_3d(shape: tuple[int, ...], dims: tuple[str, ...]) -> bool:
    if "z" not in dims:
        return False
    z_idx = dims.index("z")
    return shape[z_idx] > 1


def _inner_chunk_size(axis: str, axis_size: int, *, is_3d: bool) -> int:
    if axis in ("c", "t"):
        return 1
    if axis == "z":
        return min(INNER_CHUNK_3D, axis_size) if is_3d else axis_size
    if axis in ("y", "x"):
        return min(INNER_CHUNK_3D if is_3d else INNER_CHUNK_2D, axis_size)
    return axis_size


def _inner_chunks(shape: tuple[int, ...], dims: tuple[str, ...]) -> tuple[int, ...]:
    is_3d = _volume_is_3d(shape, dims)
    return tuple(_inner_chunk_size(ax, int(sz), is_3d=is_3d) for ax, sz in zip(dims, shape, strict=True))


def _align_shard(chunk: int, shard: int, axis_size: int) -> int:
    if chunk <= 0:
        return axis_size
    shard = min(shard, axis_size)
    if shard <= chunk:
        return chunk
    n = shard // chunk
    return min(n * chunk, axis_size)


def _shard_chunks(
    chunks: tuple[int, ...],
    shape: tuple[int, ...],
    dims: tuple[str, ...],
    *,
    itemsize: int,
    target_bytes: int = TARGET_SHARD_BYTES,
) -> tuple[int, ...]:
    chunk_volume = int(np.prod(chunks))
    if chunk_volume <= 0:
        return chunks
    bytes_per_chunk = chunk_volume * itemsize
    n_chunks = max(1, target_bytes // max(1, bytes_per_chunk))
    spatial = [i for i, d in enumerate(dims) if d in ("x", "y", "z") and chunks[i] < shape[i]]
    if not spatial:
        return chunks
    per_axis = max(1, int(round(n_chunks ** (1 / len(spatial)))))
    shards = list(chunks)
    for i in spatial:
        shards[i] = _align_shard(chunks[i], chunks[i] * per_axis, int(shape[i]))
    return tuple(shards)


def _storage_options_for_level(
    shape: tuple[int, ...],
    dims: tuple[str, ...],
    dtype: np.dtype,
) -> JSONDict:
    chunks = _inner_chunks(shape, dims)
    shards = _shard_chunks(chunks, shape, dims, itemsize=int(dtype.itemsize))
    return {"chunks": chunks, "shards": shards}


def _storage_options_for_datatree(tree) -> list[JSONDict]:
    from spatialdata._utils import get_pyramid_levels

    shapes = get_pyramid_levels(tree, attr="shape")
    dims_list = get_pyramid_levels(tree, attr="dims")
    dtypes = get_pyramid_levels(tree, attr="dtype")
    assert len(set(dims_list)) == 1
    dims = tuple(dims_list[0])
    return [_storage_options_for_level(tuple(map(int, sh)), dims, np.dtype(dt)) for sh, dt in zip(shapes, dtypes)]


def write_sharded_spatialdata(
    sdata: SpatialData,
    file_path: str | Path,
    *,
    overwrite: bool = False,
    consolidate_metadata: bool = True,
    raster_compressor: dict[Literal["lz4", "zstd"], int] | None = None,
) -> None:
    """Write ``sdata`` to a Zarr v3 SpatialData store with sharded images and labels."""
    from spatialdata._io._utils import _validate_compressor_args

    _validate_compressor_args(raster_compressor)
    file_path = Path(file_path)
    sdata._validate_can_safely_write_to_path(file_path, overwrite=overwrite)
    sdata._validate_all_elements()
    parsed = _parse_formats(None)
    zarr_format = parsed["SpatialData"].zarr_format

    store = _resolve_zarr_store(file_path)
    zarr_group = zarr.create_group(store=store, overwrite=overwrite, zarr_format=zarr_format)
    sdata.write_attrs(zarr_group=zarr_group, sdata_format=parsed["SpatialData"])
    store.close()

    raster_format = parsed["raster"]
    for element_type, element_name, element in sdata.gen_elements():
        root_group, element_type_group, element_group = _get_groups_for_element(
            zarr_path=file_path,
            element_type=element_type,
            element_name=element_name,
            use_consolidated=False,
        )
        if element_type == "images":
            storage_options = _storage_options_for_datatree(element)
            _write_raster(
                "image",
                element,
                element_group,
                element_name,
                raster_format,
                storage_options=storage_options,
                raster_compressor=raster_compressor,
            )
        elif element_type == "labels":
            storage_options = _storage_options_for_datatree(element)
            _write_raster(
                "labels",
                element,
                root_group,
                element_name,
                raster_format,
                storage_options=storage_options,
                raster_compressor=raster_compressor,
            )
        elif element_type == "points":
            write_points(element, element_group, element_format=parsed["points"])
        elif element_type == "shapes":
            write_shapes(element, element_group, element_format=parsed["shapes"])
        elif element_type == "tables":
            write_table(element, element_type_group, element_name, element_format=parsed["tables"])
        else:
            raise ValueError(f"Unknown element type: {element_type}")

    sdata.path = file_path
    if consolidate_metadata:
        sdata.write_consolidated_metadata()


def reshard_spatialdata_store(
    source: str | Path,
    dest: str | Path | None = None,
    *,
    in_place: bool = False,
    overwrite: bool = False,
) -> Path:
    """Re-encode rasters in an existing SpatialData zarr without rebuilding from Pyxa."""
    from spatialdata import read_zarr

    source = Path(source)
    if in_place:
        if dest is not None and Path(dest) != source:
            raise ValueError("Pass either --reshard-in-place or --reshard-out, not both with conflicting paths.")
        dest = source
        tmp = Path(tempfile.mkdtemp(prefix=f"{source.name}.reshard.", dir=source.parent))
        try:
            sdata = read_zarr(source)
            write_sharded_spatialdata(sdata, tmp, overwrite=True)
            if source.exists():
                shutil.rmtree(source)
            tmp.rename(source)
            return source
        except Exception:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
            raise
    dest = Path(dest or source)
    sdata = read_zarr(source)
    write_sharded_spatialdata(sdata, dest, overwrite=overwrite)
    return dest


def _folder_entry_counts(root: Path) -> dict[str, int]:
    """Count direct children (files + subdirs) per directory for HF layout checks."""
    counts: dict[str, int] = defaultdict(int)
    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        key = "." if rel == "." else rel
        counts[key] += len(dirnames) + len(filenames)
    return counts


def _collect_shard_blob_sizes(root: Path) -> list[int]:
    sizes: list[int] = []
    for path in root.rglob("*"):
        if path.is_file() and path.name not in {"zarr.json", ".zmetadata"}:
            parent = path.parent
            # Zarr v3 shard objects are leaf blobs under c/... paths without further zarr.json siblings
            if parent.name.isdigit() or (parent.parent and parent.parent.name in {"c", "0"}):
                sizes.append(path.stat().st_size)
    return sizes


def measure_store_layout(root: str | Path) -> StoreLayoutStats:
    root = Path(root)
    total_files = sum(1 for p in root.rglob("*") if p.is_file())
    counts = _folder_entry_counts(root)
    worst_folder, max_entries = max(counts.items(), key=lambda kv: kv[1])
    shard_sizes = _collect_shard_blob_sizes(root)
    return StoreLayoutStats(
        total_files=total_files,
        max_entries_per_folder=max_entries,
        worst_folder=worst_folder,
        shard_byte_sizes=shard_sizes,
    )


def assert_hf_layout_limits(stats: StoreLayoutStats, *, max_repo_files: int = 100_000, max_folder_entries: int = 10_000) -> None:
    if stats.total_files > max_repo_files:
        raise AssertionError(f"store has {stats.total_files} files (limit {max_repo_files})")
    if stats.max_entries_per_folder > max_folder_entries:
        raise AssertionError(
            f"folder {stats.worst_folder!r} has {stats.max_entries_per_folder} entries "
            f"(limit {max_folder_entries})"
        )


def dump_layout_report(stats: StoreLayoutStats) -> str:
    return json.dumps(stats.as_dict(), indent=2)
