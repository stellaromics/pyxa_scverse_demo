"""Build the Glasgow colon A2 SpatialData for ``3_colon_3d_milume.py`` with spatialdata-io's ``pyxa`` reader.

``--download`` fetches the region from the ``colon/`` folder of the
`Stellaromics/demo <https://huggingface.co/datasets/Stellaromics/demo>`_ dataset into
``data/hf/colon/`` (about 22 GB; the transcripts stay on the Hub). ``--source`` also takes a
Pyxa ``Analysis Group`` directory. The reader returns the table, the mosaic (read from its zip
in place) and 3D cell labels drawn onto the mosaic's grid; this script keeps the cells Pyxa
Studio kept and writes ``data/colon_a2.sdata.zarr``.

Images and labels are written as Zarr v3 with sharded OME-NGFF (16–64 MB shards; 256² / 64³
inner chunks). Use ``--reshard`` on an existing local store (for example Clarence's
``~/scverse_conference_2026/pyxa_scverse_demo/data/colon_a2.sdata.zarr``) to re-encode
rasters without rebuilding from raw Pyxa files.

    uv run python build_colon_a2.py --download --overwrite
    uv run python build_colon_a2.py --reshard ~/scverse_conference_2026/pyxa_scverse_demo/data/colon_a2.sdata.zarr --reshard-in-place
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from spatialdata import SpatialData, read_zarr
from spatialdata_io.experimental import pyxa

from colon_a2_sharding import measure_store_layout, reshard_spatialdata_store, write_sharded_spatialdata

DATA = Path(__file__).resolve().parent / "data"
HF_REPO = "Stellaromics/demo"
HF_FOLDER = "colon"
DEFAULT_SOURCE = DATA / "hf" / HF_FOLDER
OUT = DATA / "colon_a2.sdata.zarr"
# What the build reads from the Hub; the transcripts (cell_assigned_gene_v1.csv) stay there.
HF_FILES = [
    "cell_by_gene_v1.csv",
    "cell_metadata_v1.csv",
    "pyxa_studio_v1.csv",
    "segmentation_geometries_v1.parquet",
    "mosaic_3d.ome.zarr.zip",
]


def download(dest: Path = DEFAULT_SOURCE) -> Path:
    """Fetch the colon region from the Hub into ``dest``."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        HF_REPO, repo_type="dataset", allow_patterns=[f"{HF_FOLDER}/{f}" for f in HF_FILES], local_dir=dest.parent
    )
    return dest.parent / HF_FOLDER


def read(source: Path) -> SpatialData:
    """The region as a SpatialData; an ``Analysis Group`` keeps its tables and mosaic apart."""
    if (source / "ag_output").is_dir():
        return pyxa(
            source / "ag_output",
            cell_assigned_gene=False,
            labels=True,
            image=source / "Region" / "mosaic" / "mosaic_3d.ome.zarr",
        )
    return pyxa(source, cell_assigned_gene=False, labels=True)


def _filter_studio_cells(sdata: SpatialData) -> SpatialData:
    """Drop cells without a Pyxa Studio ``Cluster`` when that column exists (colon only)."""
    table = sdata.tables["rna"]
    if "Cluster" not in table.obs.columns:
        return sdata
    sdata.tables["rna"] = table[table.obs["Cluster"].notna()].copy()
    return sdata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Hub folder or Pyxa 'Analysis Group'")
    parser.add_argument("--download", action="store_true", help=f"fetch {HF_REPO}/{HF_FOLDER} into --source first")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--reshard",
        type=Path,
        metavar="STORE",
        help="re-encode an existing SpatialData zarr with sharded images/labels (no Pyxa rebuild)",
    )
    parser.add_argument(
        "--reshard-out",
        type=Path,
        metavar="PATH",
        help="output path for --reshard (default: same as --reshard)",
    )
    parser.add_argument(
        "--reshard-in-place",
        action="store_true",
        help="replace --reshard store atomically (writes to a temp dir, then swaps)",
    )
    parser.add_argument("--layout-report", action="store_true", help="print file/shard layout stats after writing")
    args = parser.parse_args()

    if args.reshard is not None:
        t0 = time.perf_counter()
        out = reshard_spatialdata_store(
            args.reshard,
            args.reshard_out,
            in_place=args.reshard_in_place,
            overwrite=args.overwrite or args.reshard_in_place,
        )
        sdata = read_zarr(out)
        elapsed = time.perf_counter() - t0
        summary = {
            "mode": "reshard",
            "out": str(out),
            "n_obs": sdata.tables["rna"].n_obs,
            "labels": list(sdata.labels),
            "images": list(sdata.images),
            "seconds": round(elapsed, 1),
        }
        if args.layout_report:
            summary["layout"] = measure_store_layout(out).as_dict()
        print(json.dumps(summary, indent=1))
        return

    if args.download:
        args.source = download(args.source)

    sdata = _filter_studio_cells(read(args.source))
    t0 = time.perf_counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_sharded_spatialdata(sdata, args.out, overwrite=args.overwrite)
    elapsed = time.perf_counter() - t0
    summary = {
        "out": str(args.out),
        "n_obs": sdata.tables["rna"].n_obs,
        "labels": list(sdata.labels),
        "images": list(sdata.images),
        "seconds": round(elapsed, 1),
    }
    if args.layout_report:
        summary["layout"] = measure_store_layout(args.out).as_dict()
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
