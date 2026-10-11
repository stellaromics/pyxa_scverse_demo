# pyxa_scverse_demo

marimo notebooks for the Stellaromics workshop on 3D spatial omics with Pyxa, SpatialData and Milume.

| Notebook | What it covers | Run |
| --- | --- | --- |
| [`1_intro_marimo.py`](1_intro_marimo.py) | Intro to marimo: reactive cells, UI elements and markdown | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/stellaromics/pyxa_scverse_demo/blob/c007d1497398657584bd3c59e04f104993023174/1_intro_marimo.py) |
| [`2_pyxa_spatialdata.py`](2_pyxa_spatialdata.py) | Read Pyxa output into SpatialData and explore it in the Milume widget | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/stellaromics/pyxa_scverse_demo/blob/c007d1497398657584bd3c59e04f104993023174/2_pyxa_spatialdata.py) |
| [`3_colon_3d_milume.py`](3_colon_3d_milume.py) | Analysis of 3D spatial omics in colorectal cancer with Milume | Run on your own machine; needs ~30 GB of memory and a ~22 GB download |

## Run the colon notebook locally

```
uv run python build_colon_a2.py --download --overwrite
uv run marimo edit 3_colon_3d_milume.py
```

The build downloads ~22.5 GB from the Hub, writes a ~13 GB SpatialData store to `data/`, and peaks at ~30 GB of memory.

### Reshard an existing store (no Pyxa rebuild)

If you already have `colon_a2.sdata.zarr` locally (for example after an earlier build), re-encode images and `labels/cell_labels` with Zarr v3 shards and consolidated metadata:

```
uv run python build_colon_a2.py --reshard ~/scverse_conference_2026/pyxa_scverse_demo/data/colon_a2.sdata.zarr --reshard-in-place --layout-report
```

To write a new path instead of replacing in place, use `--reshard-out`.

### Upload the sharded store to Hugging Face (Clarence's Mac)

After resharding, publish the unzipped SpatialData zarr folder to the `Stellaromics/demo` dataset ([Hub limits](https://huggingface.co/docs/hub/repositories-recommendations): **10,000 entries per directory** hard cap on push; **100,000 files per repo** recommended).

With **huggingface_hub 2.x** (this repo's lockfile), use ``HfApi.upload_folder`` / ``hf upload`` and set ``path_in_repo`` so the store lands at ``colon_a2.sdata.zarr/`` in the dataset:

```bash
# once: hf auth login  (or export HF_TOKEN=...)
uv run python scripts/upload_colon_a2_hf.py \
  ~/scverse_conference_2026/pyxa_scverse_demo/data/colon_a2.sdata.zarr

# inspect the plan for your installed huggingface_hub (no upload):
uv run python scripts/upload_colon_a2_hf.py --dry-run \
  ~/scverse_conference_2026/pyxa_scverse_demo/data/colon_a2.sdata.zarr
```

Older ``huggingface_hub`` releases had ``upload_large_folder`` / ``hf upload-large-folder`` without ``path_in_repo``; the script stages a parent directory when that API is present.

**Opening the store:** with **spatialdata 0.8.0**, ``spatialdata.read_zarr`` expects a local path or fsspec-backed store, not a bare ``https://`` dataset URL. Download or mount the zarr folder first (for example ``hf download``), then ``read_zarr`` the local directory.

Data: [Stellaromics/demo](https://huggingface.co/datasets/Stellaromics/demo) on Hugging Face.
