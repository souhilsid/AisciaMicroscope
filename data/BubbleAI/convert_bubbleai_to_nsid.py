"""
Convert BubbleAI packaged HDF5 files into NSID-compatible HDF5 for the DT STEM/AFM classes.

Usage:
    python convert_bubbleai_to_nsid.py
    python convert_bubbleai_to_nsid.py --input data/BubbleAI/h5/0_25_wt_coco_b_0_75wt_coco_glucoside.h5 --output data/BubbleAI/nsid/custom.h5

What it does:
- Reads a BubbleAI .h5 (images + metadata tables).
- Wraps the image stack as a sidpy Dataset (IMAGE) with dimensions: frame, y, x, channel.
- Stores metadata tables in original_metadata['bubbleai_metadata'].
- Writes NSID-compliant HDF5 via pyNSID, so SciFiReaders.NSIDReader and DT STEM/AFM classes can load it.

Requires: sidpy, pyNSID, dask (installed via pip).
"""

from __future__ import annotations

import argparse
import json
import uuid
from io import StringIO
from pathlib import Path
from typing import Dict, List

import dask.array as da
import h5py
import numpy as np
import pandas as pd
import sidpy
import sidpy.sid
import sidpy.sid.dimension as sdim
from pyNSID import io as nsid_io


def load_bubbleai(path: Path):
    """Load images and metadata tables from a BubbleAI HDF5 file."""
    with h5py.File(path, "r") as h5:
        if "images/data" not in h5:
            raise ValueError(f"No images found in {path}")
        images_ds = h5["images/data"]
        images = np.asarray(images_ds)  # load into memory

        metadata: Dict[str, pd.DataFrame] = {}
        if "metadata" in h5:
            for key, ds in h5["metadata"].items():
                raw = ds[()]
                text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
                metadata[key] = pd.read_csv(StringIO(text))

        attrs = dict(h5.attrs)

    return images, metadata, attrs


def build_sidpy_dataset(images: np.ndarray, metadata: Dict[str, pd.DataFrame], name: str) -> sidpy.Dataset:
    """Wrap the dask image stack as a sidpy Dataset with IMAGE data_type."""
    ds = sidpy.Dataset.from_array(images, name=name)
    ds.data_type = sidpy.DataType.IMAGE
    ds.quantity = "intensity"
    ds.units = "a.u."
    ds.modality = "BubbleAI"
    ds.source = "BubbleAI"
    ds.original_metadata["uuid"] = str(uuid.uuid4())

    # store metadata tables as JSON strings per key to keep things small
    meta_dict = {}
    for key, df in metadata.items():
        meta_dict[key] = {
            "columns": list(df.columns),
            "data": df.to_dict(orient="list"),
        }
    ds.original_metadata["bubbleai_metadata"] = meta_dict

    # Dimensions: frame, y, x, channel
    n_frames, height, width, channels = images.shape
    ds.set_dimension(0, sdim.Dimension(np.arange(n_frames), name="frame", units="index", quantity="frame"))
    ds.set_dimension(1, sdim.Dimension(np.arange(height), name="y", units="pixel", quantity="y"))
    ds.set_dimension(2, sdim.Dimension(np.arange(width), name="x", units="pixel", quantity="x"))
    ds.set_dimension(3, sdim.Dimension(np.arange(channels), name="channel", units="index", quantity="channel"))

    return ds


def convert_file(input_path: Path, output_path: Path) -> Path:
    images, metadata, attrs = load_bubbleai(input_path)
    ds = build_sidpy_dataset(images, metadata, name="Channel_000")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5f:
        nsid_io.write_nsid_dataset(ds, h5f, main_data_name="Channel_000", compression="gzip", compression_opts=4)
        # store top-level info
        h5f.attrs["bubbleai_source"] = str(input_path)
        h5f.attrs["bubbleai_attrs"] = json.dumps({k: str(v) for k, v in attrs.items()})

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert BubbleAI .h5 to NSID HDF5 for DT STEM/AFM.")
    parser.add_argument(
        "--input",
        type=Path,
        help="Single BubbleAI .h5 file to convert. If omitted, converts all in data/BubbleAI/h5 with images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output NSID .h5 path (only when converting a single input).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    default_input_dir = repo_root / "data" / "BubbleAI" / "h5"
    default_output_dir = repo_root / "data" / "BubbleAI" / "nsid"

    targets: List[Path] = []
    if args.input:
        targets = [args.input]
    else:
        targets = [p for p in default_input_dir.glob("*.h5")]

    if not targets:
        raise RuntimeError("No input .h5 files found.")

    for inp in targets:
        out = args.output if args.output else default_output_dir / (inp.stem + "_nsid.h5")
        print(f"Converting {inp.name} -> {out}")
        convert_file(inp, out)
    print("Done.")


if __name__ == "__main__":
    main()
