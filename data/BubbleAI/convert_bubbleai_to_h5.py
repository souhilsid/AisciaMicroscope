"""
Utility to package the BubbleAI microscope datasets into compact HDF5 files.

Each HDF5 file contains:
- images/data: uint8 array (n_images, height, width, 3) with gzip compression
- images/filenames: original PNG file names
- metadata/<sheet>: CSV-formatted table stored as UTF-8 text

Usage:
    python convert_bubbleai_to_h5.py

Inputs are read from the repo's "BubbleAI Microscope" folder.
Outputs are written to data/BubbleAI/h5.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def slugify(name: str) -> str:
    """Convert a file/folder name into a filesystem-friendly slug."""
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_")
    slug = re.sub(r"_{2,}", "_", slug)
    return slug.lower()


def write_images_group(h5: h5py.File, image_files: list[Path]) -> None:
    """Store image stack and file names."""
    first = Image.open(image_files[0]).convert("RGB")
    height, width = first.height, first.width
    image_count = len(image_files)

    images_group = h5.require_group("images")
    images_group.attrs["height"] = height
    images_group.attrs["width"] = width
    images_group.attrs["channels"] = 3
    images_group.attrs["count"] = image_count

    data_ds = images_group.create_dataset(
        "data",
        shape=(image_count, height, width, 3),
        dtype=np.uint8,
        chunks=(1, height, width, 3),
        compression="gzip",
        compression_opts=4,
    )

    name_dtype = h5py.string_dtype(encoding="utf-8")
    images_group.create_dataset(
        "filenames",
        data=np.array([path.name for path in image_files], dtype=name_dtype),
        dtype=name_dtype,
    )

    for idx, path in enumerate(tqdm(image_files, desc="images", leave=False)):
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)
        if arr.shape != (height, width, 3):
            raise ValueError(
                f"Inconsistent image shape in {path} (expected {(height, width, 3)}, found {arr.shape})"
            )
        data_ds[idx] = arr


def write_metadata_group(h5: h5py.File, xlsx_files: list[Path]) -> None:
    """Store each Excel sheet as a CSV text block."""
    meta_group = h5.require_group("metadata")
    text_dtype = h5py.string_dtype(encoding="utf-8")

    for path in xlsx_files:
        df = pd.read_excel(path)
        csv_text = df.to_csv(index=False)
        ds_name = slugify(path.stem)
        ds = meta_group.create_dataset(ds_name, data=np.array(csv_text, dtype=text_dtype))
        ds.attrs["source_file"] = path.name
        ds.attrs["rows"] = df.shape[0]
        ds.attrs["cols"] = df.shape[1]
        ds.attrs["columns"] = json.dumps([str(c) for c in df.columns])


def convert_dataset(mixture_dir: Path, output_dir: Path) -> Path:
    """Convert a single mixture folder."""
    image_files = sorted(mixture_dir.rglob("*.png"))
    xlsx_files = sorted(mixture_dir.glob("*.xlsx"))
    slug = slugify(mixture_dir.name)
    output_path = output_dir / f"{slug}.h5"

    with h5py.File(output_path, "w") as h5:
        h5.attrs["mixture_name"] = mixture_dir.name
        h5.attrs["source_dir"] = str(mixture_dir.resolve())
        h5.attrs["image_count"] = len(image_files)
        h5.attrs["metadata_files"] = json.dumps([path.name for path in xlsx_files])

        if image_files:
            write_images_group(h5, image_files)
        if xlsx_files:
            write_metadata_group(h5, xlsx_files)

    return output_path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_root = repo_root / "BubbleAI Microscope"
    output_root = Path(__file__).parent / "h5"
    output_root.mkdir(parents=True, exist_ok=True)

    mixtures = sorted([p for p in input_root.iterdir() if p.is_dir()])
    if not mixtures:
        raise RuntimeError(f"No mixture folders found in {input_root}")

    print(f"Found {len(mixtures)} mixtures in {input_root}")
    for mixture in mixtures:
        output_path = convert_dataset(mixture, output_root)
        print(f"Saved {output_path.name}")

    print(f"Done. HDF5 files are in {output_root}")


if __name__ == "__main__":
    main()
