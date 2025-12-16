"""
Pyro server to expose BubbleAI HDF5 datasets via a microscope-like API.

Usage:
    python -m DTMicroscope.server.server_bubbleai --h5 data/BubbleAI/h5/0_25_wt_coco_b_0_75wt_coco_glucoside.h5

Endpoints (matching the STEM server style):
    - get_overview_image(idx=0) -> list, shape, dtype
    - get_point_data(idx, x, y)  -> list/int, shape, dtype
    - get_metadata_keys()        -> list of available metadata tables
    - get_metadata_table(key)    -> CSV string of the table
"""

from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import pandas as pd
import Pyro5.api


def _read_bubbleai_h5(path: Path):
    """Load images, filenames, metadata tables, and attributes from an HDF5 file."""
    with h5py.File(path, "r") as h5:
        images = None
        filenames: List[str] = []
        if "images" in h5:
            images = np.array(h5["images/data"])
            filenames = [
                f.decode("utf-8") if isinstance(f, (bytes, bytearray)) else str(f)
                for f in h5["images/filenames"]
            ]

        metadata = {}
        if "metadata" in h5:
            for key, ds in h5["metadata"].items():
                raw = ds[()]
                text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
                metadata[key] = pd.read_csv(StringIO(text))

        attrs = dict(h5.attrs)
    return images, filenames, metadata, attrs


class BubbleAIMicroscope:
    """Minimal microscope-like wrapper around BubbleAI HDF5 files."""

    def __init__(self, h5_path: Path):
        self.h5_path = Path(h5_path)
        self.images, self.filenames, self.metadata, self.attrs = _read_bubbleai_h5(self.h5_path)
        self.frame_count = 0 if self.images is None else self.images.shape[0]
        self.height = None if self.images is None else self.images.shape[1]
        self.width = None if self.images is None else self.images.shape[2]

    def get_overview_image(self, idx: int = 0) -> np.ndarray:
        if self.images is None or self.frame_count == 0:
            raise RuntimeError("No images stored in this H5 file")
        return self.images[idx]

    def get_point_data(self, idx: int, x: int, y: int):
        frame = self.get_overview_image(idx)
        return frame[y, x]

    def get_metadata_keys(self) -> List[str]:
        return list(self.metadata.keys())

    def get_metadata_table(self, key: str) -> pd.DataFrame:
        return self.metadata[key]


@Pyro5.api.expose
class BubbleAIServer:
    """Pyro-exposed server that forwards calls to BubbleAIMicroscope."""

    def __init__(self, h5_path: Path):
        self.mic = BubbleAIMicroscope(h5_path)

    def get_overview_image(self, idx: int = 0) -> Tuple[list, Tuple[int, ...], str]:
        img = self.mic.get_overview_image(idx)
        return img.tolist(), img.shape, str(img.dtype)

    def get_point_data(self, idx: int, x: int, y: int):
        val = self.mic.get_point_data(idx, x, y)
        try:
            shape = val.shape  # numpy scalar has shape ()
        except AttributeError:
            shape = ()
        return val.tolist() if hasattr(val, "tolist") else val, shape, str(getattr(val, "dtype", type(val)))

    def get_metadata_keys(self) -> List[str]:
        return self.mic.get_metadata_keys()

    def get_metadata_table(self, key: str) -> str:
        df = self.mic.get_metadata_table(key)
        return df.to_csv(index=False)


def main():
    parser = argparse.ArgumentParser(description="Run BubbleAI Pyro server.")
    parser.add_argument(
        "--h5",
        type=Path,
        default=Path("data/BubbleAI/h5/0_25_wt_coco_b_0_75wt_coco_glucoside.h5"),
        help="Path to BubbleAI .h5 file.",
    )
    parser.add_argument("--port", type=int, default=9091, help="Pyro port to bind.")
    args = parser.parse_args()

    if not args.h5.exists():
        raise FileNotFoundError(f"BubbleAI file not found: {args.h5}")

    daemon = Pyro5.api.Daemon(port=args.port)
    uri = daemon.register(BubbleAIServer(args.h5), objectId="microscope.server")
    print("BubbleAI server ready:", uri)
    print(f"Serving file: {args.h5}")
    daemon.requestLoop()


if __name__ == "__main__":
    main()
