"""
Quick test client for the BubbleAI Pyro server.

Usage:
    python -m DTMicroscope.server.client_bubbleai_test --uri PYRO:microscope.server@localhost:9091
"""

from __future__ import annotations

import argparse

import Pyro5.api


def main():
    parser = argparse.ArgumentParser(description="Test BubbleAI Pyro client.")
    parser.add_argument(
        "--uri",
        type=str,
        default="PYRO:microscope.server@localhost:9091",
        help="Pyro URI of the BubbleAI server.",
    )
    args = parser.parse_args()

    proxy = Pyro5.api.Proxy(args.uri)

    img_list, img_shape, img_dtype = proxy.get_overview_image(0)
    print("Received image:", img_shape, img_dtype)

    keys = proxy.get_metadata_keys()
    print("Metadata keys:", keys)
    if keys:
        csv_text = proxy.get_metadata_table(keys[0])
        print("First metadata table preview:")
        print("\n".join(csv_text.splitlines()[:5]))


if __name__ == "__main__":
    main()
