# BubbleAI microscope data

This folder contains the packaged HDF5 versions of the BubbleAI microscope datasets and the helper script that generates them from the raw data in `BubbleAI Microscope/`.

## Contents
- `convert_bubbleai_to_h5.py`: conversion utility that bundles images + Excel metadata for each mixture into a single `.h5` file.
- `h5/`: output HDF5 files (one per mixture).

## How the converter works
1. Looks for each mixture folder inside `BubbleAI Microscope/`.
2. Stacks all PNGs it finds (keeps original file names) and stores them compressed under `/images/data` in the HDF5 file.
3. Reads every `.xlsx` file in the mixture folder and stores its table as CSV text under `/metadata/<name>`, keeping column names and the source filename in attributes.

Run it from the repository root:
```bash
python data/BubbleAI/convert_bubbleai_to_h5.py
```

Outputs are written to `data/BubbleAI/h5/<mixture>.h5`. The raw data stays untouched in `BubbleAI Microscope/`.
