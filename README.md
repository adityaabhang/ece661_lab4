# ece661_lab4

ECE 661 (Fall 2026) Homework 4 — Aditya Abhang.

## Files

- `hw4_AdityaAbhang.py` — all code for Tasks 1–3 (and the Bonus)
- `environment.yml` — conda environment, exported with `conda env export`
- `output/` — figures produced by the script

The provided images (`view_a.jpg`, `view_b.jpg`, `edge_patch.jpg`) are not included, per the submission instructions.

## How to run

```bash
conda env create -f environment.yml
conda activate ece661
# place view_a.jpg, view_b.jpg and edge_patch.jpg in this directory
# (or change DATA_DIR at the top of the script)
python hw4_AdityaAbhang.py
```

Outputs are written to `output/`.

## Seeds

No random numbers are used; results are deterministic.

## Compute setup

- CPU: AMD Ryzen 7 5700U (8 cores / 16 threads), 18 GB RAM, no GPU
- OS: Linux (kernel 6.8)
- Python 3.11.16
