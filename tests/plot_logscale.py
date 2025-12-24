#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "vhsh",
# ]
# [tool.uv.sources]
# vhsh = { path = "../", editable = true }
# ///

import matplotlib.pyplot as plt
import numpy as np

from vhsh.scene import interpolate_log


def linear(min_, max_, value):
    return min_ + value * (max_ - min_)

def log_simple(min_, max_, value):
     return np.exp(np.log(max_ - min_ + 1) * value) + min_ - 1

def log_vhsh(min_, max_, value):
    return np.frompyfunc(interpolate_log, 3, 1)(value, min_, max_).astype(float)

plt.figure(figsize=(16,4))
cases = [
    [(0, 1), (-1, 0), (-2, -1), (1, 2), (-1, 1)],
    [(1, 0), (0, -1), (-1, -2), (2, 1), (1, -1)],
]
n_rows = len(cases)
n_cols = len(cases[0])
n = 100
xs = np.linspace(0, 1, n)
for row_idx, row in enumerate(cases):
    for col_idx, (min_, max_) in enumerate(row):
        print()
        print(row_idx, col_idx, min_, max_)
        plt.subplot(n_rows, n_cols, n_cols * row_idx + col_idx + 1)
        plt.title(f"[{min_},{max_}]")

        plt.plot(xs, linear(min_, max_, xs), label="linear")
        plt.plot(xs, log_simple(min_, max_, xs), label="simple")
        plt.plot(xs, log_vhsh(min_, max_, xs), label="vhsh")

        plt.legend()

plt.show()
