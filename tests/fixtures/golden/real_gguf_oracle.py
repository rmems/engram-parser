#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Read one bounded real GGUF tensor with gguf-py; stdout is little-endian f32."""

import argparse
import importlib.metadata
import sys

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

PINNED_GGUF = "0.19.0"
MAX_ELEMENTS = 8_388_608


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("tensor")
    parser.add_argument("dtype", choices=("Q8_0", "Q5_K", "Q6_K"))
    parser.add_argument("elements", type=int)
    args = parser.parse_args()

    installed = importlib.metadata.version("gguf")
    if installed != PINNED_GGUF:
        parser.error(f"install gguf=={PINNED_GGUF}; found {installed}")

    reader = GGUFReader(args.model)
    tensor = next((t for t in reader.tensors if t.name == args.tensor), None)
    if tensor is None:
        parser.error(f"tensor not found: {args.tensor!r}")
    if tensor.tensor_type.name != args.dtype:
        parser.error(f"expected {args.dtype}; found {tensor.tensor_type.name}")
    if not 0 < args.elements <= MAX_ELEMENTS or tensor.n_elements != args.elements:
        parser.error(f"expected {args.elements} bounded elements; found {tensor.n_elements}")

    decoded = np.asarray(dequantize(tensor.data, tensor.tensor_type), dtype="<f4")
    sys.stdout.buffer.write(decoded.tobytes(order="C"))


if __name__ == "__main__":
    main()
