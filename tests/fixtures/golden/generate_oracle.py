#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generate the independent gguf-py oracle vectors in ./oracle (RM-2209).

For each supported GGUF wire dtype this reads <name>.packed.bin -- the same
deterministic inputs used by generate_golden.py -- and writes:

  oracle/<name>.bin        -- f32 outputs, little-endian IEEE-754
  oracle/provenance.json   -- tool, version, and SHA-256 of every vector

Unlike generate_golden.py (an in-repo port of the llama.cpp scalar
decoders), the expected outputs here come from an implementation this
repository does not author: `gguf.quants` from gguf-py, the Python package
maintained in-tree in ggml-org/llama.cpp and published to PyPI as `gguf`
(numpy decoders used by llama.cpp's own tooling).

IQ3_M has no upstream implementation -- engram-parser's 111-byte block
layout is internal (REVIEW.md; RM-2012) -- so it has no oracle vector and
its .expected.bin remains a regression lock only.

The tool is pinned as `gguf==0.19.0`; provenance.json stamps the versions
actually used at generation time. Regenerate with the pin to reproduce the
checked-in vectors byte-for-byte.

Usage:

    uv run --python 3.13 --with 'gguf==0.19.0' \
        python tests/fixtures/golden/generate_oracle.py

    # or:
    pip install 'gguf==0.19.0'
    python tests/fixtures/golden/generate_oracle.py
"""
import hashlib
import importlib.metadata
import json
import os
import sys

import numpy as np
from gguf import GGMLQuantizationType
from gguf.quants import dequantize

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'oracle')

PINNED = '0.19.0'

# name -> (GGMLQuantizationType member, packed block size in bytes).
TYPES = {
    'q8_0': ('Q8_0', 34),
    'q5_k': ('Q5_K', 176),
    'q6_k': ('Q6_K', 210),
    'q4_0': ('Q4_0', 18),
    'q4_1': ('Q4_1', 20),
    'q2_k': ('Q2_K', 84),
    'q3_k': ('Q3_K', 110),
    'q4_k': ('Q4_K', 144),
}


def main():
    actual = importlib.metadata.version('gguf')
    if actual != PINNED:
        print(
            f'warning: gguf {actual} != pinned {PINNED}; '
            'vectors may differ from those checked in',
            file=sys.stderr,
        )
    os.makedirs(OUT, exist_ok=True)
    vectors = {}
    for name, (member, bsize) in TYPES.items():
        packed_path = os.path.join(HERE, f'{name}.packed.bin')
        with open(packed_path, 'rb') as f:
            packed = f.read()
        blocks = np.frombuffer(packed, dtype=np.uint8).reshape(-1, bsize)
        data = np.asarray(
            dequantize(blocks, GGMLQuantizationType[member]), dtype=np.float32
        ).ravel().tobytes()
        out_path = os.path.join(OUT, f'{name}.bin')
        with open(out_path, 'wb') as f:
            f.write(data)
        vectors[name] = {
            'input': f'{name}.packed.bin',
            'input_sha256': hashlib.sha256(packed).hexdigest(),
            'output': f'oracle/{name}.bin',
            'output_sha256': hashlib.sha256(data).hexdigest(),
            'elements': len(data) // 4,
        }
        print(f'{name}: {len(data) // 4} values -> oracle/{name}.bin')
    provenance = {
        'issue': 'RM-2209',
        'tool': 'gguf-py (ggml-org/llama.cpp in-tree Python package, PyPI `gguf`)',
        'decoder': 'gguf.quants.dequantize',
        'tool_version_pinned': PINNED,
        'tool_version_actual': actual,
        'numpy_version': np.__version__,
        'notes': (
            'IQ3_M has no upstream implementation (internal 111-byte layout; '
            'RM-2012) and therefore no oracle vector.'
        ),
        'vectors': vectors,
    }
    with open(os.path.join(OUT, 'provenance.json'), 'w') as f:
        json.dump(provenance, f, indent=2)
        f.write('\n')
    print('provenance -> oracle/provenance.json')


if __name__ == '__main__':
    main()
