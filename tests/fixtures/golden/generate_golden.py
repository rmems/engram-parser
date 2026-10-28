#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Regenerate the golden dequant fixtures in this directory.

For each supported packed dtype this writes:

  <name>.packed.bin    -- deterministic packed blocks (PRNG seed 0xE9)
  <name>.expected.bin  -- expected f32 outputs, little-endian IEEE-754

The expected outputs are produced by a bit-exact Python port of the
llama.cpp scalar decoders (dequantize_row_q8_0 / q5_K / q6_K in
ggml/src/ggml-quants.c, tag b4800). The Q8_0, Q5_K, and Q6_K vectors are
therefore *conformance* vectors against an external reference.

IQ3_M is an internal 111-byte block layout, not a GGUF wire type; its
vectors are a literal port of `dequantize_row_iq3_m` and only lock in
current behavior as a *regression* check (they prove nothing about
external conformance).

Usage:

    python3 tests/fixtures/golden/generate_golden.py
"""
import random
import struct
import os

OUT = os.path.dirname(os.path.abspath(__file__))
BLOCKS_PER_TYPE = 4


def f16_to_f32(bits: int) -> float:
    return struct.unpack('<e', struct.pack('<H', bits))[0]


def get_scale_min_k4(is_: int, scales: bytes):
    if is_ < 4:
        return scales[is_] & 63, scales[is_ + 4] & 63
    return (
        (scales[is_ + 4] & 0x0F) | ((scales[is_ - 4] >> 6) << 4),
        (scales[is_ + 4] >> 4) | ((scales[is_] >> 6) << 4),
    )


def dequant_q8_0(blocks):
    out = []
    for b in blocks:
        d = f16_to_f32(b[0] | (b[1] << 8))
        for i in range(32):
            q = b[2 + i]
            q = q - 256 if q >= 128 else q
            out.append(q * d)
    return out


def dequant_q5_k(blocks):
    out = []
    for x in blocks:
        d = f16_to_f32(x[0] | (x[1] << 8))
        dmin = f16_to_f32(x[2] | (x[3] << 8))
        scales, qh, ql = x[4:16], x[16:48], x[48:176]
        is_ = 0
        u1, u2 = 1, 2
        for _j in range(0, 256, 64):
            sc, m = get_scale_min_k4(is_, scales)
            d1, m1 = d * sc, dmin * m
            sc, m = get_scale_min_k4(is_ + 1, scales)
            d2, m2 = d * sc, dmin * m
            for l in range(32):
                out.append(d1 * ((ql[l] & 0xF) + (16 if qh[l] & u1 else 0)) - m1)
            for l in range(32):
                out.append(d2 * ((ql[l] >> 4) + (16 if qh[l] & u2 else 0)) - m2)
            ql = ql[32:]
            is_ += 2
            u1 = (u1 << 2) & 0xFF
            u2 = (u2 << 2) & 0xFF
    return out


def dequant_q6_k(blocks):
    out = []
    for x in blocks:
        ql, qh = x[0:128], x[128:192]
        sc = [(b - 256 if b >= 128 else b) for b in x[192:208]]
        d = f16_to_f32(x[208] | (x[209] << 8))
        y = [0.0] * 256
        pos = 0
        for _n in range(0, 256, 128):
            for l in range(32):
                is_ = l // 16
                q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32
                q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32
                q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32
                q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32
                y[pos + l] = d * sc[is_] * q1
                y[pos + l + 32] = d * sc[is_ + 2] * q2
                y[pos + l + 64] = d * sc[is_ + 4] * q3
                y[pos + l + 96] = d * sc[is_ + 6] * q4
            pos += 128
            ql, qh, sc = ql[64:], qh[32:], sc[8:]
        out.extend(y)
    return out


def dequant_iq3_m(blocks):
    """Literal port of engram-parser dequantize_row_iq3_m (internal layout).
    Regression vectors only; IQ3_M is not a GGUF wire type."""
    out = []
    for x in blocks:
        d = f16_to_f32(x[0] | (x[1] << 8))
        hmask, qs, scales, scales_h = x[2:34], x[34:98], x[98:110], x[110]
        sc = [0] * 16
        bit_pos = 0
        for i in range(16):
            byte_idx, bit_shift = bit_pos // 8, bit_pos % 8
            val = (scales[byte_idx] >> bit_shift) & 0x3F if byte_idx < 12 else 0
            if bit_shift > 2 and byte_idx + 1 < 12:
                rem = 6 - (8 - bit_shift)
                val |= (scales[byte_idx + 1] & ((1 << rem) - 1)) << (8 - bit_shift)
            sc[i] = val
            bit_pos += 6
        for i in range(16):
            sc[i] |= ((scales_h >> (i * 2)) & 0x03) << 6
        for i in range(256):
            q = ((qs[i // 4] >> ((i % 4) * 2)) & 0x03) | (
                ((hmask[i // 8] >> (i % 8)) & 0x01) << 2
            )
            out.append(d * sc[i // 16] * (q - 4))
    return out


def f32(x: float) -> float:
    """Round to IEEE binary32, mirroring C `float` arithmetic per operation."""
    return struct.unpack('<f', struct.pack('<f', x))[0]


def dequant_q4_0(blocks):
    out = []
    for x in blocks:
        d = f16_to_f32(x[0] | (x[1] << 8))
        qs = x[2:18]
        out.extend(f32(((q & 0xF) - 8) * d) for q in qs)
        out.extend(f32(((q >> 4) - 8) * d) for q in qs)
    return out


def dequant_q4_1(blocks):
    out = []
    for x in blocks:
        d = f16_to_f32(x[0] | (x[1] << 8))
        m = f16_to_f32(x[2] | (x[3] << 8))
        qs = x[4:20]
        out.extend(f32(f32((q & 0xF) * d) + m) for q in qs)
        out.extend(f32(f32((q >> 4) * d) + m) for q in qs)
    return out


def dequant_q2_k(blocks):
    out = []
    for x in blocks:
        scales, qs = x[0:16], x[16:80]
        d = f16_to_f32(x[80] | (x[81] << 8))
        dmin = f16_to_f32(x[82] | (x[83] << 8))
        is_ = 0
        for n in range(0, 256, 128):
            q = qs[n // 4:n // 4 + 32]
            for shift in (0, 2, 4, 6):
                for half in (0, 16):
                    sc = scales[is_]
                    is_ += 1
                    dl = f32(d * (sc & 0xF))
                    ml = f32(dmin * (sc >> 4))
                    for l in range(16):
                        out.append(f32(f32(dl * ((q[half + l] >> shift) & 3)) - ml))
    return out


def dequant_q3_k(blocks):
    out = []
    for x in blocks:
        hmask, qs, packed = x[0:32], x[32:96], x[96:108]
        d_all = f16_to_f32(x[108] | (x[109] << 8))
        k1, k2 = 0x03030303, 0x0F0F0F0F
        a = list(struct.unpack('<3I', packed)) + [0]
        tmp = a[2]
        a[2] = ((a[0] >> 4) & k2) | (((tmp >> 4) & k1) << 4)
        a[3] = ((a[1] >> 4) & k2) | (((tmp >> 6) & k1) << 4)
        a[0] = (a[0] & k2) | (((tmp >> 0) & k1) << 4)
        a[1] = (a[1] & k2) | (((tmp >> 2) & k1) << 4)
        sc = [b - 256 if b >= 128 else b for b in struct.pack('<4I', *a)]
        is_, m = 0, 1
        for n in range(0, 256, 128):
            q = qs[n // 4:n // 4 + 32]
            for shift in (0, 2, 4, 6):
                for half in (0, 16):
                    dl = f32(d_all * (sc[is_] - 32))
                    is_ += 1
                    for l in range(16):
                        v = ((q[half + l] >> shift) & 3) - (0 if hmask[half + l] & m else 4)
                        out.append(f32(dl * v))
                m <<= 1
    return out


def dequant_q4_k(blocks):
    out = []
    for x in blocks:
        d = f16_to_f32(x[0] | (x[1] << 8))
        dmin = f16_to_f32(x[2] | (x[3] << 8))
        scales, qs = x[4:16], x[16:144]
        is_ = 0
        for j in range(0, 256, 64):
            q = qs[j // 2:j // 2 + 32]
            sc, m = get_scale_min_k4(is_, scales)
            d1, m1 = f32(d * sc), f32(dmin * m)
            sc, m = get_scale_min_k4(is_ + 1, scales)
            d2, m2 = f32(d * sc), f32(dmin * m)
            out.extend(f32(f32(d1 * (b & 0xF)) - m1) for b in q)
            out.extend(f32(f32(d2 * (b >> 4)) - m2) for b in q)
            is_ += 2
    return out


# New dtypes are appended so the shared PRNG stream (and therefore the
# existing fixtures) stay byte-identical when regenerated.
TYPES = {
    'q8_0': (34, 32, dequant_q8_0),
    'q5_k': (176, 256, dequant_q5_k),
    'q6_k': (210, 256, dequant_q6_k),
    'iq3_m': (111, 256, dequant_iq3_m),
    'q4_0': (18, 32, dequant_q4_0),
    'q4_1': (20, 32, dequant_q4_1),
    'q2_k': (84, 256, dequant_q2_k),
    'q3_k': (110, 256, dequant_q3_k),
    'q4_k': (144, 256, dequant_q4_k),
}

# Scale-field bit patterns exercised per dtype: one PRNG block plus blocks
# whose f16 scale field(s) cover zero, a denormal, and the max half value.
# q5_k has two f16 fields (d at 0, dmin at 2); both get the sweep.
SCALE_BITS = (0x0000, 0x0001, 0x7BFF)
SCALE_OFFSETS = {
    'q8_0': (0,),
    'q5_k': (0, 2),
    'q6_k': (208,),
    'iq3_m': (0,),
    'q4_0': (0,),
    'q4_1': (0, 2),
    'q2_k': (80, 82),
    'q3_k': (108,),
    'q4_k': (0, 2),
}


def main():
    rng = random.Random(0xE9)
    for name, (bsize, _n, fn) in TYPES.items():
        blocks = [
            bytes(rng.randrange(256) for _ in range(bsize))
            for _ in range(BLOCKS_PER_TYPE - len(SCALE_BITS) + 0)
        ]
        for dbits in SCALE_BITS:
            b = bytearray(rng.randrange(256) for _ in range(bsize))
            for off in SCALE_OFFSETS[name]:
                b[off] = dbits & 0xFF
                b[off + 1] = dbits >> 8
            blocks.append(bytes(b))
        packed = b''.join(blocks)
        expected = b''.join(struct.pack('<f', v) for v in fn(blocks))
        with open(os.path.join(OUT, f'{name}.packed.bin'), 'wb') as f:
            f.write(packed)
        with open(os.path.join(OUT, f'{name}.expected.bin'), 'wb') as f:
            f.write(expected)
        print(f'{name}: {len(blocks)} blocks, {len(expected) // 4} values')


if __name__ == '__main__':
    main()
