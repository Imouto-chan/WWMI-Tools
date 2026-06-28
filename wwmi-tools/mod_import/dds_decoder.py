"""
Pure-Python DDS decoder for texture formats Blender 3.6's bundled OIIO cannot load.

Blender 3.6's OpenImageIO build handles BC7_UNORM (DX10 header, format 98) but fails on:
  - BC3/DXT5 (legacy OR DX10 header)  -> IMB_ibImageFromMemory: unknown file-format
  - BC7_UNORM_SRGB (DX10, format 99)  -> same error
  - BC4/BC5 (ATI1/ATI2 FourCC)        -> same error

This module decodes BC1/BC3/BC4/BC5/BC7 entirely in pure Python (struct + zlib only),
writes the result as an uncompressed PNG, then lets the caller load that instead.

For formats Blender CAN load natively, decode_dds_to_rgba() returns None.
"""

import struct
import zlib


# ---------------------------------------------------------------------------
# DX10 header patching (kept for BC3 formats as a fast path backup)
# ---------------------------------------------------------------------------

_DXGI_TO_LEGACY_FOURCC = {
    70: b'DXT1', 71: b'DXT1', 72: b'DXT1',  # BC1
    73: b'DXT3', 74: b'DXT3', 75: b'DXT3',  # BC2
    76: b'DXT5', 77: b'DXT5', 78: b'DXT5',  # BC3
    79: b'BC4U', 80: b'ATI1', 81: b'BC4S',  # BC4
    82: b'BC5U', 83: b'ATI2', 84: b'BC5S',  # BC5
}

_DXGI_SRGB_TO_UNORM = {
    99: 98,   # BC7_UNORM_SRGB -> BC7_UNORM
    29: 28,   # R8G8B8A8_UNORM_SRGB -> R8G8B8A8_UNORM
    91: 87,   # B8G8R8A8_UNORM_SRGB -> B8G8R8A8_UNORM
}


def patch_dx10_to_legacy(data: bytes) -> bytes:
    """Normalise DDS header for Blender 3.6 OIIO compatibility."""
    if len(data) < 148 or data[:4] != b'DDS ':
        return data
    if data[84:88] != b'DX10':
        return data

    dxgi_format = struct.unpack_from('<I', data, 128)[0]

    legacy_fourcc = _DXGI_TO_LEGACY_FOURCC.get(dxgi_format)
    if legacy_fourcc is not None:
        header = bytearray(data[:128])
        header[84:88] = legacy_fourcc
        return bytes(header) + data[148:]

    unorm_format = _DXGI_SRGB_TO_UNORM.get(dxgi_format)
    if unorm_format is not None:
        patched = bytearray(data)
        struct.pack_into('<I', patched, 128, unorm_format)
        return bytes(patched)

    return data


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

# Formats we decode ourselves (returned by _read_dds_header as 'format' string)
_DXGI_TO_FMT = {
    # BC1
    70: 'bc1', 71: 'bc1', 72: 'bc1',
    # BC2 (rare; we decode as BC1 rgb + flat 0xFF alpha - good enough for preview)
    73: 'bc2', 74: 'bc2', 75: 'bc2',
    # BC3
    76: 'bc3', 77: 'bc3', 78: 'bc3',
    # BC4
    79: 'bc4u', 80: 'bc4u', 81: 'bc4s',
    # BC5
    82: 'bc5u', 83: 'bc5u', 84: 'bc5s',
    # BC7
    97: 'bc7', 98: 'bc7', 99: 'bc7',
}

_FOURCC_TO_FMT = {
    b'DXT1': 'bc1',
    b'DXT3': 'bc2',
    b'DXT5': 'bc3',
    b'ATI1': 'bc4u', b'BC4U': 'bc4u', b'3DC1': 'bc4u', b'BC4S': 'bc4s',
    b'ATI2': 'bc5u', b'BC5U': 'bc5u', b'3DC2': 'bc5u', b'A2XY': 'bc5u', b'BC5S': 'bc5s',
}


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


# ---------------------------------------------------------------------------
# BC1 (DXT1) RGB decoder
# ---------------------------------------------------------------------------

def _bc1_palette(c0_raw, c1_raw):
    def rgb565(v):
        return (((v >> 11) & 0x1F) * 255 // 31,
                ((v >>  5) & 0x3F) * 255 // 63,
                ( v        & 0x1F) * 255 // 31)
    e0, e1 = rgb565(c0_raw), rgb565(c1_raw)
    if c0_raw > c1_raw:
        c2 = tuple((2*e0[i] + e1[i] + 1) // 3 for i in range(3))
        c3 = tuple((e0[i] + 2*e1[i] + 1) // 3 for i in range(3))
    else:
        c2 = tuple((e0[i] + e1[i]) // 2 for i in range(3))
        c3 = (0, 0, 0)
    return [e0, e1, c2, c3]

def _decode_bc1_block_rgb(block):
    """Decode 8-byte BC1 block -> 16 (r,g,b) tuples."""
    c0, c1 = struct.unpack_from('<HH', block, 0)
    pal = _bc1_palette(c0, c1)
    bits = struct.unpack_from('<I', block, 4)[0]
    return [pal[(bits >> (2*i)) & 3] for i in range(16)]


# ---------------------------------------------------------------------------
# BC4 single-channel decoder (also used as alpha block in BC3)
# ---------------------------------------------------------------------------

def _bc4_palette_unorm(a0, a1):
    if a0 > a1:
        return [a0, a1,
                (6*a0+1*a1+3)//7, (5*a0+2*a1+3)//7,
                (4*a0+3*a1+3)//7, (3*a0+4*a1+3)//7,
                (2*a0+5*a1+3)//7, (1*a0+6*a1+3)//7]
    else:
        return [a0, a1,
                (4*a0+1*a1+2)//5, (3*a0+2*a1+2)//5,
                (2*a0+3*a1+2)//5, (1*a0+4*a1+2)//5,
                0, 255]

def _bc4_palette_snorm(a0, a1):
    # convert byte to signed
    def s(v): return v if v < 128 else v - 256
    a0, a1 = s(a0), s(a1)
    a0, a1 = _clamp(a0, -127, 127), _clamp(a1, -127, 127)
    if a0 > a1:
        raw = [a0,a1,(6*a0+a1)//7,(5*a0+2*a1)//7,(4*a0+3*a1)//7,(3*a0+4*a1)//7,(2*a0+5*a1)//7,(a0+6*a1)//7]
    else:
        raw = [a0,a1,(4*a0+a1)//5,(3*a0+2*a1)//5,(2*a0+3*a1)//5,(a0+4*a1)//5,-127,127]
    return [int(_clamp(v/127.0*127.0,-127,127)+128) for v in raw]

def _decode_bc4_block(block, signed=False):
    """8-byte BC4 -> 16 values 0-255."""
    pal = _bc4_palette_snorm(block[0], block[1]) if signed else _bc4_palette_unorm(block[0], block[1])
    bits = int.from_bytes(block[2:8], 'little')
    return [pal[(bits >> (3*i)) & 7] for i in range(16)]


# ---------------------------------------------------------------------------
# BC3 (DXT5) = BC1(RGB) + BC4(Alpha)
# ---------------------------------------------------------------------------

def _decode_bc3_block(block):
    """16-byte BC3 block -> 16 (r,g,b,a) tuples."""
    alpha_vals = _decode_bc4_block(block[0:8])
    rgb_vals   = _decode_bc1_block_rgb(block[8:16])
    return [(r, g, b, a) for (r, g, b), a in zip(rgb_vals, alpha_vals)]

def _decode_bc1_block_rgba(block):
    """8-byte BC1 block -> 16 (r,g,b,a=255) tuples."""
    return [(r, g, b, 255) for r, g, b in _decode_bc1_block_rgb(block)]

def _decode_bc2_block(block):
    """16-byte BC2 block -> 16 (r,g,b,a) tuples. Alpha is stored as 4-bit values."""
    # 8 bytes of alpha (4 bits per pixel, packed)
    alpha_bits = int.from_bytes(block[0:8], 'little')
    alphas = [((alpha_bits >> (4*i)) & 0xF) * 255 // 15 for i in range(16)]
    rgb_vals = _decode_bc1_block_rgb(block[8:16])
    return [(r, g, b, a) for (r, g, b), a in zip(rgb_vals, alphas)]


# ---------------------------------------------------------------------------
# BC7 full decoder (all 8 modes)
# ---------------------------------------------------------------------------

# Partition tables for 2-subset (64 entries) and 3-subset (64 entries)
_P2 = [
    [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],[0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1],
    [0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1],[0,0,0,1,0,0,1,1,0,0,1,1,0,1,1,1],
    [0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,1],[0,0,1,1,0,1,1,1,0,1,1,1,1,1,1,1],
    [0,0,0,1,0,0,1,1,0,1,1,1,1,1,1,1],[0,0,0,0,0,0,0,1,0,0,1,1,0,1,1,1],
    [0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0,0,1,1,0,1,1,0,1,1,0,0,1,0,0,0],
    [0,0,0,0,0,1,1,0,1,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1],
    [0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1],[0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
    [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1],[0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1],
    [0,0,0,0,1,0,0,0,1,1,1,0,1,1,1,1],[0,1,1,1,0,0,0,1,1,0,0,0,1,1,1,0],
    [0,0,0,1,0,1,0,0,1,1,0,0,1,1,1,0],[0,0,0,0,0,0,1,0,0,1,0,0,1,1,0,0],
    [0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0],[0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0],
    [0,0,1,1,1,1,0,0,1,1,0,0,0,0,0,0],[0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0],
    [0,1,1,0,1,1,0,0,1,0,0,0,0,0,0,0],[0,0,1,1,0,1,1,0,0,1,1,0,0,0,0,0],
    [0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0],[0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0],[0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0],
    [0,1,0,0,1,1,1,0,0,1,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,1,1,1,0,0,1,1,1],
    [0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,0],[0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0],
    [0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0],[0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0],
    [0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0],
    [0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0],[0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0],
    [0,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
    [0,1,0,0,1,1,0,0,1,0,0,0,0,0,0,0],[0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0],[0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0],
    [0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0],
    [0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0],[0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0],
    [0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0],[0,0,0,1,0,0,1,1,1,1,0,0,0,0,0,0],
    [0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0],[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1],
]

_P3 = [
    [0,0,1,1,0,0,1,1,0,2,2,1,2,2,2,2],[0,0,0,1,0,0,1,1,2,2,1,1,2,2,2,1],
    [0,0,0,0,2,0,0,1,2,2,1,1,2,2,1,1],[0,2,2,2,0,0,2,2,0,0,1,1,0,1,1,1],
    [0,0,0,2,0,0,0,1,0,0,1,1,0,1,2,2],[0,0,1,2,0,0,1,2,0,0,2,2,0,2,2,2],
    [0,2,2,0,0,2,2,0,1,1,2,2,1,2,2,1],[0,0,1,1,0,1,2,2,0,1,2,2,0,0,1,1],
    [0,0,0,0,0,0,0,0,1,1,2,2,1,1,2,2],[0,0,1,2,0,0,1,1,0,1,1,0,1,2,2,1],
    [0,1,2,0,0,1,2,0,0,1,2,0,0,1,2,0],[0,0,1,2,1,1,2,2,1,2,2,0,0,0,1,2],
    [0,1,1,0,1,2,2,1,1,2,2,1,0,1,1,0],[0,0,0,2,1,1,1,2,1,1,1,2,0,0,0,2],
    [0,1,1,0,0,1,1,0,2,0,0,2,2,2,2,2],[0,0,0,0,1,1,1,1,2,2,2,2,0,0,0,0],
    [0,0,2,2,0,0,2,2,0,0,2,2,0,0,2,2],[0,0,1,2,0,1,2,1,0,2,1,0,1,2,1,0],
    [0,0,0,0,0,0,0,0,2,1,2,1,2,1,2,1],[0,0,0,2,1,1,1,2,0,0,0,2,1,1,1,2],
    [0,0,1,1,0,1,2,2,0,1,2,2,0,0,1,1],[0,1,2,0,0,1,2,0,0,1,2,0,0,1,2,0],
    [0,0,0,0,1,1,1,1,0,0,0,0,2,2,2,2],[0,0,0,0,0,0,0,0,2,1,2,1,2,1,2,1],
    [0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0],[0,0,0,0,2,0,0,0,2,2,0,0,2,2,2,0],
    [0,0,0,2,0,0,1,2,0,0,1,2,0,0,0,2],[0,2,2,2,0,0,0,2,1,2,2,2,1,2,2,2],
    [0,0,0,0,2,2,0,0,2,2,0,0,2,2,0,0],[0,0,2,2,0,0,1,2,1,1,2,2,1,2,2,2],
    [0,2,2,0,1,2,2,1,0,2,2,0,1,2,2,1],[0,0,1,2,0,0,1,2,1,1,2,2,1,2,2,2],
    [0,1,2,0,0,1,2,0,1,2,0,1,2,0,1,2],[0,2,0,2,0,2,0,2,0,2,0,2,0,2,0,2],
    [0,0,0,0,2,2,2,2,1,1,1,1,0,0,0,0],[0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2],
    [0,2,2,2,0,1,1,1,0,2,2,2,0,1,1,1],[0,0,0,2,0,0,0,1,0,0,0,2,0,0,0,1],
    [0,2,2,2,1,2,2,2,0,2,2,2,1,1,1,1],[0,1,1,0,1,2,2,1,2,2,1,0,1,0,0,1],
    [0,0,1,2,1,2,2,0,2,2,0,1,2,0,1,2],[0,0,2,2,0,1,1,0,0,1,1,0,2,2,0,0],
    [0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0],[0,1,2,0,2,0,1,2,1,2,0,1,0,1,2,0],
    [0,2,0,2,1,1,1,1,0,2,0,2,1,1,1,1],[0,1,1,1,2,0,1,1,2,2,0,1,2,2,2,0],
    [0,1,0,1,2,2,0,1,2,2,0,1,2,2,0,1],[0,1,1,0,2,2,1,0,2,2,1,0,2,2,1,0],
    [0,0,1,1,0,0,1,1,2,2,1,1,2,2,1,1],[0,0,1,1,2,2,1,1,2,2,1,1,0,0,1,1],
    [0,2,2,0,0,2,2,0,0,2,2,0,0,2,2,0],[0,1,0,1,0,1,0,1,2,2,2,2,2,2,2,2],
    [0,2,2,2,0,1,1,1,0,2,2,2,0,1,1,1],[0,0,0,2,1,1,1,2,0,0,0,2,1,1,1,2],
    [0,1,0,1,2,2,0,1,2,2,0,1,2,2,0,1],[0,0,0,0,2,1,2,1,2,1,2,1,2,1,2,1],
    [0,0,0,2,0,0,0,1,2,2,2,0,2,2,2,0],[0,0,1,2,0,0,1,2,1,1,2,2,1,2,2,2],
    [0,2,2,0,0,2,2,0,0,2,2,0,0,2,2,0],[0,0,0,0,2,2,1,1,2,2,0,0,2,2,1,1],
    [0,0,2,2,1,2,2,0,1,1,2,2,1,2,2,0],[0,2,2,1,0,2,1,1,0,2,1,1,0,2,1,1],
    [0,2,2,0,0,2,2,1,0,2,2,1,0,2,2,1],[0,0,0,0,0,0,1,2,0,0,1,2,0,0,1,2],
]

_ANCHOR2  = [15,15,15,15,15,15,15,15,15,8,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
             2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
_ANCHOR3A = [3,3,15,15,8,3,15,15,8,8,6,6,6,5,3,3,3,3,8,15,3,3,6,10,5,8,8,6,8,5,15,15,
             8,15,15,3,15,5,15,15,15,15,15,15,3,15,5,5,5,15,14,21,1,2,3,3,6,7,8,9,10,11,15,15]
_ANCHOR3B = [15,8,8,3,15,15,3,8,15,15,15,15,15,15,15,8,15,8,15,3,15,8,15,8,3,15,6,10,15,
             15,10,8,15,3,15,10,10,8,9,10,6,15,8,15,3,6,6,8,15,3,15,15,15,15,15,15,15,15,15,15,3,15,15,8]

_BC7_W = [None, [0,64], [0,21,43,64],
          [0,9,18,27,37,46,55,64],
          [0,4,9,13,17,21,26,30,34,38,43,47,51,55,60,64]]

def _bc7_interp(e0, e1, idx, n):
    w = _BC7_W[n][idx]
    return ((64-w)*e0 + w*e1 + 32) >> 6

def _bc7_expand(v, b):
    if b >= 8: return v >> (b-8)
    return (v << (8-b)) | (v >> (2*b-8))

def _rb7(blk, off, n):
    return (blk >> off) & ((1 << n) - 1)

def _decode_bc7_block(block_bytes):
    """Decode one 16-byte BC7 block -> list of 16 (r,g,b,a) tuples."""
    lo = struct.unpack_from('<Q', block_bytes, 0)[0]
    hi = struct.unpack_from('<Q', block_bytes, 8)[0]
    blk = lo | (hi << 64)

    mode = next((m for m in range(8) if blk & (1 << m)), -1)
    if mode == -1:
        return [(0,0,0,255)] * 16

    # (ns, pb, rb, isb, cb, ab, epb, spb, ib, ib2)
    M = {0:(3,4,0,0,4,0,1,0,3,0), 1:(2,6,0,0,6,0,0,1,3,0),
         2:(3,6,0,0,5,0,0,0,2,0), 3:(2,6,0,0,7,0,1,0,2,0),
         4:(1,0,2,1,5,6,0,0,2,3), 5:(1,0,2,0,7,8,0,0,2,2),
         6:(1,0,0,0,7,7,1,0,4,0), 7:(2,6,0,0,5,5,1,0,2,0)}
    ns,pb,rb,isb,cb,ab,epb,spb,ib,ib2 = M[mode]
    bit = mode + 1

    part_id = _rb7(blk, bit, pb);   bit += pb
    rotation = _rb7(blk, bit, rb);  bit += rb
    index_sel = _rb7(blk, bit, isb); bit += isb

    n_ep = ns * 2
    R=[_rb7(blk,bit+i*cb,cb) for i in range(n_ep)]; bit+=n_ep*cb
    G=[_rb7(blk,bit+i*cb,cb) for i in range(n_ep)]; bit+=n_ep*cb
    B=[_rb7(blk,bit+i*cb,cb) for i in range(n_ep)]; bit+=n_ep*cb
    if ab>0:
        A=[_rb7(blk,bit+i*ab,ab) for i in range(n_ep)]; bit+=n_ep*ab
    else:
        A=[255]*n_ep

    if epb:
        P=[_rb7(blk,bit+i,1) for i in range(n_ep)]; bit+=n_ep
        for i in range(n_ep):
            R[i]=(R[i]<<1)|P[i]; G[i]=(G[i]<<1)|P[i]; B[i]=(B[i]<<1)|P[i]
            if ab>0: A[i]=(A[i]<<1)|P[i]
        cb+=1
        if ab>0: ab+=1
    if spb:
        P=[_rb7(blk,bit,1),_rb7(blk,bit+1,1)]; bit+=2
        for i in range(n_ep):
            R[i]=(R[i]<<1)|P[i//2]; G[i]=(G[i]<<1)|P[i//2]; B[i]=(B[i]<<1)|P[i//2]
        cb+=1

    R=[_bc7_expand(v,cb) for v in R]; G=[_bc7_expand(v,cb) for v in G]
    B=[_bc7_expand(v,cb) for v in B]
    if ab>0: A=[_bc7_expand(v,ab) for v in A]
    else: A=[255]*n_ep

    if ns==1:
        part=[0]*16; anchors=[0]
    elif ns==2:
        part=_P2[part_id%64]; anchors=[0,_ANCHOR2[part_id%64]]
    else:
        part=_P3[part_id%64]; anchors=[0,_ANCHOR3A[part_id%64],_ANCHOR3B[part_id%64]]

    idx1=[]
    for i in range(16):
        ss=part[i]; anc=anchors[ss] if ss<len(anchors) else 0
        bw=ib-(1 if i==anc else 0)
        idx1.append(_rb7(blk,bit,bw)); bit+=bw

    idx2=[]
    if ib2>0:
        anc0=anchors[0]
        for i in range(16):
            bw=ib2-(1 if i==anc0 else 0)
            idx2.append(_rb7(blk,bit,bw)); bit+=bw

    out=[]
    for i in range(16):
        ss=part[i]; e0=ss*2; e1=ss*2+1
        if mode==4 and index_sel==1:
            ci,ai,cb_,ab_=idx2[i] if idx2 else 0,idx1[i],ib2,ib
        elif mode==4:
            ci,ai,cb_,ab_=idx1[i],idx2[i] if idx2 else 0,ib,ib2
        elif mode==5:
            ci,ai,cb_,ab_=idx1[i],idx2[i] if idx2 else 0,ib,ib2 if ib2>0 else ib
        else:
            ci,ai,cb_,ab_=idx1[i],idx1[i],ib,ib

        r=_bc7_interp(R[e0],R[e1],ci,cb_)
        g=_bc7_interp(G[e0],G[e1],ci,cb_)
        b=_bc7_interp(B[e0],B[e1],ci,cb_)
        a=_bc7_interp(A[e0],A[e1],ai,ab_) if ab>0 else 255

        if rotation==1: r,a=a,r
        elif rotation==2: g,a=a,g
        elif rotation==3: b,a=a,b
        out.append((min(255,max(0,r)),min(255,max(0,g)),min(255,max(0,b)),min(255,max(0,a))))
    return out


# ---------------------------------------------------------------------------
# Header parser and main decode entry point
# ---------------------------------------------------------------------------

def _read_dds_header(data):
    """Parse DDS header. Returns dict or None."""
    if len(data) < 128 or data[0:4] != b'DDS ':
        return None
    height = struct.unpack_from('<I', data, 12)[0]
    width  = struct.unpack_from('<I', data, 16)[0]
    pf_fourcc = bytes(data[84:88])
    data_offset = 128
    dxgi_format = None
    if pf_fourcc == b'DX10':
        if len(data) < 148:
            return None
        dxgi_format = struct.unpack_from('<I', data, 128)[0]
        data_offset = 148
    fmt = _DXGI_TO_FMT.get(dxgi_format) if dxgi_format is not None else _FOURCC_TO_FMT.get(pf_fourcc)
    if fmt is None:
        return None
    return {'width': width, 'height': height, 'format': fmt, 'data_offset': data_offset}


def decode_dds_to_rgba(path):
    """
    Decode a BC1/BC2/BC3/BC4/BC5/BC7 DDS file into RGBA8 bytes.
    Returns (width, height, rgba_bytes) on success, None if format not handled.
    """
    with open(path, 'rb') as f:
        data = f.read()

    info = _read_dds_header(data)
    if info is None:
        return None

    width, height = info['width'], info['height']
    fmt, pos = info['format'], info['data_offset']
    if width <= 0 or height <= 0:
        return None

    is_bc1 = fmt == 'bc1'
    is_bc2 = fmt == 'bc2'
    is_bc3 = fmt == 'bc3'
    is_bc7 = fmt == 'bc7'
    is_bc4 = fmt in ('bc4u', 'bc4s')
    is_bc5 = fmt in ('bc5u', 'bc5s')
    signed = fmt.endswith('s')

    if is_bc7:
        block_size = 16
    elif is_bc3 or is_bc2 or is_bc5:
        block_size = 16
    else:
        block_size = 8  # BC1, BC4

    blocks_x = (width + 3) // 4
    blocks_y = (height + 3) // 4
    needed = pos + blocks_x * blocks_y * block_size
    if len(data) < needed:
        return None

    rgba = bytearray(width * height * 4)

    for by in range(blocks_y):
        for bx in range(blocks_x):
            blk = data[pos:pos+block_size]
            pos += block_size

            if is_bc7:
                pixels = _decode_bc7_block(blk)
            elif is_bc3:
                pixels = _decode_bc3_block(blk)
            elif is_bc2:
                pixels = _decode_bc2_block(blk)
            elif is_bc1:
                pixels = _decode_bc1_block_rgba(blk)
            elif is_bc5:
                ch0 = _decode_bc4_block(blk[0:8], signed)
                ch1 = _decode_bc4_block(blk[8:16], signed)
                pixels = []
                for v in range(16):
                    r, g = ch0[v], ch1[v]
                    nx = r/255.0*2.0-1.0; ny = g/255.0*2.0-1.0
                    nz_sq = 1.0-nx*nx-ny*ny
                    nz = nz_sq**0.5 if nz_sq > 0.0 else 0.0
                    b = int(_clamp((nz*0.5+0.5)*255.0, 0, 255))
                    pixels.append((r, g, b, 255))
            else:  # bc4
                vals = _decode_bc4_block(blk, signed)
                pixels = [(v, v, v, 255) for v in vals]

            base_y = by * 4; base_x = bx * 4
            for ty in range(min(4, height-base_y)):
                row_off = (base_y+ty) * width
                for tx in range(min(4, width-base_x)):
                    r,g,b,a = pixels[ty*4+tx]
                    off = (row_off + base_x+tx)*4
                    rgba[off]=r; rgba[off+1]=g; rgba[off+2]=b; rgba[off+3]=a

    return width, height, bytes(rgba)


def write_png(path, width, height, rgba_bytes):
    """Write RGBA8 as PNG using only struct+zlib (no PIL dependency)."""
    def _chunk(tag, payload):
        return (struct.pack('>I', len(payload)) + tag + payload +
                struct.pack('>I', zlib.crc32(tag+payload) & 0xFFFFFFFF))
    sig = b'\x89PNG\r\n\x1a\n'
    ihdr = struct.pack('>IIBBBBB', width, height, 8, 6, 0, 0, 0)
    stride = width * 4
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        raw.extend(rgba_bytes[y*stride:(y+1)*stride])
    with open(path, 'wb') as f:
        f.write(sig)
        f.write(_chunk(b'IHDR', ihdr))
        f.write(_chunk(b'IDAT', zlib.compress(bytes(raw), 6)))
        f.write(_chunk(b'IEND', b''))
