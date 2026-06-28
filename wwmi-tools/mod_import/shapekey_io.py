"""
shapekey_io.py - Utility to parse WuWa mod shapekey buffers and produce
per-component .shapekeys files, plus the corresponding loader for use in
blender_import.py.

File format for <Component N>.shapekeys:
  Header (4 bytes):
    uint32  shapekey_count   - number of shapekeys in this component

  Per shapekey entry (8 + num_verts * 12 bytes each):
    uint32  global_sk_id     - original shapekey ID in the global array
    uint32  num_verts        - vertex count for this component
    float32[num_verts * 3]   - per-vertex XYZ position offsets (0.0 for unaffected verts)

Reading back yields a dict {global_sk_id: ndarray(num_verts, 3)}.
"""

import struct
import numpy as np
from pathlib import Path


def parse_shapekeys_from_mod(meshes_dir: Path):
    """
    Parse the three shapekey buffers from a mod's Meshes/ folder.

    Returns a tuple:
        (shapekey_count, total_entries, offsets, vids, voffs)

    Where:
        shapekey_count  int
        total_entries   int
        offsets         tuple of 128 uint32 (shapekey entry-start table)
        vids            tuple of uint32 (global vertex IDs, length total_entries)
        voffs           np.ndarray shape (total_entries, 6) float32
                        - first 3 cols are XYZ position offsets (float16 decoded)
    """
    sk_off_path = meshes_dir / 'ShapeKeyOffset.buf'
    sk_vid_path = meshes_dir / 'ShapeKeyVertexId.buf'
    sk_voff_path = meshes_dir / 'ShapeKeyVertexOffset.buf'

    if not sk_off_path.exists():
        return None

    with open(sk_off_path, 'rb') as f:
        raw = f.read()

    # ShapeKeyOffset.buf is exactly 512 bytes (128 uint32s) - the CS constant
    # buffer slice that holds the per-shapekey first-entry offsets.
    if len(raw) < 512:
        return None
    offsets = struct.unpack('<128I', raw[:512])
    total_entries = offsets[-1]

    if total_entries == 0:
        return None

    # Shapekey count: first index where offset value equals total_entries
    shapekey_count = 0
    for i, v in enumerate(offsets):
        if v >= total_entries:
            shapekey_count = i
            break

    if shapekey_count == 0:
        return None

    with open(sk_vid_path, 'rb') as f:
        vids_raw = f.read()
    vids = struct.unpack(f'<{total_entries}I', vids_raw[:total_entries * 4])

    with open(sk_voff_path, 'rb') as f:
        voffs_raw = f.read()
    # 6 float16 per entry: [dx, dy, dz, dnx, dny, dnz] - we use only first 3
    voffs = np.frombuffer(voffs_raw, dtype=np.float16).astype(np.float32)
    voffs = voffs[:total_entries * 6].reshape(total_entries, 6)

    return shapekey_count, total_entries, offsets, vids, voffs


def build_component_shapekeys(shapekey_count, offsets, vids, voffs,
                               vmin: int, vmax: int):
    """
    Extract per-component shapekey arrays for a component covering
    global vertex range [vmin, vmax].

    Returns dict {global_sk_id: np.ndarray(shape=(num_verts, 3), dtype=float32)}
    or empty dict if no shapekeys affect this component.
    """
    num_verts = vmax - vmin + 1

    # First pass: which shapekeys touch this component?
    active_sks = []
    for sk_id in range(shapekey_count):
        first = offsets[sk_id]
        nxt = offsets[sk_id + 1]
        for eid in range(first, nxt):
            if vmin <= vids[eid] <= vmax:
                active_sks.append(sk_id)
                break

    if not active_sks:
        return {}

    # Second pass: build (num_verts, 3) arrays
    sk_data = {}
    for sk_id in active_sks:
        arr = np.zeros((num_verts, 3), dtype=np.float32)
        first = offsets[sk_id]
        nxt = offsets[sk_id + 1]
        for eid in range(first, nxt):
            vid = vids[eid]
            if vmin <= vid <= vmax:
                arr[vid - vmin] = voffs[eid, :3]
        sk_data[sk_id] = arr

    return sk_data


def write_shapekeys_file(path: Path, sk_data: dict, num_verts: int):
    """
    Serialize sk_data {global_sk_id -> (num_verts, 3) float32 array} to path.
    """
    import io
    buf = io.BytesIO()

    count = len(sk_data)
    buf.write(struct.pack('<I', count))

    for global_sk_id, arr in sorted(sk_data.items()):
        assert arr.shape == (num_verts, 3), \
            f"Expected ({num_verts}, 3), got {arr.shape}"
        buf.write(struct.pack('<II', global_sk_id, num_verts))
        buf.write(arr.astype(np.float32).tobytes())

    with open(path, 'wb') as f:
        f.write(buf.getvalue())


def read_shapekeys_file(path: Path):
    """
    Load a .shapekeys file.

    Returns dict {global_sk_id: np.ndarray(num_verts, 3)} or {} if not found.
    """
    if not path.exists():
        return {}

    with open(path, 'rb') as f:
        raw = f.read()

    offset = 0
    (count,) = struct.unpack_from('<I', raw, offset)
    offset += 4

    result = {}
    for _ in range(count):
        global_sk_id, num_verts = struct.unpack_from('<II', raw, offset)
        offset += 8
        arr = np.frombuffer(raw, dtype=np.float32,
                            count=num_verts * 3, offset=offset).copy()
        arr = arr.reshape(num_verts, 3)
        offset += num_verts * 3 * 4
        result[global_sk_id] = arr

    return result
