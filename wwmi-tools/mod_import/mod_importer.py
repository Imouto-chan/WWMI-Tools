"""
WWMI Tools - Mod Import Preparation
Converts a packed mod folder (with separate .buf files) into a flat
Import-Object-compatible directory with properly split per-component
.vb / .ib / .fmt / Metadata.json files.

Also handles:
  - Per-component bone index remapping (global -> local, with cross-component refs)
  - Bind-pose VB reconstruction from separate mod .buf files
  - IB conversion from R32_UINT -> R16_UINT with 0-based remapping
"""

import struct
import json
import os
import shutil
import re
from pathlib import Path

try:
    from . import dds_decoder
    from .shapekey_io import (parse_shapekeys_from_mod,
                               build_component_shapekeys,
                               write_shapekeys_file)
except ImportError:
    # Fallback for standalone/test execution outside the Blender addon package
    import dds_decoder
    from shapekey_io import (parse_shapekeys_from_mod,
                              build_component_shapekeys,
                              write_shapekeys_file)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(path):
    with open(path, "rb") as f:
        return f.read()


def _parse_ini_components(ini_path):
    """
    Parse a mod.ini and return a list of component dicts with keys:
      id, match_first, match_count, draws [(count, start)], vg_offset, vg_count
    Uses drawindexed entries as the true IB layout map.
    """
    with open(ini_path, "r", encoding="utf-8-sig") as f:
        text = f.read()

    components = []
    comp_blocks = re.split(r'\[TextureOverrideComponent(\d+)\]', text)

    for i in range(1, len(comp_blocks), 2):
        comp_id = int(comp_blocks[i])
        block = comp_blocks[i + 1]

        mfi = re.search(r'match_first_index\s*=\s*(\d+)', block)
        mic = re.search(r'match_index_count\s*=\s*(\d+)', block)
        vgo = re.search(r'\$\\\\WWMIv1\\\\vg_offset\s*=\s*(\d+)', block)
        vgc = re.search(r'\$\\\\WWMIv1\\\\vg_count\s*=\s*(\d+)', block)

        draws_raw = re.findall(r'drawindexed\s*=\s*(\d+)\s*,\s*(\d+)', block)
        draws = [(int(c), int(s)) for c, s in draws_raw]

        if not (mfi and mic and draws):
            continue

        components.append({
            "id":           comp_id,
            "match_first":  int(mfi.group(1)),
            "match_count":  int(mic.group(1)),
            "draws":        draws,
            "vg_offset":    int(vgo.group(1)) if vgo else 0,
            "vg_count":     int(vgc.group(1)) if vgc else 0,
        })

    components.sort(key=lambda c: c["id"])
    return components


def _build_vertex_ranges(components, indices_all):
    """
    For each component, determine the actual vertex min/max from the
    drawindexed-based index slices (not match_first_index which refers
    to the vanilla global IB).
    """
    for comp in components:
        all_idx = []
        for draw_count, draw_start in comp["draws"]:
            end = draw_start + draw_count
            if end <= len(indices_all):
                all_idx.extend(indices_all[draw_start:end])

        if all_idx:
            comp["vmin"] = min(all_idx)
            comp["vmax"] = max(all_idx)
        else:
            comp["vmin"] = 0
            comp["vmax"] = 0

    return components


def _remap_bones(components, blend_data, blend_stride=8):
    """
    For each component, find all global bone indices actually used by its
    vertices (including cross-component refs), build global->local map,
    and store the full ordered bone list and inverse map.
    """
    for comp in components:
        vmin, vmax = comp["vmin"], comp["vmax"]
        vg_offset = comp["vg_offset"]
        vg_count  = comp["vg_count"]

        global_bones_used = set()
        for vi in range(vmin, vmax + 1):
            off = vi * blend_stride
            idxs    = struct.unpack("4B", blend_data[off:off+4])
            weights = struct.unpack("4B", blend_data[off+4:off+8])
            for bi, bw in zip(idxs, weights):
                if bw > 0:
                    global_bones_used.add(bi)

        primary = sorted(b for b in global_bones_used
                         if vg_offset <= b < vg_offset + vg_count)
        cross   = sorted(b for b in global_bones_used if b not in primary)
        ordered = primary + cross

        # Guard: fully-unweighted/rigid components have all blend weights = 0,
        # so global_bones_used is empty and ordered = [].  The WWMI importer
        # builds vg_remap from vg_map and then indexes every vertex's blend
        # indices through it - even for zero-weight vertices it still does
        # vg_remap[blend_index_0].  An empty vg_remap causes IndexError.
        # Fix: always ensure at least local index 0 -> global bone 0 exists,
        # which gives the importer a valid mapping for the zero-weight case.
        if not ordered:
            ordered = [vg_offset if vg_count > 0 else 0]

        comp["ordered_bones"]   = ordered
        comp["global_to_local"] = {g: l for l, g in enumerate(ordered)}
        comp["actual_vg_count"] = len(ordered)
        comp["vg_map"]          = {str(l): g for l, g in enumerate(ordered)}

    return components


def _build_vb(comp, pos_data, vec_data, blend_data, color_data, tex_data):
    """
    Interleave per-semantic buffers into a stride-48 .vb for one component,
    remapping blend indices from global to local.
    """
    POS   = 12
    VEC   = 8
    BLEND = 8
    COL   = 4
    TEX   = 16
    STRIDE = POS + VEC + BLEND + COL + TEX  # 48

    vmin = comp["vmin"]
    vmax = comp["vmax"]
    g2l  = comp["global_to_local"]

    vb = bytearray()
    for vi in range(vmin, vmax + 1):
        vb += pos_data  [vi * POS   : (vi+1) * POS]
        vb += vec_data  [vi * VEC   : (vi+1) * VEC]

        b_off = vi * BLEND
        g_idx = struct.unpack("4B", blend_data[b_off:b_off+4])
        wts   = blend_data[b_off+4:b_off+8]
        l_idx = tuple(
            g2l[gi] if struct.unpack("B", wts[i:i+1])[0] > 0 else 0
            for i, gi in enumerate(g_idx)
        )
        vb += bytes(l_idx) + bytes(wts)

        vb += color_data[vi * COL   : (vi+1) * COL]
        vb += tex_data  [vi * TEX   : (vi+1) * TEX]

    return bytes(vb)


def _build_ib(comp, indices_all):
    """
    Slice and remap the index buffer for one component to 0-based R16_UINT.
    """
    vmin = comp["vmin"]
    ib_indices = []
    for draw_count, draw_start in comp["draws"]:
        ib_indices.extend(indices_all[draw_start:draw_start + draw_count])

    remapped = [i - vmin for i in ib_indices]
    if remapped and max(remapped) > 65535:
        raise ValueError(
            f"Component {comp['id']}: max remapped index {max(remapped)} "
            f"exceeds R16_UINT limit (65535). Cannot use 16-bit IB."
        )
    return struct.pack(f"<{len(remapped)}H", *remapped)


FMT_TEMPLATE = """\
stride: 48
topology: trianglelist
format: DXGI_FORMAT_R16_UINT
element[0]:
  SemanticName: POSITION
  SemanticIndex: 0
  Format: R32G32B32_FLOAT
  InputSlot: 0
  AlignedByteOffset: 0
  InputSlotClass: per-vertex
  InstanceDataStepRate: 0
element[1]:
  SemanticName: TANGENT
  SemanticIndex: 0
  Format: R8G8B8A8_SNORM
  InputSlot: 0
  AlignedByteOffset: 12
  InputSlotClass: per-vertex
  InstanceDataStepRate: 0
element[2]:
  SemanticName: NORMAL
  SemanticIndex: 0
  Format: R8G8B8A8_SNORM
  InputSlot: 0
  AlignedByteOffset: 16
  InputSlotClass: per-vertex
  InstanceDataStepRate: 0
element[3]:
  SemanticName: BLENDINDICES
  SemanticIndex: 0
  Format: R8G8B8A8_UINT
  InputSlot: 0
  AlignedByteOffset: 20
  InputSlotClass: per-vertex
  InstanceDataStepRate: 0
element[4]:
  SemanticName: BLENDWEIGHT
  SemanticIndex: 0
  Format: R8G8B8A8_UNORM
  InputSlot: 0
  AlignedByteOffset: 24
  InputSlotClass: per-vertex
  InstanceDataStepRate: 0
element[5]:
  SemanticName: COLOR
  SemanticIndex: 0
  Format: R8G8B8A8_UNORM
  InputSlot: 0
  AlignedByteOffset: 28
  InputSlotClass: per-vertex
  InstanceDataStepRate: 0
element[6]:
  SemanticName: TEXCOORD
  SemanticIndex: 0
  Format: R16G16_FLOAT
  InputSlot: 0
  AlignedByteOffset: 32
  InputSlotClass: per-vertex
  InstanceDataStepRate: 0
element[7]:
  SemanticName: COLOR
  SemanticIndex: 1
  Format: R16G16_UNORM
  InputSlot: 0
  AlignedByteOffset: 36
  InputSlotClass: per-vertex
  InstanceDataStepRate: 0
element[8]:
  SemanticName: TEXCOORD
  SemanticIndex: 1
  Format: R16G16_FLOAT
  InputSlot: 0
  AlignedByteOffset: 40
  InputSlotClass: per-vertex
  InstanceDataStepRate: 0
element[9]:
  SemanticName: TEXCOORD
  SemanticIndex: 2
  Format: R16G16_FLOAT
  InputSlot: 0
  AlignedByteOffset: 44
  InputSlotClass: per-vertex
  InstanceDataStepRate: 0
"""


def prepare_mod_for_import(mod_folder, output_folder, progress_cb=None):
    """
    Main entry point.

    mod_folder   : path to mod root (containing mod.ini and Meshes/)
    output_folder: where to write the flat Import-Object directory
    progress_cb  : optional callable(message: str) for UI feedback

    Returns a dict with keys:
      success      : bool
      message      : summary string
      components   : list of component summary dicts
      output_path  : Path to output folder
    """
    mod_folder   = Path(mod_folder)
    output_folder = Path(output_folder)

    def log(msg):
        if progress_cb:
            progress_cb(msg)
        print(f"[ModImport] {msg}")

    # --- Locate mod.ini ---
    ini_path = mod_folder / "mod.ini"
    if not ini_path.exists():
        raise FileNotFoundError(f"mod.ini not found in: {mod_folder}")

    # --- Locate Meshes folder ---
    meshes_dir = mod_folder / "Meshes"
    if not meshes_dir.exists():
        raise FileNotFoundError(f"Meshes/ folder not found in: {mod_folder}")

    # --- Load all source buffers ---
    log("Loading source buffers...")
    required = ["Index.buf", "Position.buf", "Vector.buf",
                "Blend.buf", "Color.buf", "TexCoord.buf"]
    for r in required:
        if not (meshes_dir / r).exists():
            raise FileNotFoundError(f"Required buffer missing: Meshes/{r}")

    idx_data   = _load(meshes_dir / "Index.buf")
    pos_data   = _load(meshes_dir / "Position.buf")
    vec_data   = _load(meshes_dir / "Vector.buf")
    blend_data = _load(meshes_dir / "Blend.buf")
    color_data = _load(meshes_dir / "Color.buf")
    tex_data   = _load(meshes_dir / "TexCoord.buf")

    num_indices = len(idx_data) // 4
    indices_all = struct.unpack(f"<{num_indices}I", idx_data)
    total_verts = len(pos_data) // 12

    log(f"Loaded: {total_verts} vertices, {num_indices} indices ({num_indices//3} triangles)")

    # --- Load shapekey buffers (optional - mods without facial animation omit them) ---
    sk_parsed = parse_shapekeys_from_mod(meshes_dir)
    if sk_parsed:
        sk_count, sk_entries, sk_offsets, sk_vids, sk_voffs = sk_parsed
        log(f"Loaded shapekeys: {sk_count} keys, {sk_entries} vertex entries")
    else:
        sk_count = 0
        log("No shapekey buffers found (ShapeKeyOffset.buf missing or empty)")

    # --- Parse mod.ini ---
    log("Parsing mod.ini component definitions...")
    components = _parse_ini_components(ini_path)
    if not components:
        raise ValueError("No TextureOverrideComponent sections found in mod.ini")
    log(f"Found {len(components)} components")

    # --- Determine per-component vertex ranges ---
    components = _build_vertex_ranges(components, indices_all)

    # --- Remap bone indices ---
    log("Analysing bone index cross-references...")
    components = _remap_bones(components, blend_data)

    # --- Extract vb0_hash from mod.ini ---
    with open(ini_path, "r", encoding="utf-8-sig") as f:
        ini_text = f.read()
    hash_match = re.search(r'\[TextureOverrideComponent0\][^\[]*hash\s*=\s*([0-9a-fA-F]+)', ini_text)
    vb0_hash = hash_match.group(1) if hash_match else "unknown"

    # --- Copy/check for a fmt file to use as template ---
    # We use our built-in FMT_TEMPLATE since we know the exact interleaved layout
    fmt_content = FMT_TEMPLATE

    # --- Build output ---
    output_folder.mkdir(parents=True, exist_ok=True)
    all_comp_meta = []
    summaries = []

    for comp in components:
        cid      = comp["id"]
        vmin     = comp["vmin"]
        vmax     = comp["vmax"]
        num_verts = vmax - vmin + 1
        total_ib  = sum(c for c, s in comp["draws"])

        log(f"  Component {cid}: verts {vmin}–{vmax} ({num_verts} verts, "
            f"{total_ib//3} tris, {comp['actual_vg_count']} bone groups)")

        # .vb
        vb_data = _build_vb(comp, pos_data, vec_data, blend_data, color_data, tex_data)
        with open(output_folder / f"Component {cid}.vb", "wb") as f:
            f.write(vb_data)

        # .ib
        ib_data = _build_ib(comp, indices_all)
        with open(output_folder / f"Component {cid}.ib", "wb") as f:
            f.write(ib_data)

        # .fmt
        with open(output_folder / f"Component {cid}.fmt", "w") as f:
            f.write(fmt_content)

        # .shapekeys (only if this component has shapekeys)
        comp_sk_count = 0
        if sk_count > 0:
            sk_data = build_component_shapekeys(
                sk_count, sk_offsets, sk_vids, sk_voffs, vmin, vmax)
            if sk_data:
                comp_sk_count = len(sk_data)
                write_shapekeys_file(
                    output_folder / f"Component {cid}.shapekeys",
                    sk_data,
                    num_verts,
                )
                log(f"    -> {comp_sk_count} shapekeys written")

        all_comp_meta.append({
            "vertex_offset": 0,
            "vertex_count":  num_verts,
            "index_offset":  0,
            "index_count":   total_ib,
            "vg_offset":     comp["vg_offset"],
            "vg_count":      comp["actual_vg_count"],
            "vg_map":        comp["vg_map"],
        })

        summaries.append({
            "id":         cid,
            "verts":      num_verts,
            "tris":       total_ib // 3,
            "bone_groups": comp["actual_vg_count"],
            "cross_refs": len([b for b in comp["ordered_bones"]
                               if not (comp["vg_offset"] <= b < comp["vg_offset"] + comp["vg_count"])]),
        })

    # --- Combined Metadata.json ---
    # Must exactly match the ExtractedObject dataclass schema in metadata_format.py
    meta = {
        "vb0_hash":    vb0_hash,
        "cb4_hash":    None,
        "vertex_count": total_verts,
        "index_count":  num_indices,
        "components":   all_comp_meta,
        "shapekeys": {
            "offsets_hash": "",
            "scale_hash": "",
            "vertex_ids_hash": "",
            "vertex_offsets_hash": "",
            "vertex_count": 0,
            "shapekey_count": 0,
            "batches": [],
            "dispatch_y": 0,
            "checksum": 0
        },
        "export_format": {
            "Index": {"semantics": [
                {"name": "INDEX", "index": 0, "format": "R32_UINT", "stride": 12}
            ]},
            "Position": {"semantics": [
                {"name": "POSITION", "index": 0, "format": "R32G32B32_FLOAT", "stride": 12}
            ]},
            "Blend": {"semantics": [
                {"name": "BLENDINDICES", "index": 0, "format": "R8_UINT", "stride": 4},
                {"name": "BLENDWEIGHT",  "index": 0, "format": "R8_UINT", "stride": 4}
            ]},
            "Vector": {"semantics": [
                {"name": "TANGENT",       "index": 0, "format": "R8G8B8A8_SNORM", "stride": 4},
                {"name": "NORMAL",        "index": 0, "format": "R8G8B8_SNORM",   "stride": 3},
                {"name": "BITANGENTSIGN", "index": 0, "format": "R8_SNORM",       "stride": 1}
            ]},
            "Color": {"semantics": [
                {"name": "COLOR", "index": 0, "format": "R8G8B8A8_UNORM", "stride": 4}
            ]},
            "TexCoord": {"semantics": [
                {"name": "TEXCOORD", "index": 0, "format": "R16G16_FLOAT",  "stride": 4},
                {"name": "COLOR",    "index": 1, "format": "R16G16_UNORM",  "stride": 4},
                {"name": "TEXCOORD", "index": 1, "format": "R16G16_FLOAT",  "stride": 4},
                {"name": "TEXCOORD", "index": 2, "format": "R16G16_FLOAT",  "stride": 4}
            ]},
            "ShapeKeyOffset": {"semantics": [
                {"name": "SHAPEKEY", "index": 0, "format": "R32G32B32A32_UINT", "stride": 16}
            ]},
            "ShapeKeyVertexId": {"semantics": [
                {"name": "SHAPEKEY", "index": 1, "format": "R32_UINT", "stride": 4}
            ]},
            "ShapeKeyVertexOffset": {"semantics": [
                {"name": "SHAPEKEY", "index": 2, "format": "R16_FLOAT", "stride": 2}
            ]}
        },
    }
    with open(output_folder / "Metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # --- Copy textures from mod Textures/ folder ---
    tex_src = mod_folder / "Textures"
    if tex_src.exists():
        copied = 0
        for dds in tex_src.glob("*.dds"):
            shutil.copy(dds, output_folder / dds.name)
            copied += 1
        log(f"Copied {copied} texture(s) from mod Textures/")

    # --- Parse and save texture map ---
    log("Parsing texture assignments from mod.ini...")
    tex_map = _parse_component_textures(ini_path, mod_folder)

    # Inherit textures for components that have no texture entries.
    # This happens when the mod author only names textures after the first
    # component (e.g. "Components-0 t=xxxx.dds") even though multiple
    # components share those textures.  We find the nearest component that
    # DOES have textures and copy its entry list, so every imported component
    # gets a material with the correct texture nodes.
    all_comp_ids = sorted([c["id"] for c in components])
    for cid in all_comp_ids:
        key = str(cid)
        if not tex_map.get(key):
            # Find nearest component with textures (prefer lower IDs first)
            donor_key = None
            for other in sorted(tex_map.keys(), key=lambda k: abs(int(k) - cid)):
                if tex_map[other]:
                    donor_key = other
                    break
            if donor_key is not None:
                tex_map[key] = tex_map[donor_key]
                log(f"Component {key}: no textures in ini - inheriting from component {donor_key}")

    tex_map_path = output_folder / "texture_map.json"
    with open(tex_map_path, "w") as f:
        json.dump(tex_map, f, indent=2)
    tex_count = sum(len(v) for v in tex_map.values())
    log(f"Saved texture map: {tex_count} texture assignment(s) across "
        f"{len([k for k,v in tex_map.items() if v])} component(s)")

    msg = (f"Done! {len(components)} components written to:\n{output_folder}\n"
           f"Now use Import Object and point it at this folder.")
    log(msg)

    return {
        "success":     True,
        "message":     msg,
        "components":  summaries,
        "output_path": output_folder,
        "texture_map": tex_map,
    }


# ---------------------------------------------------------------------------
# Texture map parsing
# ---------------------------------------------------------------------------

def _parse_component_textures(ini_path, mod_folder):
    """
    Parse a mod.ini and build a component_id -> [entry, ...] map where each
    entry is a dict:
        { 'path': str, 'res_id': int, 'share_count': int, 'override_index': int }

    res_id         = the [ResourceTextureN] index (authoring order).
    share_count    = how many components reference this texture (1 = exclusive).
    override_index = the [TextureOverrideTextureN] index - this encodes the TRUE
                     ps-t slot ordering within the shader.  Textures are sorted
                     by their per-component override_index position so the
                     Blender node labels reflect the real shader binding order.

    This ini format does NOT assign textures via ps-t slots in the
    TextureOverrideComponent blocks. Instead every texture is a standalone
    [ResourceTextureN] whose filename encodes which components share it:
        Textures/Components-0 t=xxxx.dds         -> component 0 only (share_count=1)
        Textures/Components-0-2-3-6 t=xxxx.dds   -> 4 components    (share_count=4)
        Textures/FaceLightMap t=xxxx.dds          -> global, ignored
    """
    mod_folder = Path(mod_folder)
    with open(ini_path, "r", encoding="utf-8-sig") as f:
        text = f.read()

    # Step 1: build ResourceTextureN -> (absolute_path, filename_hash)
    resource_map = {}   # res_id -> (abs_path, fname_hash)
    for m in re.finditer(
        r'\[ResourceTexture(\d+)\][^\[]*?filename\s*=\s*([^\n]+)',
        text
    ):
        res_id   = int(m.group(1))
        filename = m.group(2).strip()
        abs_path = mod_folder / filename
        fname_hash_m = re.search(r't=([0-9a-f]+)', filename)
        fname_hash = fname_hash_m.group(1) if fname_hash_m else None
        if abs_path.exists():
            resource_map[res_id] = (abs_path, fname_hash)

    header_count = len(re.findall(r'\[ResourceTexture\d+\]', text))
    print(f"[TexParse] {len(resource_map)}/{header_count} ResourceTexture entries resolved")

    # Step 2: parse TextureOverrideTextureN blocks to get the true per-component
    #         ps-t slot ordering.  The override_index (N in TextureOverrideTextureN)
    #         is authored in ascending shader slot order for the whole character.
    #         For each component, the relative position of its textures within this
    #         global ordering IS the ps-t slot assignment.
    # Also detect "replaced" textures: override game_hash != filename hash means
    # the mod author actually changed the content (new skin/outfit), vs. same hash
    # means the vanilla texture is re-declared unchanged (lightmap, spec, etc.).
    override_order = {}  # game_hash -> (override_index, res_id)
    for m in re.finditer(r'\[TextureOverrideTexture(\d+)\](.*?)(?=\[|\Z)', text, re.DOTALL):
        oi = int(m.group(1))
        block = m.group(2)
        hm = re.search(r'hash\s*=\s*([0-9a-f]+)', block)
        rm = re.search(r'this\s*=\s*ResourceTexture(\d+)', block)
        if hm and rm:
            override_order[hm.group(1)] = (oi, int(rm.group(1)))

    # Step 3: parse component membership from filename, build per-component lists
    #         keyed by (override_index) for correct slot ordering.
    comp_textures = {}   # str(comp_id) -> list of (override_index, res_id, share_count, path, is_replaced)
    for res_id, (abs_path, fname_hash) in resource_map.items():
        stem      = abs_path.stem          # "Components-0-2-3-6 t=a260e7f7"
        name_part = stem.split(' t=')[0]   # "Components-0-2-3-6"

        if not name_part.startswith('Components-'):
            continue  # FaceLightMap, Logo, etc. - truly global, skip

        nums_str = name_part[len('Components-'):]   # "0-2-3-6"
        try:
            comp_ids = [int(n) for n in nums_str.split('-')]
        except ValueError:
            continue

        share_count = len(comp_ids)

        # Find the override entry whose ResourceTexture points to this res_id
        override_index = None
        is_replaced    = False
        for game_hash, (oi, oi_res_id) in override_order.items():
            if oi_res_id == res_id:
                override_index = oi
                is_replaced    = (game_hash != fname_hash)
                break
        if override_index is None:
            override_index = 9999 + res_id   # not in any override block; sort last

        for cid in comp_ids:
            key = str(cid)
            comp_textures.setdefault(key, []).append(
                (override_index, res_id, share_count, abs_path, is_replaced)
            )

    # Step 4: sort each component's textures by override_index ascending
    #         (= the true per-component ps-t slot order) and build the result dict.
    result = {}
    for cid, entries in comp_textures.items():
        entries.sort(key=lambda e: e[0])   # sort by override_index = shader slot order
        result[cid] = [
            {
                'path':           str(e[3]),
                'res_id':         e[1],
                'share_count':    e[2],
                'override_index': e[0],
                'is_replaced':    e[4],
            }
            for e in entries
        ]
        if result[cid]:
            print(f"[TexParse]   Component {cid}: {len(result[cid])} texture(s)")

    return result


# ---------------------------------------------------------------------------
# Blender material assignment  (requires bpy - only call from operator)
# ---------------------------------------------------------------------------

def assign_mod_textures(object_source_folder, collection_name=None, progress_cb=None, diffuse_only=False, texture_selection_mode='PST0'):
    """
    Read texture_map.json from object_source_folder, then find matching
    "Component N" objects in the Blender scene and assign materials with
    the mapped textures loaded as image nodes.

    object_source_folder : path to the flat import folder (contains texture_map.json
                           and the texture .dds files)
    collection_name      : optional name of the collection to search in; if None
                           searches all scene objects
    progress_cb          : optional callable(msg: str)
    diffuse_only         : if True, only decode+load the single best diffuse texture
                           per component (fast mode - skips all utility maps).
                           All texture nodes are still created; only the Base Color
                           one has its image loaded.  Typically 5-10x faster.

    Returns dict with success, message, assigned counts.
    """
    import bpy
    from pathlib import Path

    folder = Path(object_source_folder)
    tex_map_path = folder / "texture_map.json"

    def log(msg):
        if progress_cb:
            progress_cb(msg)
        print(f"[ModTextures] {msg}")

    if not tex_map_path.exists():
        raise FileNotFoundError(
            f"texture_map.json not found in: {folder}\n"
            "Run 'Extract Object From Mod' first."
        )

    with open(tex_map_path, "r") as f:
        tex_map = json.load(f)

    # Build a map of "component N" -> bpy object, searching by name pattern
    import re as _re
    comp_pattern = _re.compile(r'.*component[ _\-]*(\d+).*', _re.IGNORECASE)

    scene_objects = {}
    search_pool = bpy.context.scene.objects
    if collection_name:
        col = bpy.data.collections.get(collection_name)
        if col:
            search_pool = col.all_objects

    for obj in search_pool:
        if obj.type != 'MESH':
            continue
        m = comp_pattern.match(obj.name)
        if m:
            cid = m.group(1)
            # Keep the first (or most recently imported) object per component id
            if cid not in scene_objects:
                scene_objects[cid] = obj

    if not scene_objects:
        raise RuntimeError(
            "No 'Component N' mesh objects found in the scene.\n"
            "Import the object first using 'Import Object' mode."
        )

    log(f"Found {len(scene_objects)} component object(s) in scene: "
        f"{sorted(scene_objects.keys(), key=int)}")

    assigned_components = 0
    assigned_textures   = 0
    skipped_components  = 0

    # Create the _preview folder - PNGs converted from DDS go here.
    # IMPORTANT: always wipe and recreate it so stale PNGs from previous runs
    # (which may have been produced by older, buggy conversion code) never poison
    # the colour-scoring pass.  The folder is cheap to rebuild.
    import shutil as _shutil
    preview_dir = folder / "_preview"
    if preview_dir.exists():
        _shutil.rmtree(preview_dir)
    preview_dir.mkdir(exist_ok=True)

    # Cross-component PNG decode cache.
    # Key: dds filename stem (e.g. "Components-0-2-3-6 t=a260e7f7")
    # Value: loaded bpy.data.images image (or None if decode failed)
    # Shared textures (e.g. Components-0-2-3-6) appear in multiple components
    # and were previously decoded once per component - 6-8 redundant decodes
    # for a 2048×2048 BC7 texture is the main reason for 300+ second imports.
    # With this cache, each unique DDS is decoded exactly once across all components.
    png_cache = {}  # stem -> bpy.data.images image or None

    # Purge ALL previously loaded texture images from bpy.data.images that belong
    # to this mod.  When _preview/ is wiped, the on-disk PNGs are gone, but
    # bpy.data.images still holds the old decoded pixel data under the same name.
    # The Priority-1 check (bpy.data.images.get(img_key) with size>0) then returns
    # the stale image from a PREVIOUS run - which may be the wrong texture entirely
    # (e.g. a purple spec map reused as a face diffuse).  Purging here forces a
    # fresh decode on every run, which is the only safe option.
    _all_tex_names = set()
    for _entries in tex_map.values():
        for _e in _entries:
            _orig = Path(_e['path'] if isinstance(_e, dict) else _e)
            _all_tex_names.add(_orig.stem + ".png")   # "Components-0 t=xxx.png"
            _all_tex_names.add(_orig.name)             # "Components-0 t=xxx.dds"
    _all_tex_names.add("__dds_convert_tmp__")
    _purge_count = 0
    for _iname in list(_all_tex_names):
        _existing = bpy.data.images.get(_iname)
        if _existing is not None:
            bpy.data.images.remove(_existing)
            _purge_count += 1
    if _purge_count:
        log(f"Purged {_purge_count} stale image data-block(s) from previous run")

    for comp_id_str, tex_entries in tex_map.items():
        if not tex_entries:
            continue

        obj = scene_objects.get(comp_id_str)
        if obj is None:
            log(f"  Component {comp_id_str}: no matching object in scene, skipping")
            skipped_components += 1
            continue

        # Support both old format (list of str paths) and new format (list of dicts)
        # Old: ["path1", "path2", ...]
        # New: [{"path": "...", "res_id": 0, "share_count": 1, "override_index": N, "is_replaced": bool}, ...]
        def _entry_path(e):
            return e['path'] if isinstance(e, dict) else e
        def _entry_res_id(e, idx):
            return e['res_id'] if isinstance(e, dict) else idx
        def _entry_share(e):
            return e['share_count'] if isinstance(e, dict) else 1
        def _entry_is_replaced(e):
            return e.get('is_replaced', False) if isinstance(e, dict) else False

        def _resolve_tex_path(e):
            """
            Resolve a texture entry to an existing Path.
            Priority order:
              1. Output folder (textures are copied here by prepare_mod_for_import)
              2. _preview folder (PNG already converted)
              3. Original stored absolute path (original mod folder)
            Returns None if not found anywhere.
            """
            orig = Path(_entry_path(e))
            fname = orig.name
            # Check output folder first (copied DDS)
            candidate = folder / fname
            if candidate.exists():
                return candidate
            # Check _preview for a pre-converted PNG
            png_candidate = preview_dir / (orig.stem + ".png")
            if png_candidate.exists():
                return png_candidate
            # Fall back to original absolute path
            if orig.exists():
                return orig
            return None

        valid_entries = [e for e in tex_entries if _resolve_tex_path(e) is not None]
        if not valid_entries:
            log(f"  Component {comp_id_str}: no texture files found (checked output folder, "
                f"_preview, and original paths). Skipping.")
            skipped_components += 1
            continue

        # Create a material named after the component
        mat_name = f"Component_{comp_id_str}_Material"
        mat = bpy.data.materials.get(mat_name)
        if mat is None:
            mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True

        node_tree = mat.node_tree
        nodes     = node_tree.nodes
        links     = node_tree.links

        # Clear existing nodes for a clean slate, then re-affirm use_nodes.
        # Guard: if the material already has image texture nodes from a previous
        # assign run, skip the clear so we don't wipe valid work on a re-run.
        has_tex_nodes = any(n.type == 'TEX_IMAGE' for n in nodes)
        if not has_tex_nodes:
            nodes.clear()
        mat.use_nodes = True

        # Layout: all texture nodes on the left, unwired, with clear slot labels.
        # A Principled BSDF + Output are placed to the right ready to connect.
        # We deliberately do NOT auto-wire any texture to Base Color - WuWa uses
        # many texture slots (diffuse, lightmap, normal, packed, …) and which slot
        # is the "diffuse" varies per component; guessing gets it wrong.  The modder
        # can see all textures in the node editor and connect them as needed.

        out_node    = nodes.new('ShaderNodeOutputMaterial')
        bsdf_node   = nodes.new('ShaderNodeBsdfPrincipled')
        out_node.location  = (600, 300)
        bsdf_node.location = (300, 300)
        links.new(bsdf_node.outputs['BSDF'], out_node.inputs['Surface'])

        # DDS textures often show as magenta in Blender's viewport because the
        # GPU-side GLSL texture upload doesn't support compressed formats like
        # BC7/BC3. However, Blender's CPU-side loader (via OpenImageIO) CAN
        # decode these formats. The fix: load the DDS into Blender, immediately
        # save as PNG to a _preview/ subfolder, then use the PNG for the material.
        # The PNG files are for viewport use only - they are not part of the mod.

        tex_x = -350
        loaded_nodes = []   # list of (slot_idx, entry, tex_node, is_bc45)

        # ---- Helper: decode one DDS -> bpy image, using cross-component cache ----
        def _load_or_cache(tex_path):
            """
            Decode tex_path to PNG and load into Blender.
            Returns (img, is_bc45) or (None, False) on failure.
            Uses png_cache to avoid re-decoding shared textures that appear
            in multiple components (e.g. Components-0-2-3-6 t=xxx.dds).
            """
            stem     = tex_path.stem
            png_name = stem + ".png"
            png_path = preview_dir / png_name
            dds_name = tex_path.name

            # Cache hit: image already decoded this session
            if stem in png_cache:
                return png_cache[stem]

            img     = None
            is_bc45 = False

            # --- Pure-Python decoder (BC1/BC2/BC3/BC4/BC5/BC7) ---
            try:
                decoded = dds_decoder.decode_dds_to_rgba(str(tex_path))
            except Exception as e:
                decoded = None
                log(f"  WARN: DDS decode raised for {dds_name}: {e}")

            if decoded is not None:
                try:
                    dw, dh, rgba = decoded
                    dds_decoder.write_png(str(png_path), dw, dh, rgba)
                    img = bpy.data.images.load(str(png_path))
                    img.name = png_name
                    fmt = dds_decoder._read_dds_header(open(str(tex_path), 'rb').read())
                    if fmt and fmt.get('format', '') in ('bc4u', 'bc4s', 'bc5u', 'bc5s'):
                        is_bc45 = True
                    log(f"  Decoded {dds_name} -> {png_name} ({dw}x{dh})")
                except Exception as e:
                    log(f"  WARN: writing decoded PNG failed for {dds_name}: {e}")
                    img = None

            # --- Fallback: Blender native loader ---
            if img is None:
                try:
                    tmp_name = "__dds_convert_tmp__"
                    if tmp_name in bpy.data.images:
                        bpy.data.images.remove(bpy.data.images[tmp_name])
                    tmp_img = bpy.data.images.load(str(tex_path))
                    tmp_img.name = tmp_name
                    loaded_size = list(tmp_img.size)
                    if loaded_size[0] == 0:
                        raise RuntimeError(f"zero-size image for {dds_name}")
                    tmp_img.file_format = 'PNG'
                    tmp_img.filepath_raw = str(png_path)
                    tmp_img.save()
                    bpy.data.images.remove(tmp_img)
                    img = bpy.data.images.load(str(png_path))
                    img.name = png_name
                    log(f"  Native-loaded {dds_name} -> {png_name} ({loaded_size[0]}x{loaded_size[1]})")
                except Exception as e:
                    log(f"  WARN: native load failed for {dds_name}: {e}")
                    try:
                        img = bpy.data.images.load(str(tex_path))
                        img.name = dds_name
                    except Exception as e2:
                        log(f"  ERROR: total failure for {dds_name}: {e2}")
                        img = None

            result = (img, is_bc45)
            png_cache[stem] = result
            return result

        # ---- In diffuse_only mode, determine the winner BEFORE decoding ----
        # Helper used by both the pre-decode sort and the post-decode wiring sort.
        def _override_idx(entry):
            if isinstance(entry, dict):
                return entry.get('override_index', 9999)
            return 9999

        # ---- In diffuse_only mode, decide which textures to decode ----
        if diffuse_only:
            _decode_paths = set()

            if texture_selection_mode == 'PST0':
                # Only need to decode the single lowest-oi texture (ps-t0).
                valid_for_pst0 = [(e, _resolve_tex_path(e)) for e in tex_entries
                                  if _resolve_tex_path(e) is not None]
                if valid_for_pst0:
                    best_pst0 = min(valid_for_pst0, key=lambda x: _override_idx(x[0]))
                    _decode_paths.add(best_pst0[1])

            elif texture_selection_mode == 'NONE':
                # No Base Color wiring needed - decode nothing in diffuse_only mode.
                pass  # _decode_paths stays empty

            else:  # 'AUTO' - need pixel data to compare, decode all share=1 textures
                for e in tex_entries:
                    tp = _resolve_tex_path(e)
                    if tp is None:
                        continue
                    if _entry_share(e) == 1:
                        _decode_paths.add(tp)
                # Also decode first group entry as fallback for share>=3 components
                group_entries = [(e, _resolve_tex_path(e)) for e in tex_entries
                                 if _resolve_tex_path(e) is not None and 3 <= _entry_share(e) <= 5]
                if group_entries:
                    best_group = min(group_entries, key=lambda x: _override_idx(x[0]))
                    _decode_paths.add(best_group[1])
        else:
            _decode_paths = None  # None means decode everything

        # ---- Main per-texture loop ----
        for slot_idx, entry in enumerate(tex_entries):
            tex_path = _resolve_tex_path(entry)
            if tex_path is None:
                continue

            share       = _entry_share(entry)
            replaced    = _entry_is_replaced(entry)
            replaced_tag = " [MOD]" if replaced else ""

            if _decode_paths is not None and tex_path not in _decode_paths:
                # Create the node shell (labelled, correctly positioned) but
                # don't load or decode the image - keeps the node editor informative
                # while skipping the expensive decode step for utility textures.
                tex_node          = nodes.new('ShaderNodeTexImage')
                tex_node.image    = None
                tex_node.location = (tex_x, 300 - slot_idx * 300)
                tex_node.label    = f"ps-t{slot_idx}  [shared={share}]{replaced_tag}  {tex_path.stem}"
                # Mark is_bc45=True so shell nodes never win Base Color selection.
                loaded_nodes.append((slot_idx, entry, tex_node, True))
                continue

            img, is_bc45 = _load_or_cache(tex_path)
            if img is None:
                continue

            tex_node          = nodes.new('ShaderNodeTexImage')
            tex_node.image    = img
            tex_node.location = (tex_x, 300 - slot_idx * 300)
            tex_node.label    = f"ps-t{slot_idx}  [shared={share}]{replaced_tag}  {tex_path.stem}"

            loaded_nodes.append((slot_idx, entry, tex_node, is_bc45))
            assigned_textures += 1

        # Wire a texture node to Base Color according to texture_selection_mode:
        #
        #   'PST0'  -> wire the texture with the lowest override_index for this
        #             component. This is the ps-t0 slot in the shader, which in
        #             WuWa is consistently the diffuse/albedo texture.  Reliable
        #             across mods because it mirrors the actual shader binding.
        #
        #   'AUTO'  -> pixel-content scoring: pick the most saturated non-excluded
        #             texture (excludes blue-dominant normals, pure-dark maps, BC4/5
        #             utility maps).  More heuristic-dependent; may pick wrong on
        #             mods where the diffuse happens to have lower saturation than
        #             a colourful packed map.
        #
        #   'NONE'  -> load all nodes, wire nothing; the modder connects manually
        #             in the Shader Editor with one drag.
        if loaded_nodes:
            best_node  = None
            best_entry = None

            if texture_selection_mode == 'NONE':
                # Don't wire anything - user connects manually
                log(f"  Component {comp_id_str}: {len(loaded_nodes)} texture node(s) loaded "
                    f"(Base Color: manual - drag the connection in Shader Editor)")

            elif texture_selection_mode == 'PST0':
                # Wire the texture with the lowest override_index.
                # The override_index encodes per-component shader slot order, so
                # the minimum oi for each component = ps-t0 = the diffuse slot.
                eligible = [(slot_idx, entry, node, is_bc45)
                            for slot_idx, entry, node, is_bc45 in loaded_nodes
                            if node.image is not None and node.image.size[0] > 0]
                if eligible:
                    # Sort by override_index ascending; pick the first (lowest oi)
                    eligible.sort(key=lambda t: _override_idx(t[1]))
                    _, best_entry, best_node, _ = eligible[0]
                    best_oi  = _override_idx(best_entry)
                    best_img = best_node.image
                    best_px  = f"{best_img.size[0]}x{best_img.size[1]}" if best_img else "?"
                    links.new(best_node.outputs['Color'], bsdf_node.inputs['Base Color'])
                    log(f"  Component {comp_id_str}: {len(loaded_nodes)} texture node(s) loaded, "
                        f"Base Color <- {Path(_entry_path(best_entry)).stem} "
                        f"(ps-t0 mode, override_idx={best_oi}, size={best_px})")
                else:
                    log(f"  Component {comp_id_str}: {len(loaded_nodes)} texture node(s) loaded "
                        f"(no image loaded successfully; Base Color not wired)")

            else:  # 'AUTO' - pixel saturation heuristic
                def _pixel_stats(node):
                    """Return (mean_r, mean_g, mean_b, mean_sat) or None on failure."""
                    img = node.image
                    if img is None or img.size[0] == 0:
                        return None
                    try:
                        pixels = list(img.pixels)
                        if not pixels:
                            return None
                        stride = 256 * 4
                        rs, gs, bs = [], [], []
                        for i in range(0, len(pixels), stride):
                            rs.append(pixels[i])
                            gs.append(pixels[i + 1])
                            bs.append(pixels[i + 2])
                        if not rs:
                            return None
                        mr = sum(rs) / len(rs)
                        mg = sum(gs) / len(gs)
                        mb = sum(bs) / len(bs)
                        sat = sum(max(r, g, b) - min(r, g, b)
                                  for r, g, b in zip(rs, gs, bs)) / len(rs)
                        return (mr, mg, mb, sat)
                    except Exception:
                        return None

                def _auto_sort_key(item):
                    """
                    Exclusion rules, validated against both the Dragon Lady mod
                    (where the diffuse has high saturation, e.g. a260e7f7 dragon-red)
                    and the Top mod (where the diffuse is muted blue/grey, e.g.
                    768f8f35 sat=0.168, e6c846f5 sat=0.198 - LOWER saturation than
                    several of the wrong-winner packed maps).  Picking by "highest
                    saturation" alone fails on the Top mod, so the approach here is
                    exclusion-first: throw out anything that is structurally a
                    utility/packed/lightmap/normal map, then let only the diffuse
                    candidates compete on saturation as a tiebreaker.

                      Rule 1 - B≈0            -> RG-packed map (tangent XY, flow, etc.)
                      Rule 2 - one channel maxed
                               and far above        -> single-channel/packed detail map
                               the next highest         (e.g. R=1.0, B=0.47, gap=0.53)
                      Rule 3a - near-pure greyscale  -> lightmap / AO / packed mid-grey
                      Rule 3b - slightly grey + bright-> specular / sheen map
                      Rule 4 - very dark             -> shadow / occlusion map
                      Rule 5 - blue-dominant, R≈G≈0.5 -> tangent-space normal map
                    """
                    slot_idx, entry, node, is_bc45 = item
                    if is_bc45:
                        return (1, 4, 0.0, 0)
                    stats = _pixel_stats(node)
                    if stats is None:
                        return (1, 4, 0.0, 0)
                    mr, mg, mb, sat = stats
                    brightness = (mr + mg + mb) / 3.0
                    chs = sorted([mr, mg, mb])
                    spread  = chs[2] - chs[0]
                    gap_top = chs[2] - chs[1]

                    # Rule 1: B near zero -> RG-packed utility map
                    if mb < 0.05:
                        return (1, 4, 0.0, 0)
                    # Rule 2: one channel maxed and far above the next -> packed/detail map
                    if chs[2] > 0.92 and gap_top > 0.45:
                        return (1, 4, 0.0, 0)
                    # Rule 3a: near-pure greyscale (any brightness) -> lightmap/AO/packed
                    # Threshold lowered from 0.06 to 0.04 so muted fabric diffuses
                    # (spread~0.057) are not excluded alongside true greyscale AO.
                    if spread < 0.04:
                        return (1, 4, 0.0, 0)
                    # Rule 3b: slightly grey but bright -> specular/sheen map
                    if spread < 0.12 and brightness > 0.45:
                        return (1, 4, 0.0, 0)
                    # Rule 4: very dark -> shadow/occlusion map
                    if brightness < 0.12:
                        return (1, 4, 0.0, 0)
                    # Rule 5: tangent-space normal map (blue-dominant, R≈G≈0.5)
                    rg_near_half = (abs(mr - mg) < 0.06 and
                                     0.38 <= mr <= 0.62 and
                                     0.38 <= mg <= 0.62)
                    if mb > mr * 1.35 and rg_near_half:
                        return (1, 4, 0.0, 0)

                    share = _entry_share(entry)
                    if share == 1:        tier = 0
                    elif 3 <= share <= 5: tier = 1
                    elif 6 <= share <= 7: tier = 2
                    elif share == 2:      tier = 3
                    else:                 tier = 4
                    oi = _override_idx(entry)
                    if share == 1:
                        return (0, tier, -sat, oi)
                    else:
                        return (0, tier, oi, -sat)

                sorted_nodes = sorted(loaded_nodes, key=_auto_sort_key)
                _, best_entry, best_node, _ = sorted_nodes[0]

                # If ALL candidates were excluded, fall back to the largest texture
                # by pixel area - diffuse maps are almost always highest resolution.
                all_excluded = all(_auto_sort_key(item)[0] == 1 for item in loaded_nodes)
                if all_excluded:
                    def _area(item):
                        img = item[2].image
                        return (img.size[0] * img.size[1]) if (img and img.size[0] > 0) else 0
                    by_area = sorted(loaded_nodes, key=_area, reverse=True)
                    _, best_entry, best_node, _ = by_area[0]
                    log(f"  Component {comp_id_str}: all textures excluded by heuristic; "
                        f"falling back to largest by area.")

                best_oi  = _override_idx(best_entry)
                best_img = best_node.image
                best_px  = f"{best_img.size[0]}x{best_img.size[1]}" if best_img else "?"
                links.new(best_node.outputs['Color'], bsdf_node.inputs['Base Color'])
                log(f"  Component {comp_id_str}: {len(loaded_nodes)} texture node(s) loaded, "
                    f"Base Color <- {Path(_entry_path(best_entry)).stem} "
                    f"(auto mode, share={_entry_share(best_entry)}, override_idx={best_oi}, size={best_px})")

        # Assign material to object
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)

        # Set viewport display so textures are visible in Material Preview
        obj.data.materials[0] = mat

        comp_tex_count = len([e for e in tex_entries if _resolve_tex_path(e) is not None])
        log(f"  Component {comp_id_str} ({obj.name}): "
            f"assigned {comp_tex_count} texture(s) ({len(loaded_nodes)} loaded)")
        assigned_components += 1

    msg = (f"Done! Assigned textures to {assigned_components} component(s) "
           f"({assigned_textures} image(s) loaded).")
    if skipped_components:
        msg += f" {skipped_components} component(s) had no matching object."
    log(msg)

    return {
        "success":            True,
        "message":            msg,
        "assigned_components": assigned_components,
        "assigned_textures":  assigned_textures,
        "skipped_components": skipped_components,
    }


# ---------------------------------------------------------------------------
# INI texture transplant
# ---------------------------------------------------------------------------

def fix_ini_textures(generated_ini_path, original_mod_folder, progress_cb=None):
    """
    Transplants the texture override section from an original Dragon Lady style
    mod.ini into a freshly exported mod.ini, preserving all mesh/skeleton
    sections from the new export.

    generated_ini_path : path to the newly exported mod.ini
    original_mod_folder: path to the original mod folder (contains mod.ini with
                         correct Dragon Lady texture hashes)

    Modifies generated_ini_path in-place.
    Returns summary dict.
    """
    generated_ini_path  = Path(generated_ini_path)
    original_mod_folder = Path(original_mod_folder)

    def log(msg):
        if progress_cb:
            progress_cb(msg)
        print(f"[IniTexFix] {msg}")

    orig_ini = original_mod_folder / "mod.ini"
    if not orig_ini.exists():
        raise FileNotFoundError(f"Original mod.ini not found: {orig_ini}")
    if not generated_ini_path.exists():
        raise FileNotFoundError(f"Generated mod.ini not found: {generated_ini_path}")

    with open(orig_ini, "r", encoding="utf-8-sig") as f:
        orig_text = f.read()
    with open(generated_ini_path, "r", encoding="utf-8-sig") as f:
        new_text = f.read()

    # --- Extract texture section from original ---
    TEX_MARKER   = "; Shading: Textures"
    SKEL_MARKER  = "; Resources: Skeleton"

    if TEX_MARKER not in orig_text:
        raise ValueError("Original mod.ini missing '; Shading: Textures' section marker")
    if SKEL_MARKER not in new_text:
        raise ValueError("Generated mod.ini missing '; Resources: Skeleton' section marker")

    orig_tex_section = orig_text[orig_text.index(TEX_MARKER):orig_text.index(SKEL_MARKER)]
    log(f"Extracted texture section from original ({len(orig_tex_section)} chars)")

    # Count texture entries
    n_textures = len(re.findall(r'\[ResourceTexture\d+\]', orig_tex_section))
    n_injured  = len(re.findall(r'injured', orig_tex_section, re.IGNORECASE))
    log(f"Found {n_textures} texture resources, {n_injured} injured-state overrides")

    # --- Rebuild new ini: pre-texture + orig texture section + post-texture ---
    if TEX_MARKER in new_text:
        pre_texture = new_text[:new_text.index(TEX_MARKER)]
    else:
        pre_texture = new_text[:new_text.index(SKEL_MARKER)]

    post_texture = new_text[new_text.index(SKEL_MARKER):]

    # Fix required_wwmi_version if needed (1.00 -> 0.70 for older WWMI installs)
    if "$required_wwmi_version = 1.00" in pre_texture:
        pre_texture = pre_texture.replace(
            "global $required_wwmi_version = 1.00",
            "global $required_wwmi_version = 0.70"
        )
        log("Patched $required_wwmi_version: 1.00 -> 0.70")

    final = pre_texture + orig_tex_section + "\n" + post_texture

    # --- Copy original mod textures that may be missing from new export ---
    new_mod_folder = generated_ini_path.parent
    orig_tex_dir   = original_mod_folder / "Textures"
    new_tex_dir    = new_mod_folder / "Textures"
    new_tex_dir.mkdir(exist_ok=True)

    copied = 0
    if orig_tex_dir.exists():
        for dds in orig_tex_dir.glob("*.dds"):
            dst = new_tex_dir / dds.name
            if not dst.exists():
                shutil.copy(dds, dst)
                copied += 1
    log(f"Copied {copied} missing texture file(s) from original mod")

    with open(generated_ini_path, "w", encoding="utf-8") as f:
        f.write(final)

    msg = (f"mod.ini updated with {n_textures} original texture overrides "
           f"({n_injured} injured states). {copied} texture files copied.")
    log(msg)
    return {"success": True, "message": msg,
            "textures": n_textures, "injured": n_injured, "copied": copied}