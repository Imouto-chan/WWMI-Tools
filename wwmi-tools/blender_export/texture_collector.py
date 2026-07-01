import os
import re
import json

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Texture:
    hash: str
    path: Path
    filename: str
    # Hash 3DMigoto needs to match the game's live draw call at runtime. Usually identical to
    # `hash`, but diverges whenever the vanilla dump reported both a content-disambiguated hash
    # (used for the on-disk filename) and the resource's normal-mode hash -- see TextureHashes.json.
    override_hash: Optional[str] = None

    def __post_init__(self):
        if self.override_hash is None:
            self.override_hash = self.hash


def get_textures(object_source_folder: Path, exclude_hashes: List[str]):
    textures = {}

    # Written by "Extract Objects From Dump" when available; maps each texture's content hash
    # (the `t=XXXXXXXX` embedded in its filename) to the correct runtime-matching hash. Object
    # source folders extracted before this manifest existed simply won't have the file, and
    # override_hash falls back to hash (the old, sometimes-incorrect behavior) for those.
    texture_hashes = {}
    texture_hashes_path = object_source_folder / 'TextureHashes.json'
    if texture_hashes_path.is_file():
        with open(texture_hashes_path, 'r', encoding='utf-8') as f:
            texture_hashes = json.load(f)

    for texture_filename in os.listdir(object_source_folder):
        if texture_filename.endswith(".dds") or texture_filename.endswith(".jpg"): 
            # Handle new format
            hash_pattern = re.compile(r'.*t=([a-f0-9]{8}).*')
            result = hash_pattern.findall(texture_filename.lower())
            
            if len(result) != 1:
                # Handle old format
                hash_pattern = re.compile(r'.*component_\d-ps-t\d-([a-f0-9]{8}).*')
                result = hash_pattern.findall(texture_filename.lower())
                if len(result) != 1:
                    continue

            texture_hash = result[0]

            if exclude_hashes and texture_hash in exclude_hashes:
                continue

            textures[texture_hash] = Texture(
                hash=texture_hash,
                path=object_source_folder / texture_filename,
                filename=texture_filename,
                override_hash=texture_hashes.get(texture_hash),
            )
    return list(textures.values())
