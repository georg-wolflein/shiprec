import dask.array as da
import zarr
from pathlib import Path
from typing import Union
import openslide
import numpy as np
import cv2

from .slide_store import OpenSlideStore
from .mpp import get_slide_mpp


def load_slide(slide_file: Union[Path, str], target_mpp: float = 256.0 / 224.0) -> da.Array:
    slide = openslide.OpenSlide(str(slide_file))
    slide_mpp = get_slide_mpp(slide)

    zoom_factor = int(target_mpp / slide_mpp)
    print(f"Zoom factor: {zoom_factor}")

    # loaded_slide_size = np.array(slide.dimensions).astype(int)
    # target_slide_size = (loaded_slide_size * slide_mpp / target_mpp).astype(int)

    target_slide_chunk_size = 224 * 20
    loaded_slide_chunk_size = np.ceil(target_slide_chunk_size * target_mpp / slide_mpp).astype(int)

    def resize_chunk(chunk):
        return cv2.resize(chunk, (target_slide_chunk_size, target_slide_chunk_size))

    store = OpenSlideStore(slide_file, tilesize=loaded_slide_chunk_size, pad=True)
    grp = zarr.open(store, mode="r")

    level = 0
    z = grp[level]
    dz = da.from_zarr(z)

    dz_resized = da.map_blocks(resize_chunk, dz, chunks=(target_slide_chunk_size, target_slide_chunk_size, 3))

    return dz_resized


__all__ = ["load_slide", "OpenSlideStore", "get_slide_mpp", "read_region"]
