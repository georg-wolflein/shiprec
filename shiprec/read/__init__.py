import dask.array as da
import zarr
from pathlib import Path
from typing import Union
import openslide
import numpy as np
import cv2
from typing import Optional
from loguru import logger

from .slide_store import OpenSlideStore
from .mpp import get_slide_mpp


def load_slide(
    slide_file: Union[Path, str], target_mpp: float = 256.0 / 224.0, level: Optional[int] = None
) -> da.Array:
    slide = openslide.OpenSlide(str(slide_file))
    slide_mpp = get_slide_mpp(slide)

    logger.debug(
        f"Slide has {slide.level_count} levels with following downsamples: {({k: v for k, v in enumerate(slide.level_downsamples)})}"
    )

    if level is None:
        # Intelligently choose correct level
        level = slide.get_best_level_for_downsample(target_mpp / slide_mpp)

    level_mpp = slide.level_downsamples[level] * slide_mpp
    logger.info(f"Using level {level} with {level_mpp=:.3f} for {slide_mpp=:.3f} and {target_mpp=:.3f}")

    target_slide_chunk_size = 224 * 20
    loaded_slide_chunk_size = np.ceil(target_slide_chunk_size * target_mpp / level_mpp).astype(int)

    def resize_chunk(chunk):
        return cv2.resize(chunk, (target_slide_chunk_size, target_slide_chunk_size))

    store = OpenSlideStore(slide_file, tilesize=loaded_slide_chunk_size, pad=True)
    grp = zarr.open(store, mode="r")

    z = grp[level]
    dz = da.from_zarr(z)

    dz_resized = da.map_blocks(resize_chunk, dz, chunks=(target_slide_chunk_size, target_slide_chunk_size, 3))

    return dz_resized


__all__ = ["load_slide", "OpenSlideStore", "get_slide_mpp", "read_region"]
