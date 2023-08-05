import openslide
from openslide import lowlevel as openslide_ll
from ctypes import c_uint32, POINTER, cast, addressof, c_void_p, byref, c_uint8, sizeof, c_uint16
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union
from loguru import logger
import re
import cv2
from multiprocessing import RawArray
import concurrent
from concurrent.futures import ProcessPoolExecutor
from concurrent import futures
from pathlib import Path
from tqdm import tqdm

from shiprec.mpp import get_slide_mpp


def read_region(
    slide: openslide.OpenSlide, loc: Tuple[int, int], level: int, size: Tuple[int, int], resize=None
) -> np.ndarray:
    """Adapted from openslide.lowlevel.read_region() to not use PIL images, but directly return a numpy array."""
    x, y = loc
    w, h = size
    if w < 0 or h < 0:
        raise openslide.OpenSlideError("negative width (%d) or negative height (%d) not allowed" % (w, h))
    buf = (w * h * c_uint32)()
    openslide_ll._read_region(slide._osr, buf, x, y, level, w, h)
    openslide_ll._convert.argb2rgba(buf)
    img = np.frombuffer(buf, dtype=np.uint8).reshape(*size[::-1], 4)[..., :3]  # remove alpha channel
    if resize is not None:
        img = cv2.resize(img, resize)
    return img


_ws_buf = None


def _init_worker(ws_buf):
    global _ws_buf
    _ws_buf = ws_buf


def _worker(slide_file, loaded_tile_size, target_tile_size, target_slide_size, i):
    slide = openslide.OpenSlide(slide_file)
    ws_arr = np.ndarray((*target_slide_size[::-1], 3), dtype=np.uint8, buffer=_ws_buf)
    tile = read_region(slide, np.array(loaded_tile_size) * (0, i), 0, loaded_tile_size, target_tile_size)
    tw, th = target_tile_size
    ws_arr[th * i : th * (i + 1), 0:tw] = tile


def load_slide(
    slide_file: Union[Path, str],
    target_mpp: float = 256.0 / 224.0,
    num_processes: int = 16,
    num_steps: int = 32,
) -> np.ndarray:
    slide = openslide.OpenSlide(str(slide_file))
    slide_mpp = get_slide_mpp(slide)

    stride = np.ceil(np.array(slide.dimensions) / num_steps)  # stride in pixels at level 0
    stride = stride.astype(int)

    loaded_slide_size = np.array(slide.dimensions).astype(int)

    stride_height = loaded_slide_size[1] // num_steps
    loaded_slide_width = loaded_slide_size[0]
    loaded_tile_size = np.array([loaded_slide_width, stride_height])

    target_tile_size = (loaded_tile_size * slide_mpp / target_mpp).astype(int)
    target_slide_size = target_tile_size * (1, num_steps)

    ws_buf = RawArray(c_uint8, int(np.prod(target_slide_size) * 3))

    if num_processes == 1 and num_steps == 1:
        _init_worker(ws_buf)
        _worker(slide_file, loaded_tile_size, target_tile_size, target_slide_size, 0)
    else:
        fs = []
        with ProcessPoolExecutor(num_processes, initializer=_init_worker, initargs=(ws_buf,)) as p:
            for i in range(num_steps):
                fs.append(p.submit(_worker, slide_file, loaded_tile_size, target_tile_size, target_slide_size, i))
            for _ in tqdm(futures.as_completed(fs), total=len(fs), desc="Loading slide", position=0):
                pass
    return np.ndarray((*target_slide_size[::-1], 3), dtype=np.uint8, buffer=ws_buf)


if __name__ == "__main__":
    from timeit import timeit

    SLIDE_FILE = "/data/data/TCGA-BRCA-DX-IMGS_1133/TCGA-AO-A12B-01Z-00-DX1.B215230B-5FF7-4B0A-9C1E-5F1658534B11.svs"
    print(timeit(lambda: load_slide(SLIDE_FILE, num_processes=16, num_steps=16), number=1))
