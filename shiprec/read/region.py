import openslide
from openslide import lowlevel as openslide_ll
from ctypes import c_uint32
import numpy as np
from typing import Tuple


def read_region(slide: openslide.OpenSlide, loc: Tuple[int, int], level: int, size: Tuple[int, int]) -> np.ndarray:
    """Adapted from openslide.lowlevel.read_region() to not use PIL images, but directly return a numpy array."""
    x, y = loc
    w, h = size
    if w < 0 or h < 0:
        raise openslide.OpenSlideError("negative width (%d) or negative height (%d) not allowed" % (w, h))
    buf = (w * h * c_uint32)()
    openslide_ll._read_region(slide._osr, buf, x, y, level, w, h)
    openslide_ll._convert.argb2rgba(buf)
    img = np.frombuffer(buf, dtype=np.uint8).reshape(*size[::-1], 4)[..., :3]  # remove alpha channel
    return img
