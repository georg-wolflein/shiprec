from cucim import CuImage
from ctypes import c_uint32
import numpy as np
from typing import Tuple
from pathlib import Path
from skimage.util import img_as_float

from .readers import SlideReader
from .mpp import MPPExtractionError, MPPExtractor


class CucimReader(SlideReader):
    def __init__(self, path: Path):
        super().__init__(path)
        self._slide = CuImage(str(path))
        self._mpp = None

    def read_region(self, loc: Tuple[int, int], level: int, size: Tuple[int, int]) -> np.ndarray:
        region = self._slide.read_region(loc, size, level)
        return ((img_as_float(np.asarray(region))) * 255).astype(np.uint8)

    @property
    def mpp(self) -> float:
        if self._mpp is None:
            # TODO: implement mpp extraction for cucim
            from .openslide import openslide_mpp_extractor
            from openslide import OpenSlide

            with OpenSlide(str(self._path)) as slide:
                self._mpp = openslide_mpp_extractor(slide)
        return self._mpp

    @property
    def level_dimensions(self) -> Tuple[int, int]:
        return self._slide.resolutions["level_dimensions"]

    @property
    def level_downsamples(self) -> Tuple[float]:
        return self._slide.resolutions["level_downsamples"]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._slide.close()
