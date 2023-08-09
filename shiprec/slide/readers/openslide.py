import openslide
from openslide import lowlevel as openslide_ll
from ctypes import c_uint32
import numpy as np
from typing import Tuple
from pathlib import Path

from .readers import SlideReader
from .mpp import MPPExtractionError, MPPExtractor


class OpenSlideReader(SlideReader):
    def __init__(self, path: Path):
        super().__init__(path)
        self._slide = openslide.OpenSlide(str(path))
        self._mpp = None

    def read_region(self, loc: Tuple[int, int], level: int, size: Tuple[int, int]) -> np.ndarray:
        """Adapted from openslide.lowlevel.read_region() to not use PIL images, but directly return a numpy array."""
        x, y = loc
        w, h = size
        if w < 0 or h < 0:
            raise openslide.OpenSlideError("negative width (%d) or negative height (%d) not allowed" % (w, h))
        buf = (w * h * c_uint32)()
        openslide_ll._read_region(self._slide._osr, buf, x, y, level, w, h)
        openslide_ll._convert.argb2rgba(buf)
        img = np.frombuffer(buf, dtype=np.uint8).reshape(*size[::-1], 4)[..., :3]  # remove alpha channel
        return img

    @property
    def mpp(self) -> float:
        if self._mpp is None:
            self._mpp = openslide_mpp_extractor(self._slide)
        return self._mpp

    @property
    def level_dimensions(self) -> Tuple[int, int]:
        return self._slide.level_dimensions

    @property
    def level_downsamples(self) -> Tuple[float]:
        return self._slide.level_downsamples

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._slide.close()


openslide_mpp_extractor = MPPExtractor()


@openslide_mpp_extractor.register
def extract_mpp_from_properties(slide: openslide.OpenSlide) -> float:
    try:
        return float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
    except KeyError:
        raise MPPExtractionError


@openslide_mpp_extractor.register
def extract_mpp_from_metadata(slide: openslide.OpenSlide) -> float:
    import xml.dom.minidom as minidom

    xml_path = slide.properties["tiff.ImageDescription"]
    try:
        doc = minidom.parseString(xml_path)
    except Exception:
        raise MPPExtractionError
    collection = doc.documentElement
    images = collection.getElementsByTagName("Image")
    pixels = images[0].getElementsByTagName("Pixels")
    mpp = float(pixels[0].getAttribute("PhysicalSizeX"))
    if not mpp:
        raise MPPExtractionError
    return mpp


@openslide_mpp_extractor.register
def extract_mpp_from_comments(slide: openslide.OpenSlide) -> float:
    import re

    slide_properties = slide.properties.get("openslide.comment")
    pattern = r"<PixelSizeMicrons>(.*?)</PixelSizeMicrons>"
    match = re.search(pattern, slide_properties)
    if not match:
        raise MPPExtractionError
    return match.group(1)
