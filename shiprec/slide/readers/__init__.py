from pathlib import Path
from typing import Literal

from .readers import SlideReader
from .openslide import OpenSlideReader
from .cucim import CucimReader

Backend = Literal["openslide", "cucim", "auto"]


def make_slide_reader(file: Path, backend: Backend) -> SlideReader:
    if backend == "auto":
        backend = "cucim"  # TODO: switch backend depending on file type (e.g. .svs -> cucim, .czi -> libczi)
    return {"openslide": OpenSlideReader, "cucim": CucimReader}[backend](file)
