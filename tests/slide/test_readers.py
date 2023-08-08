import openslide
import numpy as np
from pathlib import Path

from shiprec.slide.readers import SlideReader, OpenSlideReader

TEST_SLIDE_PATH = Path(__file__).parent.parent / "test_slide.svs"


def test_read_region():
    with OpenSlideReader(TEST_SLIDE_PATH) as slide:
        loc = (0, 0)
        level = 0
        size = (100, 100)
        img = slide.read_region(loc, level, size)
        assert isinstance(img, np.ndarray)
        assert img.shape == (100, 100, 3)
        assert np.all(img >= 0) and np.all(img <= 255)


def test_read_region_vs_openslide():
    with openslide.OpenSlide(TEST_SLIDE_PATH) as openslide_slide:
        with OpenSlideReader(TEST_SLIDE_PATH) as slide:
            loc = (0, 0)
            level = 0
            size = (100, 100)
            img = slide.read_region(loc, level, size)
            img2 = np.array(openslide_slide.read_region(loc, level, size))[..., :3]
            assert np.all(img == img2)
