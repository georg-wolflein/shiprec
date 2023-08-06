import openslide
import numpy as np
from pathlib import Path

from shiprec.slide.load_slide import read_region


def test_read_region():
    with openslide.OpenSlide(Path(__file__).parent / "test_slide.svs") as slide:
        loc = (0, 0)
        level = 0
        size = (100, 100)
        img = read_region(slide, loc, level, size)
        assert isinstance(img, np.ndarray)
        assert img.shape == (100, 100, 3)
        assert np.all(img >= 0) and np.all(img <= 255)


def test_read_region_vs_openslide():
    with openslide.OpenSlide(Path(__file__).parent / "test_slide.svs") as slide:
        loc = (0, 0)
        level = 0
        size = (100, 100)
        img = read_region(slide, loc, level, size)
        img2 = np.array(slide.read_region(loc, level, size))[..., :3]
        assert np.all(img == img2)


def test_read_region_resize():
    with openslide.OpenSlide(Path(__file__).parent / "test_slide.svs") as slide:
        loc = (0, 0)
        level = 0
        size = (100, 100)
        img = read_region(slide, loc, level, size, resize=(50, 50))
        assert img.shape == (50, 50, 3)
