import numpy as np
import cv2
import dask.array as da


def is_background(patch: np.ndarray) -> bool:
    # patch: np.ndarray grayscale image of shape (patch_size, patch_size)

    # hardcoded thresholds
    edge = cv2.Canny(patch, 40, 100)

    # avoid dividing by zero
    edge = (edge / np.max(edge)) if np.max(edge) != 0 else 0
    num_pixels = np.prod(patch.shape)
    edge = ((np.sum(np.sum(edge)) / num_pixels) * 100) if num_pixels != 0 else 0

    # hardcoded limit. Less or equal to 2 edges will be rejected (i.e., not saved)
    return edge < 2.0


def vectorized_is_background(patches: np.ndarray) -> np.ndarray:
    # patches: np.ndarray grayscale image of shape (n_patches, patch_size, patch_size)
    return np.array([is_background(patch) for patch in patches])


def rgb2gray(rgb: da.Array) -> da.Array:
    return da.dot(rgb, da.array([0.299, 0.587, 0.114])).astype(
        np.uint8
    )  # see https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html


def get_canny_foreground_mask(patches: da.Array) -> da.Array:
    # patches: da.Array of shape (n_patches, patch_size, patch_size, 3)
    gray_patches = rgb2gray(patches)
    patch_mask = ~da.map_blocks(vectorized_is_background, patches, dtype=bool, drop_axis=(1, 2, 3))
    return patch_mask
