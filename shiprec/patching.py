import dask.array as da
from typing import Tuple


def _split_slide_into_patches(slide: da.Array, patch_size: int = 224) -> da.Array:
    # slide: da.Array of shape (h, w, 3)
    # returns: da.Array of shape (n_patches_h, n_patches_w, patch_size, patch_size, 3)
    h, w, *extra_dims = slide.shape
    slide = slide.reshape(h // patch_size, patch_size, w, *extra_dims)
    slide = slide.reshape(h // patch_size, patch_size, w // patch_size, patch_size, *extra_dims)
    slide = slide.swapaxes(1, 2)
    return slide


def split_slide_into_patches(slide: da.Array, patch_size: int = 224) -> Tuple[da.Array, da.Array]:
    patches = _split_slide_into_patches(slide, patch_size=patch_size)
    xs, ys = da.meshgrid(da.arange(patches.shape[1]), da.arange(patches.shape[0]))
    patch_coords = da.stack([xs, ys], axis=-1) * patch_size
    patch_coords = patch_coords.rechunk(-1)

    return patches, patch_coords


def flatten_patches_and_coords(patches: da.Array, coords: da.Array) -> Tuple[da.Array, da.Array]:
    # patches: da.Array of shape (n_patches_h, n_patches_w, patch_size, patch_size, 3)
    # coords: da.Array of shape (n_patches_h, n_patches_w, 2)
    # returns: da.Array of shape (n_patches_h * n_patches_w, patch_size, patch_size, 3)
    #          da.Array of shape (n_patches_h * n_patches_w, 2)
    patches = patches.reshape(-1, *patches.shape[-3:])
    coords = coords.reshape(-1, coords.shape[-1])
    return patches, coords
