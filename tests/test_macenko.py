from pathlib import Path
import pytest
import cv2
import dask.array as da
import numpy as np
import dask
from functools import partial

from shiprec.macenko import NumpyMacenkoNormalizer, DaskMacenkoNormalizer


TARGET_PATCH_FILE = Path(__file__).parent.parent / "normalization_template.jpg"
NORMALIZERS = [
    NumpyMacenkoNormalizer,
    partial(DaskMacenkoNormalizer, exact=False),
    partial(DaskMacenkoNormalizer, exact=True),
]


@pytest.mark.parametrize("normalizer", NORMALIZERS)
def test_fit_shapes(normalizer):
    target = cv2.cvtColor(cv2.imread(str(TARGET_PATCH_FILE)), cv2.COLOR_BGR2RGB)
    norm = normalizer()
    if isinstance(norm, DaskMacenkoNormalizer):
        target = da.from_array(target, chunks=-1)
    norm.fit(target)
    assert norm.HERef.shape == (3, 2)
    assert norm.maxCRef.shape == (2,)


@pytest.mark.parametrize("normalizer", NORMALIZERS)
def test_normalize_shapes(normalizer):
    target = cv2.cvtColor(cv2.imread(str(TARGET_PATCH_FILE)), cv2.COLOR_BGR2RGB)
    img = cv2.resize(target[200:300, 200:500], (2048, 3128))

    norm = normalizer()
    if isinstance(norm, DaskMacenkoNormalizer):
        target = da.from_array(target, chunks=-1)
        img = da.from_array(img, chunks=(128, 128, 3))
    norm.fit(target)
    result = norm.normalize(img)
    assert result.shape == img.shape


def test_numpy_vs_dask_implementation():
    target_np = cv2.cvtColor(cv2.imread(str(TARGET_PATCH_FILE)), cv2.COLOR_BGR2RGB)
    img_np = cv2.resize(target_np[200:300, 200:500], (2048, 2560))

    norm_np = NumpyMacenkoNormalizer()
    norm_np.fit(target_np)

    target_da = da.from_array(target_np, chunks=-1)
    img_da = da.from_array(img_np, chunks=(128, 128, 3))

    norm_da = DaskMacenkoNormalizer(exact=True)
    norm_da.fit(target_da)

    # Compare HERef and maxCRef
    dask_HERef, dask_maxCRef = dask.compute(norm_da.HERef, norm_da.maxCRef)
    assert np.allclose(norm_np.HERef, dask_HERef)
    assert np.allclose(norm_np.maxCRef, dask_maxCRef)

    # Compare normalization results
    result_np = norm_np.normalize(img_np)
    result_da = norm_da.normalize(img_da).compute()
    assert np.allclose(result_np, result_da)


def test_numpy_vs_dask_implementation_inexact():
    target_np = cv2.cvtColor(cv2.imread(str(TARGET_PATCH_FILE)), cv2.COLOR_BGR2RGB)
    img_np = cv2.resize(target_np[200:300, 200:500], (2048, 2560))

    norm_np = NumpyMacenkoNormalizer()
    norm_np.fit(target_np)

    target_da = da.from_array(target_np, chunks=-1)
    img_da = da.from_array(img_np, chunks=(128, 128, 3))

    norm_da = DaskMacenkoNormalizer(exact=False)
    norm_da.fit(target_da)

    # Compare HERef and maxCRef
    dask_HERef, dask_maxCRef = dask.compute(norm_da.HERef, norm_da.maxCRef)
    assert np.allclose(norm_np.HERef, dask_HERef, atol=1e-1)
    assert np.allclose(norm_np.maxCRef, dask_maxCRef, atol=1e-1)

    # Compare normalization results
    result_np = norm_np.normalize(img_np)
    result_da = norm_da.normalize(img_da).compute()
    assert np.allclose(result_np, result_da, atol=3)
