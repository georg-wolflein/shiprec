import dask
import dask.array as da
import numpy as np
from functools import partial

from .base import HENormalizer


@dask.delayed
def delayed_eigh(X):
    """Delayed version of `np.linalg.eigh`."""
    _, eigh = np.linalg.eigh(X)
    return eigh


def _cov(X: da.Array, N: da.Array) -> da.Array:
    """Compute covariance matrix of X.

    Args:
        X: array of shape (N, D) where N is the number of samples and D is the number of features.
        N: number of samples as an array of shape ().

    Unlike `da.cov(X)`, this function doesn't break when X.shape[0] is unknown at graph construction time.
    """

    # Compute mean of each column
    mean = X.mean(axis=-1, keepdims=True)

    # Subtract mean from each column
    X_centered = X - mean

    # Compute covariance matrix
    cov = np.dot(X_centered, X_centered.T) / (N - 1)
    return cov


def _np_percentile(a: da.Array, q: float) -> da.Array:
    """Compute exact percentile of a.

    Args:
        a: array of shape (N,).
        q: percentile to compute.

    Returns:
        percentile of a as an array of shape ().

    Note:
        Unlike `da.percentile(a, q)`, this function produces the exact percentile, not an approximation.
    """
    return da.from_delayed(dask.delayed(np.percentile)(a, q), shape=(), dtype=a.dtype)


def _tdigest_percentile(a: da.Array, q: float) -> da.Array:
    return da.percentile(a, q, internal_method="tdigest").squeeze(-1)


_PERCENTILE_METHODS = {
    "np": _np_percentile,
    "tdigest": _tdigest_percentile,
}


class DaskMacenkoNormalizer(HENormalizer):
    def __init__(self, exact: bool = False):
        """Macenko stain normalization implemented using dask.

        Args:
            exact: whether to compute percentiles exactly or using an approximation.

        Note:
            The exact method is much slower than the approximate method, but produces the correct result compared to the numpy implementation.
            The inexact method usually isn't far off, however, especially for large tile chunk sizes.
        """
        super().__init__()
        self.HERef = da.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]])
        self.maxCRef = da.array([1.9705, 1.0308])

        self._exact = exact

    @property
    def _percentile(self):
        return _PERCENTILE_METHODS["np" if self._exact else "tdigest"]

    def _convert_rgb2od(self, I, Io=240, beta=0.15):
        # Calculate optical density
        OD = -da.log((I.astype(float) + 1) / Io)

        # Remove transparent pixels
        mask = ~da.any(OD < beta, axis=1)
        ODhat = OD[mask]

        ODhatN = mask.sum()

        return OD, ODhat, ODhatN

    def _find_HE(self, ODhat, eigvecs, alpha):
        # Project on the plane spanned by the eigenvectors corresponding to the two largest eigenvalues
        That = ODhat.dot(eigvecs[:, 1:3])

        phi = da.arctan2(That[:, 1], That[:, 0])

        minPhi = self._percentile(phi, alpha)
        maxPhi = self._percentile(phi, 100 - alpha)

        vMin = eigvecs[:, 1:3].dot(da.expand_dims(da.stack([da.cos(minPhi), da.sin(minPhi)], axis=0), axis=0).T)
        vMax = eigvecs[:, 1:3].dot(da.expand_dims(da.stack([da.cos(maxPhi), da.sin(maxPhi)], axis=0), axis=0).T)

        vMin = vMin[:, 0]
        vMax = vMax[:, 0]

        # The next few lines are a heuristic to make the vector corresponding to hematoxylin first and the one corresponding to eosin second.
        # It is equivalent to the following code:
        # HE = da.array((vMin, vMax)).T if vMin[0] > vMax[0] else da.array((vMax, vMin)).T

        stacked = da.stack([vMin, vMax], axis=0)
        is_bigger = vMin[0] > vMax[0]
        HE = da.where(is_bigger, stacked, stacked[::-1]).T
        HE = HE.rechunk(-1)

        return HE

    def _find_concentration(self, OD, HE):
        # Rows correspond to channels (RGB), columns to OD values
        Y = da.reshape(OD, (-1, 3)).T

        # Determine concentrations of the individual stains
        C = da.linalg.lstsq(HE, Y)[0]

        return C

    def _compute_matrices(self, I, Io, alpha, beta):
        I = I.reshape((-1, 3))

        OD, ODhat, ODhatN = self._convert_rgb2od(I, Io=Io, beta=beta)

        # Compute eigenvectors
        cov = _cov(ODhat.T, ODhatN)

        # Now cov has shape (3, 3), so we can compute eigenvectors locally
        cov = cov.rechunk(-1)
        eigvecs = delayed_eigh(cov)
        eigvecs = da.from_delayed(eigvecs, (3, 3), dtype="float")

        HE = self._find_HE(ODhat, eigvecs, alpha)
        C = self._find_concentration(OD, HE)

        # Normalize stain concentrations
        maxC = da.stack(
            [
                self._percentile(C[0], 99),
                self._percentile(C[1], 99),
            ]
        )

        return HE, C, maxC

    def fit(self, I, Io=240, alpha=1, beta=0.15, persist: bool = True):
        HE, _, maxC = self._compute_matrices(I, Io, alpha, beta)

        if persist:
            HE, maxC = dask.persist(HE, maxC)

        self.HERef = HE
        self.maxCRef = maxC

    def normalize_flattened_image(self, I: da.Array, Io=240, alpha=1, beta=0.15):
        # I should be of shape (N, 3)
        HE, C, maxC = self._compute_matrices(I, Io, alpha, beta)

        maxC = da.divide(maxC, self.maxCRef)
        C2 = da.divide(C, da.expand_dims(maxC, axis=-1))

        # Recreate the image using reference mixing matrix
        Inorm = da.multiply(Io, da.exp(-self.HERef.dot(C2)))
        Inorm[Inorm > 255] = 255
        Inorm = Inorm.astype(I.dtype)
        Inorm = Inorm.T
        Inorm = Inorm.rechunk(I.chunks)
        return Inorm, HE, maxC

    def normalize(self, I, Io=240, alpha=1, beta=0.15):
        I_chunks = I.chunks
        h, w, c = I.shape
        I = I.reshape((-1, 3))

        Inorm, HE, maxC = self.normalize_flattened_image(I, Io, alpha, beta)

        Inorm = da.reshape(Inorm, (h, w, c))
        return Inorm, HE, maxC

    def __getstate__(self) -> object:
        return {
            "HERef": self.HERef,
            "maxCRef": self.maxCRef,
            "_exact": self._exact,
        }

    def __setstate__(self, state: object) -> None:
        self.HERef = state["HERef"]
        self.maxCRef = state["maxCRef"]
        self._exact = state["_exact"]
