"""
Source code adapted from: https://github.com/EIDOSLAB/torchstain/blob/main/torchstain/numpy/normalizers/macenko.py
  which was adapted from: https://github.com/schaugf/HEnorm_python
Original implementation: https://github.com/mitkovetta/staining-normalization
"""

import numpy as np

from .base import HENormalizer


def printd(name, value):
    print(f"=== {name} ===")
    print(value)


class NumpyMacenkoNormalizer(HENormalizer):
    def __init__(self):
        super().__init__()

        self.HERef = np.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]])
        self.maxCRef = np.array([1.9705, 1.0308])

    def __convert_rgb2od(self, I, Io=240, beta=0.15):
        # calculate optical density
        OD = -np.log((I.astype(float) + 1) / Io)

        # remove transparent pixels
        ODhat = OD[~np.any(OD < beta, axis=1)]

        return OD, ODhat

    def __find_HE(self, ODhat, eigvecs, alpha):
        # project on the plane spanned by the eigenvectors corresponding to the two
        # largest eigenvalues
        That = ODhat.dot(eigvecs[:, 1:3])

        phi = np.arctan2(That[:, 1], That[:, 0])

        printd("That", That)
        printd("phi", phi)

        minPhi = np.percentile(phi, alpha)
        maxPhi = np.percentile(phi, 100 - alpha)

        printd("minPhi", minPhi), printd("maxPhi", maxPhi)

        vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
        vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

        printd("vMin", vMin[:, 0])
        printd("vMax", vMax[:, 0])

        # a heuristic to make the vector corresponding to hematoxylin first and the
        # one corresponding to eosin second
        if vMin[0] > vMax[0]:
            HE = np.array((vMin[:, 0], vMax[:, 0])).T
        else:
            HE = np.array((vMax[:, 0], vMin[:, 0])).T

        return HE

    def __find_concentration(self, OD, HE):
        # rows correspond to channels (RGB), columns to OD values
        Y = np.reshape(OD, (-1, 3)).T

        # determine concentrations of the individual stains
        C = np.linalg.lstsq(HE, Y, rcond=None)[0]

        return C

    def __compute_matrices(self, I, Io, alpha, beta):
        I = I.reshape((-1, 3))

        OD, ODhat = self.__convert_rgb2od(I, Io=Io, beta=beta)

        # compute eigenvectors
        cov = np.cov(ODhat.T)
        printd("cov", cov)
        _, eigvecs = np.linalg.eigh(cov)
        printd("eigvecs", eigvecs)

        HE = self.__find_HE(ODhat, eigvecs, alpha)

        C = self.__find_concentration(OD, HE)

        # normalize stain concentrations
        maxC = np.array([np.percentile(C[0], 99), np.percentile(C[1], 99)])

        printd("HE", HE)
        printd("C", C)
        printd("maxC", maxC)

        return HE, C, maxC

    def fit(self, I, Io=240, alpha=1, beta=0.15):
        HE, _, maxC = self.__compute_matrices(I, Io, alpha, beta)

        self.HERef = HE
        self.maxCRef = maxC

    def normalize(self, I, Io=240, alpha=1, beta=0.15, stains=True):
        """Normalize staining appearence of H&E stained images

        Example use:
            see test.py

        Input:
            I: RGB input image
            Io: (optional) transmitted light intensity

        Output:
            Inorm: normalized image
            H: hematoxylin image
            E: eosin image

        Reference:
            A method for normalizing histology slides for quantitative analysis. M.
            Macenko et al., ISBI 2009
        """
        h, w, c = I.shape
        I = I.reshape((-1, 3))

        HE, C, maxC = self.__compute_matrices(I, Io, alpha, beta)

        maxC = np.divide(maxC, self.maxCRef)
        C2 = np.divide(C, maxC[:, np.newaxis])

        # recreate the image using reference mixing matrix
        Inorm = np.multiply(Io, np.exp(-self.HERef.dot(C2)))
        Inorm[Inorm > 255] = 255
        Inorm = np.reshape(Inorm.T, (h, w, c)).astype(np.uint8)

        return Inorm


if __name__ == "__main__":
    import cv2
    from pathlib import Path

    TARGET_PATCH_FILE = Path(__file__).parent.parent.parent / "normalization_template.jpg"
    target_np = cv2.cvtColor(cv2.imread(str(TARGET_PATCH_FILE)), cv2.COLOR_BGR2RGB)
    img_np = cv2.resize(target_np[200:300, 200:500], (2048, 2560))

    norm_np = NumpyMacenkoNormalizer()
    norm_np.fit(target_np)

    print("=========================")
    print("Normalizing image")
    print("=========================")

    normed_np = norm_np.normalize(img_np)
    print(normed_np)
