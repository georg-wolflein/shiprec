"""
Adapted from https://github.com/wanghao14/Stain_Normalization/tree/master.
"""

from __future__ import division

import numpy as np
import spams

from .base import HENormalizer


class NumpyMacenkoNormalizer(HENormalizer):
    def __init__(self):
        self.stain_matrix_target = None
        self.target_concentrations = None

    def fit(self, target):
        target = self._standardize_brightness(target)
        self.stain_matrix_target = self._get_stain_matrix(target)
        self.target_concentrations = self._get_concentrations(target, self.stain_matrix_target)

    def transform(self, I):
        I = self._standardize_brightness(I)
        stain_matrix_source = self._get_stain_matrix(I)
        source_concentrations = self._get_concentrations(I, stain_matrix_source)
        maxC_source = np.percentile(source_concentrations, 99, axis=0).reshape((1, 2))
        maxC_target = np.percentile(self.target_concentrations, 99, axis=0).reshape((1, 2))
        source_concentrations *= maxC_target / maxC_source
        return (255 * np.exp(-1 * np.dot(source_concentrations, self.stain_matrix_target).reshape(I.shape))).astype(
            np.uint8
        )

    @classmethod
    def _standardize_brightness(cls, I):
        p = np.percentile(I, 90)
        return np.clip(I * 255.0 / p, 0, 255).astype(np.uint8)

    @classmethod
    def _remove_zeros(cls, I):
        """
        Remove zeros, replace with 1's.
        :param I: uint8 array
        :return:
        """
        mask = I == 0
        I[mask] = 1
        return I

    @classmethod
    def _RGB_to_OD(cls, I):
        """
        Convert from RGB to optical density
        :param I:
        :return:
        """
        I = cls._remove_zeros(I)
        return -1 * np.log(I / 255)

    @classmethod
    def _normalize_rows(cls, A):
        """
        Normalize rows of an array
        """
        return A / np.linalg.norm(A, axis=1)[:, None]

    @classmethod
    def _get_concentrations(cls, I, stain_matrix, lamda=0.01):
        """
        Get concentrations, a npix x 2 matrix
        :param I:
        :param stain_matrix: a 2x3 stain matrix
        :return:
        """
        OD = cls._RGB_to_OD(I).reshape((-1, 3))
        return spams.lasso(OD.T, D=stain_matrix.T, mode=2, lambda1=lamda, pos=True).toarray().T

    @classmethod
    def _get_stain_matrix(cls, I, beta=0.15, alpha=1):
        """
        Get stain matrix (2x3)
        """
        OD = cls._RGB_to_OD(I).reshape((-1, 3))
        OD = OD[(OD > beta).any(axis=1), :]
        _, V = np.linalg.eigh(np.cov(OD, rowvar=False))
        V = V[:, [2, 1]]
        if V[0, 0] < 0:
            V[:, 0] *= -1
        if V[0, 1] < 0:
            V[:, 1] *= -1
        That = np.dot(OD, V)
        phi = np.arctan2(That[:, 1], That[:, 0])
        minPhi = np.percentile(phi, alpha)
        maxPhi = np.percentile(phi, 100 - alpha)
        v1 = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi)]))
        v2 = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
        if v1[0] > v2[0]:
            HE = np.array([v1, v2])
        else:
            HE = np.array([v2, v1])
        return cls._normalize_rows(HE)
