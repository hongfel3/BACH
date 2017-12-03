"""
Stain normalization inspired by method of:

A. Vahadane et al., ‘Structure-Preserving Color Normalization and Sparse Stain Separation for Histological Images’, IEEE Transactions on Medical Imaging, vol. 35, no. 8, pp. 1962–1971, Aug. 2016.

Uses the spams package:

http://spams-devel.gforge.inria.fr/index.html

Use with python via e.g https://anaconda.org/conda-forge/python-spams
"""

import stainNorm_utils as ut
import numpy as np
import spams


def get_stain_matrix(I, threshold=0.8, sub_sample=10000):
    """
    Get 2x3 stain matrix. First row H and second row E
    :param I:
    :param threshold:
    :param sub_sample:
    :return:
    """
    mask = ut.notwhite_mask(I, thresh=threshold).reshape((-1,))
    OD = ut.RGB_to_OD(I).reshape((-1, 3))
    OD = OD[mask]
    n = OD.shape[0]
    if n > sub_sample:
        OD = OD[np.random.choice(range(n), sub_sample)]
    dictionary = spams.trainDL(OD.T, K=2, lambda1=1, mode=2, modeD=0, posAlpha=True, posD=True).T
    if dictionary[0, 0] < dictionary[1, 0]:
        dictionary = dictionary[[1, 0], :]
    dictionary = ut.normalize_rows(dictionary)
    return dictionary


def get_concentrations(I, stain_matrix):
    """
    Get concentrations, a npix x 2 matrix
    :param I:
    :param stain_matrix:
    :return:
    """
    OD = ut.RGB_to_OD(I).reshape((-1, 3))
    return spams.lasso(OD.T, D=stain_matrix.T, mode=2, lambda1=0.01, pos=True).toarray().T


def normalize_Vahadane(I, targetImg):
    """
    Normalize a patch  to stain of target
    :param I:
    :param targetImg:
    :return:
    """
    stain_matrix_source = get_stain_matrix(I)
    stain_matrix_target = get_stain_matrix(targetImg)
    source_concentrations = get_concentrations(I, stain_matrix_source)
    return (255 * np.exp(-1 * np.dot(source_concentrations, stain_matrix_target).reshape(I.shape))).astype(np.uint8)


def HandE(I):
    """
    Get H and E (deconvolve)
    :param I:
    :return:
    """
    h, w, c = I.shape
    stain_matrix_source = get_stain_matrix(I)
    source_concentrations = get_concentrations(I, stain_matrix_source)
    H = source_concentrations[:, 0].reshape(h, w)
    H = np.exp(-1 * H)
    E = source_concentrations[:, 1].reshape(h, w)
    E = np.exp(-1 * E)
    return H, E
