"""
Uses the spams package:

http://spams-devel.gforge.inria.fr/index.html

Use with python via e.g https://anaconda.org/conda-forge/python-spams
"""

import numpy as np
import spams


def remove_zeros(I):
    """
    Remove zeros
    :param I:
    :return:
    """
    mask = (I == 0)
    I[mask] = 1
    return I


def normalize_rows(A):
    """
    Normalize rows of an array
    :param A:
    :return:
    """
    return A / np.linalg.norm(A, axis=1)[:, None]


def RGB_to_OD(I):
    """
    Convert from RGB to optical density
    :param I:
    :return:
    """
    I = remove_zeros(I)
    return -1 * np.log(I / 255)


def normalize_columns(A):
    """
    Normalize columns of an array
    :param A:
    :return:
    """
    return A / np.linalg.norm(A, axis=0)


def get_stain_matrix(I, beta=0.15, alpha=1):
    """
    Get stain matrix (2x3)
    :param I:
    :param beta:
    :param alpha:
    :return:
    """
    OD = RGB_to_OD(I).reshape((-1, 3))
    ODhat = (OD[(OD > beta).any(axis=1), :])
    _, V = np.linalg.eigh(np.cov(ODhat, rowvar=False))
    V = V[:, [2, 1]]
    if V[0, 0] < 0: V[:, 0] *= -1
    if V[0, 1] < 0: V[:, 1] *= -1
    That = np.dot(ODhat, V)
    phi = np.arctan2(That[:, 1], That[:, 0])
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)
    v1 = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi)]))
    v2 = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
    if v1[0] > v2[0]:
        HE = np.array([v1, v2])
    else:
        HE = np.array([v2, v1])
    return normalize_rows(HE)


def get_concentrations(I, stain_matrix):
    """
    Get concentrations, a npix x 2 matrix
    :param I:
    :param stain_matrix:
    :return:
    """
    OD = RGB_to_OD(I).reshape((-1, 3))
    return spams.lasso(OD.T, D=stain_matrix.T, mode=2, lambda1=0.01, pos=True).toarray().T


####################################################

def normalize_Macenko(patch, targetImg, beta=0.15, alpha=1):
    """
    Stain normalization based on the method of:

    M. Macenko et al., ‘A method for normalizing histology slides for quantitative analysis’, in 2009 IEEE International Symposium on Biomedical Imaging: From Nano to Macro, 2009, pp. 1107–1110.

    Adapted from:

    T. Araújo et al., ‘Classification of breast cancer histology images using Convolutional Neural Networks’, PLOS ONE, vol. 12, no. 6, p. e0177544, Jun. 2017.

    and the MATLAB toolbox available at:

    https://warwick.ac.uk/fac/sci/dcs/research/tia/software/sntoolbox/
    :param patch: a patch RGB in format uint8
    :param targetImg: a target RGB image in format uint8
    :param Io:
    :param beta:
    :param alpha:
    :param intensity_norm:
    :return:
    """

    HE = get_stain_matrix(patch, beta=beta, alpha=alpha)

    HE_target = get_stain_matrix(targetImg, beta=beta, alpha=alpha)

    # Get source concentrations
    C = get_concentrations(patch, HE)

    ### Modify concentrations ###
    maxC = np.percentile(C, 99, axis=0).reshape((1, 2))
    C_target = get_concentrations(targetImg, HE_target)
    maxC_target = np.percentile(C_target, 99, axis=0).reshape((1, 2))
    C = C * maxC_target / maxC

    # Final tidy up
    Inorm = 255 * np.exp(- np.dot(C, HE_target))
    Inorm = np.reshape(Inorm, patch.shape)
    Inorm = np.clip(Inorm, 0, 255)
    Inorm = np.array(Inorm, dtype=np.uint8)

    return Inorm
