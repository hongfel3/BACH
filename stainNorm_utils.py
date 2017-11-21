"""Stain normalization utilities"""

import cv2 as cv
import numpy as np


def float2uint8(x):
    """
    Converts a float (range 0-1 or 0-255) to a uint8
    :param x:
    :return:
    """
    if x.max() <= 1.0:
        return (255.0 * x).astype(np.uint8)
    else:
        return x.astype(np.uint8)


def normalize_columns(A):
    """
    Normalize columns of an array
    :param A:
    :return:
    """
    return A / np.linalg.norm(A, axis=0)


def read_image(path):
    """
    Read an image and output as uint8 RGB (not BGR!!)
    :param path:
    :return:
    """
    im = cv.imread(path)
    return cv.cvtColor(im, cv.COLOR_BGR2RGB)


def normalize_Reinhard(patch, targetImg):
    """
    Normalize a patch stain to the target image using the method of:

    E. Reinhard, M. Adhikhmin, B. Gooch, and P. Shirley, ‘Color transfer between images’, IEEE Computer Graphics and Applications, vol. 21, no. 5, pp. 34–41, Sep. 2001.

    :param patch: a patch RGB in format uint8
    :param targetImg: a target RGB image in format uint8
    :return:
    """

    # Convert from RGB to Lab color space
    patch = cv.cvtColor(patch, cv.COLOR_RGB2LAB)
    patch = patch.astype(np.float32)
    s1, s2, s3 = cv.split(patch)
    s1 /= 2.55
    s2 -= 128.0
    s3 -= 128.0

    # Mean and Standard Deviation of Source image channels in Lab Colourspace
    mS1, sdS1 = cv.meanStdDev(s1)
    mS2, sdS2 = cv.meanStdDev(s2)
    mS3, sdS3 = cv.meanStdDev(s3)

    # Don't think this is necessary...
    # if sdS1 == 0:
    #     sdS1 = 1;
    # if sdS2 == 0:
    #     sdS2 = 1;
    # if sdS3 == 0:
    #     sdS3 = 1;

    # Convert from RGB to Lab color space
    targetImg = cv.cvtColor(targetImg, cv.COLOR_RGB2LAB)
    targetImg = targetImg.astype(np.float32)
    t1, t2, t3 = cv.split(targetImg)
    t1 /= 2.55
    t2 -= 128.0
    t3 -= 128.0

    # Mean and Standard Deviation of Target image channels in Lab Colourspace
    mT1, sdT1 = cv.meanStdDev(t1)
    mT2, sdT2 = cv.meanStdDev(t2)
    mT3, sdT3 = cv.meanStdDev(t3)

    # Normalise each channel based on statistics of source and target images
    normLab_1 = ((s1 - mS1) * (sdT1 / sdS1)) + mT1
    normLab_2 = ((s2 - mS2) * (sdT2 / sdS2)) + mT2
    normLab_3 = ((s3 - mS3) * (sdT3 / sdS3)) + mT3

    # Merge back
    normLab_1 *= 2.55
    normLab_2 += 128.0
    normLab_3 += 128.0
    normLab = cv.merge((normLab_1, normLab_2, normLab_3))
    normLab = normLab.astype(np.uint8)

    # Convert back to RGB
    norm = cv.cvtColor(normLab, cv.COLOR_LAB2RGB)

    return norm


def normalize_Macenko(patch, targetImg, Io=255, beta=0.15, alpha=1, intensity_norm=True):
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
    # Remove zeros
    mask = (patch == 0)
    patch[mask] = 1
    patch = patch.astype(np.float32)
    mask = (targetImg == 0)
    targetImg[mask] = 1
    targetImg = targetImg.astype(np.float32)

    # Get source stain matrix
    (h, w, c) = np.shape(patch)
    patch = np.reshape(patch, (h * w, c))
    OD = - np.log(patch / Io)
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
    v3 = np.cross(v1, v2)
    if v1[0] > v2[0]:
        HE = np.array([v1, v2, v3]).T
    else:
        HE = np.array([v2, v1, v3]).T
    HE = normalize_columns(HE)

    # Get target stain matrix
    targetImg = np.reshape(targetImg, (-1, 3))
    OD_target = - np.log(targetImg / Io)
    ODhat_target = (OD_target[(OD_target > beta).any(axis=1), :])
    _, V = np.linalg.eigh(np.cov(ODhat_target, rowvar=False))
    V = V[:, [2, 1]]
    if V[0, 0] < 0: V[:, 0] *= -1
    if V[0, 1] < 0: V[:, 1] *= -1
    That = np.dot(ODhat_target, V)
    phi = np.arctan2(That[:, 1], That[:, 0])
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)
    v1 = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi)]))
    v2 = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
    v3 = np.cross(v1, v2)
    if v1[0] > v2[0]:
        HE_target = np.array([v1, v2, v3]).T
    else:
        HE_target = np.array([v2, v1, v3]).T
    HE_target = normalize_columns(HE_target)

    # Get source concentrations
    Y = OD.T
    C = np.linalg.lstsq(HE, Y)[0]

    ### Modify concentrations ###

    maxC = np.percentile(C, 99, axis=1).reshape((3, 1))
    maxC[2] = 1

    Y_target = OD_target.T
    C_target = np.linalg.lstsq(HE_target, Y_target)[0]
    maxC_target = np.percentile(C_target, 99, axis=1).reshape((3, 1))
    maxC_target[2] = 1

    C = C * maxC_target / maxC

    ### Done ###

    # Final tidy up
    Inorm = Io * np.exp(- np.dot(HE_target, C))
    Inorm = np.reshape(np.transpose(Inorm), (h, w, c))
    Inorm = np.clip(Inorm, 0, 255)
    Inorm = np.array(Inorm, dtype=np.uint8)

    return Inorm
