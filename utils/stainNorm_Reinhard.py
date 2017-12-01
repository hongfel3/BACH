"""Stain normalization utilities"""

import cv2 as cv
import numpy as np


def normalize_columns(A):
    """
    Normalize columns of an array
    :param A:
    :return:
    """
    return A / np.linalg.norm(A, axis=0)


def notBackground_Mask(I, thresh=0.9):
    """

    :param I:
    :return:
    """
    I_LAB = cv.cvtColor(I, cv.COLOR_RGB2LAB)
    L = I_LAB[:, :, 0] / 255.0
    return (L < thresh)


#########################################################################

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


