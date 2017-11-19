import numpy as np


def normalizeStaining(I):
    """
    Stain normalization via the method of:

    M. Macenko et al., ‘A method for normalizing histology slides for quantitative analysis’, in 2009 IEEE International Symposium on Biomedical Imaging: From Nano to Macro, 2009, pp. 1107–1110.

    Adapted from:

    T. Araújo et al., ‘Classification of breast cancer histology images using Convolutional Neural Networks’, PLOS ONE, vol. 12, no. 6, p. e0177544, Jun. 2017.

    :param I:
    :return:
    """

    Io = 240
    beta = 0.15
    alpha = 1

    HERef = np.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]])
    maxCRef = np.array([1.9705, 1.0308])

    (h, w, c) = np.shape(I)
    I = np.reshape(I, (h * w, c))

    OD = - np.log((I + 1) / Io)
    ODhat = (OD[(np.logical_not((OD < beta).any(axis=1))), :])

    _, V = np.linalg.eigh(np.cov(ODhat, rowvar=False))
    V = V[:, [2, 1, 0]]

    Vec = - np.transpose(np.array([V[:, 1], V[:, 0]]))
    That = np.dot(ODhat, Vec)
    phi = np.arctan2(That[:, 1], That[:, 0])
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)
    vMin = np.dot(Vec, np.array([np.cos(minPhi), np.sin(minPhi)]))
    vMax = np.dot(Vec, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
    if vMin[0] > vMax[0]:
        HE = np.array([vMin, vMax])
    else:
        HE = np.array([vMax, vMin])

    HE = np.transpose(HE)
    Y = np.transpose(OD)

    C = np.linalg.lstsq(HE, Y)
    maxC = np.percentile(C[0], 99, axis=1)

    C = C[0] / maxC[:, None]
    C = C * maxCRef[:, None]
    Inorm = Io * np.exp(- np.dot(HERef, C))
    Inorm = np.reshape(np.transpose(Inorm), (h, w, c))
    Inorm = np.clip(Inorm, 0, 255)
    Inorm = np.array(Inorm, dtype=np.uint8)
    return Inorm


import cv2 as cv
import matplotlib.pyplot as plt

I = cv.cvtColor(cv.imread('im1.tif'), cv.COLOR_BGR2RGB)
plt.figure(1)
plt.imshow(I)
I = np.array(I, dtype=np.uint8)
I = np.array(I, dtype=np.float64)
x = normalizeStaining(I)
plt.figure(2)
plt.imshow(x)
plt.show()
