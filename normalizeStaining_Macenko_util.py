import numpy as np


def normalize_columns(A):
    temp = np.zeros(A.shape)
    n = A.shape[1]
    for i in range(n):
        v = A[:, i]
        v = v / np.linalg.norm(v)
        temp[:, i] = v
    return temp


def normalizeStaining(I, target):
    """
    Stain normalization based on the method of:

    M. Macenko et al., ‘A method for normalizing histology slides for quantitative analysis’, in 2009 IEEE International Symposium on Biomedical Imaging: From Nano to Macro, 2009, pp. 1107–1110.

    Adapted from:

    T. Araújo et al., ‘Classification of breast cancer histology images using Convolutional Neural Networks’, PLOS ONE, vol. 12, no. 6, p. e0177544, Jun. 2017.

    :param I:
    :param target:
    :return:
    """
    mask = (I == 0)
    I[mask] = 1
    I = I.astype(np.float32)

    mask = (target == 0)
    target[mask] = 1
    target = target.astype(np.float32)

    Io = 255.0
    beta = 0.15
    alpha = 1

    (h, w, c) = np.shape(I)
    I = np.reshape(I, (h * w, c))
    OD = - np.log(I / Io)
    ODhat = (OD[(np.logical_not((OD < beta).any(axis=1))), :])
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

    target = np.reshape(target, (h * w, c))
    OD_target = - np.log(target / Io)
    ODhat_target = (OD_target[(np.logical_not((OD_target < beta).any(axis=1))), :])
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

    Y = OD.T
    C = np.linalg.lstsq(HE, Y)[0]

    Inorm = Io * np.exp(- np.dot(HE_target, C))

    Inorm = np.reshape(np.transpose(Inorm), (h, w, c))
    Inorm = np.clip(Inorm, 0, 255)
    Inorm = np.array(Inorm, dtype=np.uint8)

    return Inorm
