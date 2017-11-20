import numpy as np


def normalize_columns(A):
    """
    Normalize columns of an array
    :param A:
    :return:
    """
    temp = np.zeros(A.shape)
    n = A.shape[1]
    for i in range(n):
        v = A[:, i]
        v = v / np.linalg.norm(v)
        temp[:, i] = v
    return temp


def normalizeStaining(I, target, Io=255, beta=0.15, alpha=1):
    """
    Stain normalization based on the method of:

    M. Macenko et al., ‘A method for normalizing histology slides for quantitative analysis’, in 2009 IEEE International Symposium on Biomedical Imaging: From Nano to Macro, 2009, pp. 1107–1110.

    Adapted from:

    T. Araújo et al., ‘Classification of breast cancer histology images using Convolutional Neural Networks’, PLOS ONE, vol. 12, no. 6, p. e0177544, Jun. 2017.

    and the MATLAB toolbox available at:

    https://warwick.ac.uk/fac/sci/dcs/research/tia/software/sntoolbox/
    :param I:
    :param target:
    :param alpha:
    :param beta:
    :param Io:
    :return:
    """
    # Remove zeros
    mask = (I == 0)
    I[mask] = 1
    I = I.astype(np.float32)
    mask = (target == 0)
    target[mask] = 1
    target = target.astype(np.float32)

    # Get source stain matrix
    (h, w, c) = np.shape(I)
    I = np.reshape(I, (h * w, c))
    OD = - np.log(I / Io)
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
    target = np.reshape(target, (-1, 3))
    OD_target = - np.log(target / Io)
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
