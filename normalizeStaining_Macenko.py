import numpy as np


def normalize_columns(A):
    temp = np.zeros(A.shape)
    n = A.shape[1]
    for i in range(n):
        v = A[:, i]
        v = v / np.linalg.norm(v)
        temp[:, i] = v
    return temp


def normalizeStaining(I):
    """
    Stain normalization based on the method of:

    M. Macenko et al., ‘A method for normalizing histology slides for quantitative analysis’, in 2009 IEEE International Symposium on Biomedical Imaging: From Nano to Macro, 2009, pp. 1107–1110.

    Adapted from:

    T. Araújo et al., ‘Classification of breast cancer histology images using Convolutional Neural Networks’, PLOS ONE, vol. 12, no. 6, p. e0177544, Jun. 2017.

    :param I:
    :return:
    """
    mask = (I == 0)
    I[mask] = 1
    I = I.astype(np.float32)

    Io = 255.0  # 240?
    beta = 0.15
    alpha = 5

    HE_Ref = np.array([[0.5626, 0.7201, 0.4062], [0.2159, 0.8012, 0.5581]]).T
    HE_Ref = normalize_columns(HE_Ref)
    print(HE_Ref)

    (h, w, c) = np.shape(I)
    I = np.reshape(I, (h * w, c))

    OD = - np.log(I / Io)
    ODhat = (OD[(np.logical_not((OD < beta).any(axis=1))), :])

    _, V = np.linalg.eigh(np.cov(ODhat, rowvar=False))
    V = V[:, [2, 1, 0]]
    V = V[:, [0, 1]]
    if V[0, 0] < 0: V[:, 0] *= -1
    if V[0, 1] < 0: V[:, 1] *= -1

    That = np.dot(ODhat, V)
    phi = np.arctan2(That[:, 1], That[:, 0])
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)

    vE = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi)]))
    vH = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi)]))

    HE = np.array([vH, vE]).T
    HE = normalize_columns(HE)

    Y = OD.T
    C = np.linalg.lstsq(HE, Y)[0]

    # maxC_Ref = np.array([1.9705, 1.0308]).reshape((2, 1))
    # maxC = np.percentile(C, 99, axis=1).reshape((2, 1))
    # C = C * maxC_Ref / maxC

    Inorm = Io * np.exp(- np.dot(HE_Ref, C))

    Inorm = np.reshape(np.transpose(Inorm), (h, w, c))
    Inorm = np.clip(Inorm, 0, 255)
    Inorm = np.array(Inorm, dtype=np.uint8)

    return Inorm
