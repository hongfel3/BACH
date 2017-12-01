import cv2 as cv
import numpy as np
from sklearn.decomposition import DictionaryLearning
from sklearn.linear_model import Lasso


# M0 = np.array([[0.60968958, 0.72542171, 0.31944007], [0.41556174, 0.83809855, 0.3534109]])

def RGB_to_OD(I):
    """

    :param I:
    :return:
    """
    I = remove_zeros(I)
    return -1 * np.log(I / 255)


def OD_to_RGB(OD):
    """

    :param OD:
    :return:
    """
    return 255 * np.exp(-1 * OD)


def normalize_rows(A):
    """
    Normalize rows of an array
    :param A:
    :return:
    """
    return A / np.linalg.norm(A, axis=1)


def remove_zeros(I):
    """
    Remove zeros
    :param I:
    :return:
    """
    mask = (I == 0)
    I[mask] = 1
    return I


def notwhite_mask(I, thresh=0.8):
    """
    Get a binary mask where true denotes 'not white'
    :param I:
    :param thresh:
    :return:
    """
    I_LAB = cv.cvtColor(I, cv.COLOR_RGB2LAB)
    L = I_LAB[:, :, 0] / 255.0
    return (L < thresh)


def sign(x):
    """
    Returns the sign of x
    :param x:
    :return:
    """
    if x > 0:
        return +1
    elif x < 0:
        return -1
    elif x == 0:
        return 0


def enforce_rows_positive(X):
    """
    Make rows positive if possible. Return is tuple of X and bool (true if success)
    :param X:
    :return:
    """
    n, l = X.shape
    for i in range(n):
        sign0 = sign(X[i, 0])
        for j in range(1, l):
            if sign(X[i, j]) != sign0:
                print('Mixed sign rows.')
                return X, False
        X[i] = sign0 * X[i]
    return X, True


######################################################

def get_stain_matrix(I, threshold=0.8, sub_sample=2000):
    """

    :param I:
    :param threshold:
    :param sub_sample:
    :return:
    """
    mask = notwhite_mask(I, thresh=threshold).reshape((-1,))
    OD = RGB_to_OD(I).reshape((-1, 3))
    OD = OD[mask]
    n = OD.shape[0]
    if n > sub_sample:
        OD = OD[np.random.choice(range(n), sub_sample)]
    dictionary_learner = DictionaryLearning(n_components=2, alpha=1, verbose=True)
    done = False
    while done == False:
        dictionary_learner.fit(OD)
        HE, done = enforce_rows_positive(dictionary_learner.components_)
    if HE[0, 0] < HE[1, 0]:
        HE = HE[[1, 0], :]
    M = np.zeros((3, 3))
    M[[0, 1], :] = HE
    M[2] = np.cross(HE[0], HE[1])
    M = normalize_rows(M)
    return HE, M


def get_H_channel(I):
    """

    :param I:
    :return:
    """
    M, _ = get_stain_matrix(I)
    OD = RGB_to_OD(I).reshape((-1, 3))
    lasso = Lasso(alpha=1, positive=True)
    lasso.fit(M.T,OD.T)
    C = lasso.coef_
    print(C.shape)
    OD = np.matmul(C[:, 0].reshape((-1, 1)), M[0].reshape((1, 3)))
    print(OD.min())
    out = OD_to_RGB(OD).reshape(I.shape)
    return 255 * out / out.max()

###################################################
