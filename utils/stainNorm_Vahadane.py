import cv2 as cv


def notwhite_Mask(I, thresh=0.9):
    """

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
    Make rows positive if possible, else return an error.
    :param X:
    :return:
    """
    n, l = X.shape
    for i in range(n):
        sign0 = sign(X[i, 0])
        for j in range(1, l):
            if sign(X[i, j]) != sign0:
                print('Error: mixed sign rows!!')
                return None
        X[i] = sign0 * X[i]
    return X

###################################################
