import cv2 as cv
import os
import random
from skimage import transform


def empty_dir(folder):
    """
    Empty folder
    :param folder:
    :return:
    """
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        if os.path.isfile(file_path):
            os.remove(file_path)


def RandRot(im):
    """
    Random rotation
    :param im:
    :return:
    """
    rand = random.choice(range(4))
    return transform.rotate(im, 90 * rand, preserve_range=True)


def read_image(path):
    """
    Read an image to RGB uint8
    :param path:
    :return:
    """
    im = cv.imread(path)
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    return im


def i2str(i):
    """
    Convert an integer <=999 to a string
    :param i:
    :return:
    """
    s = str(i)
    if len(s) == 1:
        return '00' + s
    elif len(s) == 2:
        return '0' + s
    else:
        return s


def save_aspng(im, full_save_path, compression=3):
    """
    Save an image as png with optional compression (not sure this works!). Specify full_save_path e.g. '/home/peter/mypic.png'. Directory is built if not present.
    :param im:
    :param full_save_path:
    :param compression:
    :return:
    """
    os.makedirs(os.path.dirname(full_save_path), exist_ok=True)
    im = cv.cvtColor(im, cv.COLOR_RGB2BGR)
    cv.imwrite(full_save_path, im, [cv.IMWRITE_PNG_COMPRESSION, compression])
