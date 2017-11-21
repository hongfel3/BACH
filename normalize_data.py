import cv2 as cv
import os
import stainNorm_utils as stain

data_dir = '/home/peter/datasets/ICIAR2018_BACH_Challenge/Photos'
save_dir = '/home/peter/datasets/BACH_normalized'

classes = ('Benign', 'InSitu', 'Invasive', 'Normal')
prefixes = {'Benign': 'b', 'InSitu': 'is', 'Invasive': 'iv', 'Normal': 'n'}


#############################################

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
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    im = cv.cvtColor(im, cv.COLOR_RGB2BGR)
    cv.imwrite(full_save_path, im, [cv.IMWRITE_PNG_COMPRESSION, compression])


###########################################

target = read_image(os.path.join(data_dir, 'Benign', 'b027.tif'))

for c in classes:
    for i in range(100):
        filename = prefixes[c] + i2str(i + 1) + '.tif'
        print('Doing Image {}'.format(filename))
        path = os.path.join(data_dir, c, filename)
        image = read_image(path)

        normal = stain.normalize_Reinhard(image, target)

        save_filename = prefixes[c] + i2str(i + 1) + '_normalized.png'
        save_path = os.path.join(save_dir, c, save_filename)

        save_aspng(normal, save_path, compression=1)
