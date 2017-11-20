import cv2 as cv
import os
import normalizeStaining_Macenko_util as sn
import numpy as np

data_dir = '/home/peter/datasets/ICIAR2018_BACH_Challenge/Photos'
save_dir = '/home/peter/BACH_remake/thresholded'

classes = ('Benign', 'InSitu', 'Invasive', 'Normal')
prefixes = {'Benign': 'b', 'InSitu': 'is', 'Invasive': 'iv', 'Normal': 'n'}


#############################################

def read_image(path):
    im = cv.imread(path)
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    return im


def i2str(i):
    s = str(i)
    if len(s) == 1:
        return '00' + s
    elif len(s) == 2:
        return '0' + s
    else:
        return s


#############################################


for c in classes:
    for i in range(100):
        filename = prefixes[c] + i2str(i + 1) + '.tif'
        print('Doing Image {}'.format(filename))
        path = os.path.join(data_dir, c, filename)
        image = read_image(path)

        mask = (image == 0)
        image[mask] = 1
        image = image.astype(np.float32)

        Io = 240
        OD = -np.log(image / Io)

        # print(np.mean(OD))
        # print(np.std(OD))
        # mask = (OD > np.mean(OD)/2).any(axis=2)
        mask = (OD > 0.15).any(axis=2)

        save_filename = prefixes[c] + i2str(i + 1) + '.png'
        save_path = os.path.join(save_dir, c, save_filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv.imwrite(save_path, 255 * mask.astype(np.uint8), [cv.IMWRITE_PNG_COMPRESSION, 5])
