import numpy as np
import cv2 as cv
import os

data_dir = '/home/peter/BACH_remake/Photos_normalized'

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


def get_patches(image):
    stack = np.zeros((35 * 8, 512, 512, 3), dtype=np.uint8)
    counter = 0
    for i in range(5):
        for j in range(7):
            patch = image[i * 256:i * 256 + 512, j * 256:j * 256 + 512, :]
            stack[counter] = patch
            counter += 1

            patch_flip = cv.flip(patch, 1)
            stack[counter] = patch_flip
            counter += 1

            m1 = cv.getRotationMatrix2D((256, 256), 90, 1)
            rot1 = cv.warpAffine(patch, m1, (512, 512))
            stack[counter] = rot1
            counter += 1

            rot1_flip = cv.flip(rot1, 1)
            stack[counter] = rot1_flip
            counter += 1

            m2 = cv.getRotationMatrix2D((256, 256), 180, 1)
            rot2 = cv.warpAffine(patch, m2, (512, 512))
            stack[counter] = rot2
            counter += 1

            rot2_flip = cv.flip(rot2, 1)
            stack[counter] = rot2_flip
            counter += 1

            m3 = cv.getRotationMatrix2D((256, 256), 270, 1)
            rot3 = cv.warpAffine(patch, m3, (512, 512))
            stack[counter] = rot3
            counter += 1

            rot3_flip = cv.flip(rot3, 1)
            stack[counter] = rot3_flip
            counter += 1

    return stack


#############################################


for c in classes:
    for i in range(100):
        filename = prefixes[c] + i2str(i + 1) + '_normalized.tif'
        print('Doing Image {}'.format(filename))
        path = os.path.join(data_dir, c, filename)
        image = read_image(path)

        patches = get_patches(image)

        save_filename = prefixes[c] + i2str(i + 1) + '_patches.npy'
        save_path = os.path.join(os.getcwd(), 'patches', c, save_filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, patches)
