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

X=np.zeros((280*4,512,512,3),dtype=np.uint8)
Y=np.zeros((280*4,1),dtype=np.uint8)

cnt=0
for c in classes:
    filename = prefixes[c] + '001_normalized.tif'
    print('Doing Image {}'.format(filename))
    path = os.path.join(data_dir, c, filename)
    image = read_image(path)

    patches = get_patches(image)

    X[cnt * 280:(cnt + 1) * 280] = patches
    Y[cnt * 280:(cnt + 1) * 280] = cnt

    cnt+=1

np.save('babyX.npy',X)
np.save('babyY.npy',Y)