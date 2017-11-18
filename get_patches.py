import numpy as np
import cv2 as cv
import os
import visual_utils as vu
import matplotlib.pyplot as plt

data_dir = '/home/peter/datasets/ICIAR2018_BACH_Challenge/Photos'

classes = ('Benign', 'InSitu', 'Invasive', 'Normal')
prefixes = {'Benign': 'b', 'InSitu': 'is', 'Invasive': 'iv', 'Normal': 'n'}


#############################################

def read_image(path):
    im = cv.imread(path)
    # im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    # im = im.astype(np.float32) / 255.0
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
    save_dir = os.path.join(os.getcwd(), 'patches', c)
    os.makedirs(save_dir, exist_ok=True)

    for i in range(100):
        filename = prefixes[c] + i2str(i + 1) + '.tif'
        print('Doing Image {}'.format(filename))
        path = os.path.join(data_dir, c, filename)
        image = read_image(path)

        cnt = 1
        for h in range(5):
            for w in range(7):
                patch = image[h * 256:h * 256 + 512, w * 256:w * 256 + 512, :]
                save_filename = prefixes[c] + i2str(i + 1) + '_p' + i2str(cnt) + '.png'
                cv.imwrite(os.path.join(save_dir,save_filename),patch)
                cnt+=1

                patch_flip = cv.flip(patch, 1)
                save_filename = prefixes[c] + i2str(i + 1) + '_p' + i2str(cnt) + '.png'
                cv.imwrite(os.path.join(save_dir,save_filename),patch_flip)
                cnt+=1

                m1 = cv.getRotationMatrix2D((256, 256), 90, 1)
                rot1 = cv.warpAffine(patch, m1, (512, 512))
                save_filename = prefixes[c] + i2str(i + 1) + '_p' + i2str(cnt) + '.png'
                cv.imwrite(os.path.join(save_dir,save_filename),rot1)
                cnt+=1

                rot1_flip = cv.flip(rot1, 1)
                save_filename = prefixes[c] + i2str(i + 1) + '_p' + i2str(cnt) + '.png'
                cv.imwrite(os.path.join(save_dir,save_filename),rot1_flip)
                cnt+=1

                m2 = cv.getRotationMatrix2D((256, 256), 180, 1)
                rot2 = cv.warpAffine(patch, m2, (512, 512))
                save_filename = prefixes[c] + i2str(i + 1) + '_p' + i2str(cnt) + '.png'
                cv.imwrite(os.path.join(save_dir,save_filename),rot2)
                cnt+=1

                rot2_flip = cv.flip(rot2, 1)
                save_filename = prefixes[c] + i2str(i + 1) + '_p' + i2str(cnt) + '.png'
                cv.imwrite(os.path.join(save_dir,save_filename),rot2_flip)
                cnt+=1

                m3 = cv.getRotationMatrix2D((256, 256), 270, 1)
                rot3 = cv.warpAffine(patch, m3, (512, 512))
                save_filename = prefixes[c] + i2str(i + 1) + '_p' + i2str(cnt) + '.png'
                cv.imwrite(os.path.join(save_dir,save_filename),rot3)
                cnt+=1

                rot3_flip = cv.flip(rot3, 1)
                save_filename = prefixes[c] + i2str(i + 1) + '_p' + i2str(cnt) + '.png'
                cv.imwrite(os.path.join(save_dir,save_filename),rot3_flip)
                cnt+=1

