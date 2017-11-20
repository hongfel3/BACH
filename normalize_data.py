import cv2 as cv
import os
import normalizeStaining_Macenko_util as sn

data_dir = '/home/peter/datasets/ICIAR2018_BACH_Challenge/Photos'
save_dir = '/home/peter/BACH_remake/Photos_normalized'

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


def save_image(im, path):
    im = cv.cvtColor(im, cv.COLOR_RGB2BGR)
    cv.imwrite(path, im)


###########################################

target=read_image(os.path.join(data_dir,'Benign','b027.tif'))

for c in classes:
    for i in range(100):
        filename = prefixes[c] + i2str(i + 1) + '.tif'
        print('Doing Image {}'.format(filename))
        path = os.path.join(data_dir, c, filename)
        image = read_image(path)

        normal=sn.normalizeStaining(image,target,Io=240)

        save_filename = prefixes[c] + i2str(i + 1) + '_normalized.tif'
        save_path = os.path.join(save_dir, c, save_filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_image(normal, save_path)
