import cv2 as cv
import os

data_dir = '/home/peter/datasets/ICIAR2018_BACH_Challenge/Photos'
save_dir = '/home/peter/BACH_remake/thumbs'

classes = ('Benign', 'InSitu', 'Invasive', 'Normal')
prefixes = {'Benign': 'b', 'InSitu': 'is', 'Invasive': 'iv', 'Normal': 'n'}


#############################################

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
        image = cv.imread(path)

        save_filename = prefixes[c] + i2str(i + 1) + '.png'
        save_path = os.path.join(save_dir, c, save_filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv.imwrite(save_path, image, [cv.IMWRITE_PNG_COMPRESSION, 5])
