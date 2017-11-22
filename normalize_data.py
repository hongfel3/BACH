import cv2 as cv
import os
import stainNorm_utils as stain

data_dir = '/home/peter/datasets/ICIAR2018_BACH_Challenge/Photos'
save_dir = '/home/peter/datasets/BACH_normalized'

classes = ('Benign', 'InSitu', 'Invasive', 'Normal')
prefixes = {'Benign': 'b', 'InSitu': 'is', 'Invasive': 'iv', 'Normal': 'n'}

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
