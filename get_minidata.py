import os

from utils import misc_utils as mu

data_dir = '/home/peter/datasets/ICIAR2018_BACH_Challenge/BACH_normalized'
save_dir = '/home/peter/datasets/ICIAR2018_BACH_Challenge/Mini_set'

classes = ('Benign', 'InSitu', 'Invasive', 'Normal')
prefixes = {'Benign': 'b', 'InSitu': 'is', 'Invasive': 'iv', 'Normal': 'n'}

###

def get_patches(img, save_path_sub):
    cnt = 1
    for i in range(5):
        for j in range(7):
            patch = img[256 * i:256 * i + 512, 256 * j:256 * j + 512]
            mu.save_aspng(patch, save_path_sub + '_patch' + mu.i2str(cnt) + '.png', compression=1)
            cnt += 1


###

for c in classes:
    filename = prefixes[c] + mu.i2str(1) + '_normalized.png'
    path = os.path.join(data_dir, c, filename)
    image = mu.read_image(path)

    sub = os.path.join(save_dir, c, prefixes[c] + mu.i2str(1))
    get_patches(image, sub)