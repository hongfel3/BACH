import glob
import os

import numpy as np

from utils import misc_utils as mu

data_dir = '/home/peter/datasets/ICIAR2018_BACH_Challenge/Train_set'

classes = ('Benign', 'InSitu', 'Invasive', 'Normal')
prefixes = {'Benign': 'b', 'InSitu': 'is', 'Invasive': 'iv', 'Normal': 'n'}

mean = np.zeros((512, 512, 3), dtype=np.float32)

full = []
for c in classes:
    sub = os.path.join(data_dir, c, '*.png')
    full += glob.glob(sub)

for f in full:
    tmp = mu.read_image(f).astype(np.float32)
    mean += tmp

mean /= len(full)

# mean = np.reshape(mean, (-1, 3))
# mean = np.mean(mean, axis=0)

np.save('mean.npy',mean)