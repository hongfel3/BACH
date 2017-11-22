import numpy as np
import os
import misc_utils as mu

data_dir = '/home/peter/datasets/ICIAR2018_BACH_Challenge/BACH_patches'

classes = ('Benign', 'InSitu', 'Invasive', 'Normal')
prefixes = {'Benign': 'b', 'InSitu': 'is', 'Invasive': 'iv', 'Normal': 'n'}

mean = np.zeros((512, 512, 3), dtype=np.float32)

for c in classes:
    for i in range(100):
        for j in range(35):
            filename = os.path.join(data_dir, c, prefixes[c] + mu.i2str(i + 1) + '_patch' + mu.i2str(j + 1) + '.png')
            mean += mu.read_image(filename).astype(np.float32)

mean /= (4 * 100 * 35)

np.save('./mean.npy', mean)
