import os

import numpy as np
import tensorflow as tf

from utils import misc_utils as mu

data_dir = '/home/peter/datasets/ICIAR2018_BACH_Challenge/BACH_patches'
save_dir = '/home/peter/datasets/ICIAR2018_BACH_Challenge/BACH_tfrecords'

classes = ('Benign', 'InSitu', 'Invasive', 'Normal')
labels = {'Benign': 0, 'InSitu': 1, 'Invasive': 2, 'Normal': 3}
prefixes = {'Benign': 'b', 'InSitu': 'is', 'Invasive': 'iv', 'Normal': 'n'}


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


###

perm1 = np.random.permutation(100)

file_names = []
file_labels = []

for c in classes:
    for i in perm1[0:80]:
        for j in range(35):
            file_names.append(
                os.path.join(data_dir, c, prefixes[c] + mu.i2str(i + 1) + '_patch' + mu.i2str(j + 1) + '.png'))
            file_labels.append(labels[c])

###

total = 4 * 80 * 35
stop = int(0.75 * total)
perm2 = np.random.permutation(total)

tfrecords_filename = os.path.join(save_dir, 'training.tfrecords')
os.makedirs(os.path.dirname(tfrecords_filename), exist_ok=True)

writer = tf.python_io.TFRecordWriter(tfrecords_filename)

for i in perm2[0:stop]:
    image = mu.read_image(file_names[i])

    image_raw = image.tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'image_raw': _bytes_feature(image_raw),
        'label': _int64_feature(file_labels[i])}))

    writer.write(example.SerializeToString())

writer.close()

###

tfrecords_filename = os.path.join(save_dir, 'valid.tfrecords')
os.makedirs(os.path.dirname(tfrecords_filename), exist_ok=True)

writer = tf.python_io.TFRecordWriter(tfrecords_filename)

for i in perm2[stop:]:
    image = mu.read_image(file_names[i])

    image_raw = image.tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'image_raw': _bytes_feature(image_raw),
        'label': _int64_feature(file_labels[i])}))

    writer.write(example.SerializeToString())

writer.close()

###

test_index = perm1[80:100]
np.save('./testing_indicies.npy', test_index)
