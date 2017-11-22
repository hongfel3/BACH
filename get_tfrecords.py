import misc_utils as mu
import numpy as np
import tensorflow as tf
import os

data_dir = '/home/peter/datasets/BACH_patches'

classes = ('Benign', 'InSitu', 'Invasive', 'Normal')
labels = {'Benign': 0, 'InSitu': 1, 'Invasive': 2, 'Normal': 3}
prefixes = {'Benign': 'b', 'InSitu': 'is', 'Invasive': 'iv', 'Normal': 'n'}


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


###

tfrecords_filename = 'dataset.tfrecords'

writer = tf.python_io.TFRecordWriter(tfrecords_filename)

for c in classes:
    for i in range(1):
        for j in range(35):
            path = os.path.join(data_dir, c, prefixes[c] + mu.i2str(i + 1) + '_patch' + mu.i2str(j + 1) + '.png')
            image = mu.read_image(path)

            height, width, _ = image.shape

            image_raw = image.tostring()

            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'image_raw': _bytes_feature(image_raw),
                'label': _int64_feature(labels[c])}))

            writer.write(example.SerializeToString())

writer.close()
