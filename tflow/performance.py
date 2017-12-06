import glob
import os

import numpy as np
import tensorflow as tf
from tflow import basic_network_tf

from utils import misc_utils as mu


def to_probability(x):
    temp1 = np.exp(x)
    temp2 = np.sum(temp1)
    return temp1 / temp2


# root_dir = 'home/peter/datasets/'
root_dir = '/media/peter/HDD 1/'
data_dir = root_dir + 'ICIAR2018_BACH_Challenge/Test_set'
classes = ('Benign', 'InSitu', 'Invasive', 'Normal')
prefixes = {'Benign': 'b', 'InSitu': 'is', 'Invasive': 'iv', 'Normal': 'n'}
labels = {'Benign': 0, 'InSitu': 1, 'Invasive': 2, 'Normal': 3}

x = tf.placeholder(tf.float32, [None, 512, 512, 3])
y = tf.placeholder(tf.uint8, [None, 4])
training = tf.placeholder(tf.bool)

out = basic_network_tf.basic_CNN(x, training=training)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, './saves/best_model.ckpt')

n_patch = 0
n_ims = 0
mean_patch_accuracy = 0.0
mean_image_accuracy = 0.0
for c in classes:
    directory = os.path.join(data_dir, c, '*')
    print(directory)
    files = glob.glob(directory)
    for f in files:
        print(f)
        n_ims += 1
        im = mu.read_image(f)
        im = im.astype(np.float32)
        patch_predictions = []
        for i in range(3):
            for j in range(4):
                n_patch += 1
                patch = im[i * 512:(i + 1) * 512, j * 512:(j + 1) * 512, :]
                patch = np.reshape(patch, (1, 512, 512, 3))
                output = sess.run(out, feed_dict={x: patch, training: False})
                output = output.reshape((4,))
                output = to_probability(output)
                patch_class_prediction = np.argmax(output)
                patch_predictions.append(patch_class_prediction)
                if patch_class_prediction == labels[c]:
                    mean_patch_accuracy += 1
        print(patch_predictions)
        im_class_prediction = max(set(patch_predictions), key=patch_predictions.count)
        if im_class_prediction == labels[c]:
            mean_image_accuracy += 1

mean_patch_accuracy /= n_patch
mean_image_accuracy /= n_ims

print(mean_patch_accuracy)
print(mean_image_accuracy)
