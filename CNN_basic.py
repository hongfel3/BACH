import tensorflow as tf
import numpy as np

X = np.load('babyX.npy')
Y = np.eye(4)[np.load('babyY.npy').reshape(-1)]

N = np.shape(X)[0]
bs = 10


######

def conv_relu3x3(x, scope, num_filters, padding='valid'):
    with tf.variable_scope(scope):
        h1 = tf.layers.conv2d(x, filters=num_filters, kernel_size=(3, 3), strides=(1, 1), padding=padding, name='conv')
        return tf.nn.relu(h1)


def max_pool3x3(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='VALID')


def max_pool2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


def dense_relu(x, scope, num_out):
    with tf.variable_scope(scope):
        h1 = tf.layers.dense(x, units=num_out, activation=None, name='fc')
        return tf.nn.relu(h1)


######

tf.reset_default_graph()
x = tf.placeholder(tf.float32, [None, 512, 512, 3])
y = tf.placeholder(tf.uint8, [None, 4])

l1 = conv_relu3x3(x, 'conv1', 16)
l2 = max_pool3x3(l1)
l3 = conv_relu3x3(l2, 'conv2', 32)
l4 = max_pool2x2(l3)
l5 = conv_relu3x3(l4, 'conv3', 64, padding='same')
l6 = max_pool2x2(l5)
l7 = conv_relu3x3(l6, 'conv4', 64, padding='same')
l8 = max_pool3x3(l7)
l9 = conv_relu3x3(l8, 'conv5', 32)
l10 = max_pool3x3(l9)
flat = tf.contrib.layers.flatten(l10)
f1 = dense_relu(flat, 'fc1', 256)
f2 = dense_relu(f1, 'fc2', 128)
out = tf.layers.dense(f2, units=4, activation=None, name='fc3')

######

sess = tf.Session()
sess.run(tf.global_variables_initializer())

idx = np.random.choice(range(N), bs)

f = sess.run(out, feed_dict={x: X[idx], y: Y[idx]})