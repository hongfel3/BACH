import tensorflow as tf


def conv_relu3x3(x, scope, num_filters, padding='VALID', train=True):
    with tf.variable_scope(scope):
        h1 = tf.layers.conv2d(x, filters=num_filters, kernel_size=(3, 3), strides=(1, 1), padding=padding, name='conv',
                              kernel_initializer=tf.contrib.layers.xavier_initializer())
        h2 = tf.layers.batch_normalization(h1, scale=True, center=True, name='bn', training=train, momentum=0.9)
        return tf.nn.relu(h2)


def max_pool3x3(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='VALID')


def max_pool2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


def dense_relu(x, scope, num_out, train=True):
    with tf.variable_scope(scope):
        h1 = tf.layers.dense(x, units=num_out, activation=None, name='fc',
                             kernel_initializer=tf.contrib.layers.xavier_initializer())
        h2 = tf.layers.batch_normalization(h1, scale=True, center=True, name='bn', training=train, momentum=0.9)
        h3 = tf.nn.relu(h2)
        return tf.layers.dropout(h3, rate=0.5, training=train)


def dense(x, scope, num_out):
    with tf.variable_scope(scope):
        h1 = tf.layers.dense(x, units=num_out, activation=None, name='fc',
                             kernel_initializer=tf.contrib.layers.xavier_initializer())
        return h1


def basic_CNN(x, training):
    l0 = tf.layers.batch_normalization(x, scale=True, center=True, name='bn', train=training, momentum=0.9)
    l1 = conv_relu3x3(l0, 'conv1', 16, train=training)
    l2 = max_pool3x3(l1)
    l3 = conv_relu3x3(l2, 'conv2', 32, train=training)
    l4 = max_pool2x2(l3)
    l5 = conv_relu3x3(l4, 'conv3', 64, padding='SAME', train=training)
    l6 = max_pool2x2(l5)
    l7 = conv_relu3x3(l6, 'conv4', 64, padding='SAME', train=training)
    l8 = max_pool3x3(l7)
    l9 = conv_relu3x3(l8, 'conv5', 32, train=training)
    l10 = max_pool3x3(l9)
    flat = tf.contrib.layers.flatten(l10)
    f1 = dense_relu(flat, 'fc1', 256, train=training)
    f2 = dense_relu(f1, 'fc2', 128, train=training)
    logits = dense(f2, 'fc3', 4)
    return logits
