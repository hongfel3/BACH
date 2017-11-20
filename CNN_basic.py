import tensorflow as tf
import numpy as np

######

X = np.load('babyX.npy')
Y = np.eye(4)[np.load('babyY.npy').reshape(-1)]
N = np.shape(X)[0]
sub = 10
idx = np.random.choice(range(N), sub)
X = X[idx]
Y = Y[idx]
X = X.astype(np.float32)
X -= np.mean(X, axis=0)

######

lr = 1e-3


######


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


######

tf.reset_default_graph()
x = tf.placeholder(tf.float32, [None, 512, 512, 3])
y = tf.placeholder(tf.uint8, [None, 4])
training = tf.placeholder(tf.bool, name='train')

l1 = conv_relu3x3(x, 'conv1', 16, train=training)
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
out = dense(f2, 'fc3', 4)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(out, 1)), 'float32'))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

######

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Check we can get 100% accuracy on training set!
for i in range(10):
    print('i={}'.format(i))
    sess.run(train_step, feed_dict={x: X, y: Y, training: True})

    if i % 2 == 0:
        print(sess.run(accuracy, {x: X, y: Y, training: False}))
