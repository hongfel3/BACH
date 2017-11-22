import tensorflow as tf
import numpy as np
import basic_networks

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

x = tf.placeholder(tf.float32, [None, 512, 512, 3])
y = tf.placeholder(tf.uint8, [None, 4])
training = tf.placeholder(tf.bool, name='train')

out = basic_networks.basic_CNN(x, training=training)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(out, 1)), 'float32'))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

######

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Check we can get 100% accuracy on training set!
for i in range(20):
    print('i={}'.format(i))
    sess.run(train_step, feed_dict={x: X, y: Y, training: True})

    if i % 2 == 0:
        print(sess.run(accuracy, {x: X, y: Y, training: False}))
