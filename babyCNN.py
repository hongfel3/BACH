import tensorflow as tf
from tensorflow.contrib import keras

from utils import basic_networks
from utils import misc_utils as mu
import numpy as np

######

bs = 16

train_gen = keras.preprocessing.image.ImageDataGenerator()
# train_gen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, preprocessing_function=mu.RandRot)
train_data = train_gen.flow_from_directory('/home/peter/datasets/ICIAR2018_BACH_Challenge/Mini_set',
                                           target_size=(512, 512), batch_size=bs)

# train_gen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, preprocessing_function=mu.RandRot)
# train_data = train_gen.flow_from_directory('/home/peter/datasets/ICIAR2018_BACH_Challenge/Train_set',
#                                            target_size=(512, 512), batch_size=bs)
#
# val_gen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, preprocessing_function=mu.RandRot)
# val_data = val_gen.flow_from_directory('/home/peter/datasets/ICIAR2018_BACH_Challenge/Val_set',
#                                            target_size=(512, 512), batch_size=bs)

#####

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

tf.summary.scalar("Accuracy", accuracy)
tf.summary.scalar("Loss", loss)
summary_op = tf.summary.merge_all()

writer_train = tf.summary.FileWriter('./log_train', graph=sess.graph)
writer_val = tf.summary.FileWriter('./log_val', graph=sess.graph)

cnt_train = 1
cnt_val = 1
num_epochs = 1
for e in range(num_epochs):

    print('Epoch {}'.format(e))
    for batch, (ims, labels) in enumerate(train_data):
        if batch >= train_data.n / bs:
            break
        print('Batch number {}'.format(batch))
        _, summary = sess.run([train_step, summary_op], feed_dict={x: ims, y: labels, training: True})
        writer_train.add_summary(summary, cnt_train)
        cnt_train += 1

    print('Validation')
    acc = 0.0
    for batch, (ims, labels) in enumerate(train_data):
        if batch >= train_data.n / bs:
            break
        summary = sess.run(summary_op, feed_dict={x: ims, y: labels, training: False})
        writer_val.add_summary(summary, cnt_val)
        cnt_val += 1
