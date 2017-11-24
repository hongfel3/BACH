import tensorflow as tf
from tensorflow.contrib import keras
import basic_networks

######

train_gen = keras.preprocessing.image.ImageDataGenerator()
train_data = train_gen.flow_from_directory('/home/peter/datasets/ICIAR2018_BACH_Challenge/Train_set',
                                           target_size=(512, 512), batch_size=16)

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

i = 0
for im, label in train_data:
    print(label)
    i+=1
    if i==10:
        break