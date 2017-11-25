import tensorflow as tf
from tensorflow.contrib import keras

from utils import basic_networks
from utils import misc_utils as mu

######

bs = 16  # batch size

lr = 1e-3  # learning rate

######

# train_gen = keras.preprocessing.image.ImageDataGenerator()
# # train_gen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, preprocessing_function=mu.RandRot)
# train_data = train_gen.flow_from_directory('/home/peter/datasets/ICIAR2018_BACH_Challenge/Mini_set',
#                                            target_size=(512, 512), batch_size=bs)

# training data
train_gen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, preprocessing_function=mu.RandRot)
train_data = train_gen.flow_from_directory('/home/peter/datasets/ICIAR2018_BACH_Challenge/Train_set',
                                           target_size=(512, 512), batch_size=bs)

# validation data
val_gen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, preprocessing_function=mu.RandRot)
val_data = val_gen.flow_from_directory('/home/peter/datasets/ICIAR2018_BACH_Challenge/Val_set',
                                       target_size=(512, 512), batch_size=bs)

######

# Build network

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

# Run

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Log the accuracy and loss
tf.summary.scalar("Accuracy", accuracy)
tf.summary.scalar("Loss", loss)
summary_op = tf.summary.merge_all()
mu.build_empty_dir('logs/train')
mu.build_empty_dir('logs/val')
writer_train = tf.summary.FileWriter('./logs/train', graph=sess.graph)
writer_val = tf.summary.FileWriter('./logs/val', graph=sess.graph)

cnt_train = 1
cnt_val = 1
num_epochs = 100
print_every = 20
mini = False
for e in range(num_epochs):

    # train
    print('Epoch {}'.format(e))
    print('Training')
    for batch, (ims, labels) in enumerate(train_data):
        if batch >= train_data.n / bs:
            break
        if batch % print_every == 0:
            print('Batch number {}'.format(batch))
        _, summary = sess.run([train_step, summary_op], feed_dict={x: ims, y: labels, training: True})
        writer_train.add_summary(summary, cnt_train)
        cnt_train += 1
        if mini == True:
            break

    # validation
    print('Validation')
    acc = 0.0
    for batch, (ims, labels) in enumerate(val_data):
        if batch >= val_data.n / bs:
            break
        if batch % print_every == 0:
            print('Batch number {}'.format(batch))
        summary = sess.run(summary_op, feed_dict={x: ims, y: labels, training: False})
        writer_val.add_summary(summary, cnt_val)
        cnt_val += 1
        if mini == True:
            break
