import os

import tensorflow as tf
from tensorflow import basic_network_tf
from tensorflow.contrib import keras

from utils import misc_utils as mu

######

bs = 64  # batch size

lr = 1e-3  # learning rate

######

# where is data stored?
root_dir = '/home/peter/datasets/ICIAR2018_BACH_Challenge'
root_dir = '/media/peter/HDD 1/ICIAR2018_BACH_Challenge'

mini = True

if mini == True:
    # training data
    train_gen = keras.preprocessing.image.ImageDataGenerator()
    train_data = train_gen.flow_from_directory(os.path.join(root_dir, 'Mini_set'),
                                               target_size=(512, 512), batch_size=bs)

    # validation data
    val_gen = keras.preprocessing.image.ImageDataGenerator()
    val_data = val_gen.flow_from_directory(os.path.join(root_dir, 'Mini_set'),
                                           target_size=(512, 512), batch_size=bs)
elif mini == False:
    # training data
    train_gen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, preprocessing_function=mu.RandRot)
    train_data = train_gen.flow_from_directory(os.path.join(root_dir, 'Train_set'),
                                               target_size=(512, 512), batch_size=bs)

    # validation data
    val_gen = keras.preprocessing.image.ImageDataGenerator()
    val_data = val_gen.flow_from_directory(os.path.join(root_dir, 'Val_set'),
                                           target_size=(512, 512), batch_size=bs)

######

# Build network

x = tf.placeholder(tf.float32, [None, 512, 512, 3])
y = tf.placeholder(tf.uint8, [None, 4])
training = tf.placeholder(tf.bool)

out = basic_network_tf.basic_CNN(x, training=training)

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
accuracy_placeholder = tf.placeholder(tf.float32, ())
loss_placeholder = tf.placeholder(tf.float32, ())
tf.summary.scalar("Accuracy", accuracy_placeholder)
tf.summary.scalar("Loss", loss_placeholder)
summary_op = tf.summary.merge_all()
mu.build_empty_dir('logs/train')
mu.build_empty_dir('logs/val')
writer_train = tf.summary.FileWriter('./logs/train', graph=sess.graph)
writer_val = tf.summary.FileWriter('./logs/val', graph=sess.graph)

# Save model variables
mu.build_empty_dir('saves')
saver = tf.train.Saver()

best_val_accuracy = 0.0
num_epochs = 100
print_every = 20
for e in range(num_epochs):

    # train
    mean_accuracy = 0.0
    mean_loss = 0.0
    print('Epoch {}'.format(e))
    print('Training')
    for batch, (ims, labels) in enumerate(train_data):
        if batch >= train_data.n / bs:
            break
        if batch % print_every == 0:
            print('Batch number {}'.format(batch))
        _, temp1, temp2 = sess.run([train_step, accuracy, loss], feed_dict={x: ims, y: labels, training: True})
        mean_accuracy += temp1
        mean_loss += temp2
    mean_accuracy /= batch
    mean_loss /= batch
    summary = sess.run(summary_op, feed_dict={accuracy_placeholder: mean_accuracy, loss_placeholder: mean_loss})
    writer_train.add_summary(summary, e + 1)

    # validation
    mean_accuracy = 0.0
    mean_loss = 0.0
    print('Validation')
    for batch, (ims, labels) in enumerate(val_data):
        if batch >= val_data.n / bs:
            break
        if batch % print_every == 0:
            print('Batch number {}'.format(batch))
        temp1, temp2 = sess.run([accuracy, loss], feed_dict={x: ims, y: labels, training: False})
        mean_accuracy += temp1
        mean_loss += temp2
    mean_accuracy /= batch
    mean_loss /= batch
    summary = sess.run(summary_op, feed_dict={accuracy_placeholder: mean_accuracy, loss_placeholder: mean_loss})
    writer_val.add_summary(summary, e + 1)
    if mean_accuracy > best_val_accuracy:
        best_val_accuracy = mean_accuracy
        save_path = saver.save(sess, './saves/best_model.ckpt')
        print("New best model saved in file: %s" % save_path)
