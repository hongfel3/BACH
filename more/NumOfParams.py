import tensorflow as tf
from tensorflow.contrib import keras

from utils import basic_networks
from utils import misc_utils as mu

######

bs = 16  # batch size

lr = 1e-3  # learning rate

######

# train_gen = keras.preprocessing.image.ImageDataGenerator()
train_gen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, preprocessing_function=mu.RandRot)
train_data = train_gen.flow_from_directory('/home/peter/datasets/ICIAR2018_BACH_Challenge/Mini_set',
                                           target_size=(512, 512), batch_size=bs)

# validation data
val_gen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, preprocessing_function=mu.RandRot)
val_data = val_gen.flow_from_directory('/home/peter/datasets/ICIAR2018_BACH_Challenge/Mini_set',
                                       target_size=(512, 512), batch_size=bs)

######

# # training data
# train_gen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, preprocessing_function=mu.RandRot)
# train_data = train_gen.flow_from_directory('/home/peter/datasets/ICIAR2018_BACH_Challenge/Train_set',
#                                            target_size=(512, 512), batch_size=bs)
#
# # validation data
# val_gen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, preprocessing_function=mu.RandRot)
# val_data = val_gen.flow_from_directory('/home/peter/datasets/ICIAR2018_BACH_Challenge/Val_set',
#                                        target_size=(512, 512), batch_size=bs)

######

# Build network

x = tf.placeholder(tf.float32, [None, 512, 512, 3])
y = tf.placeholder(tf.uint8, [None, 4])
training = tf.placeholder(tf.bool, name='train')

out = basic_networks.basic_CNN(x, training=training)

total_parameters = 0
for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    print(shape)
    print(len(shape))
    variable_parameters = 1
    for dim in shape:
        print(dim)
        variable_parameters *= dim.value
    print(variable_parameters)
    total_parameters += variable_parameters
print(total_parameters)