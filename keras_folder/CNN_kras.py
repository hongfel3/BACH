from keras.layers import Input, Dense, Conv2D, Activation, MaxPool2D, Flatten, BatchNormalization
from keras import optimizers
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint

import os
from utils import misc_utils as mu

###

learn_rate = 1e-3
batch_size = 64

###

# root_dir = '/home/peter/datasets/ICIAR2018_BACH_Challenge'
root_dir = '/media/peter/HDD 1/ICIAR2018_BACH_Challenge'

mini = True

if mini == True:
    # training data
    train_gen = ImageDataGenerator()
    train_data = train_gen.flow_from_directory(os.path.join(root_dir, 'Mini_set'),
                                               target_size=(512, 512), batch_size=batch_size)

    # validation data
    val_gen = ImageDataGenerator()
    val_data = val_gen.flow_from_directory(os.path.join(root_dir, 'Mini_set'),
                                           target_size=(512, 512), batch_size=batch_size)
elif mini == False:
    # training data
    train_gen = ImageDataGenerator(horizontal_flip=True, preprocessing_function=mu.RandRot)
    train_data = train_gen.flow_from_directory(os.path.join(root_dir, 'Train_set'),
                                               target_size=(512, 512), batch_size=batch_size)

    # validation data
    val_gen = ImageDataGenerator()
    val_data = val_gen.flow_from_directory(os.path.join(root_dir, 'Val_set'),
                                           target_size=(512, 512), batch_size=batch_size)


###

def conv3x3_relu(x, num_filters, pad='valid'):
    x = Conv2D(filters=num_filters, kernel_size=(3, 3), strides=(1, 1), padding=pad)(x)
    x = Activation('relu')(x)
    return x


def max_pool3x3(x):
    x = MaxPool2D(pool_size=(3, 3), strides=(3, 3), padding='valid')(x)
    return x


def max_pool2x2(x):
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    return x


def dense_relu(x, num_out):
    x = Dense(units=num_out)(x)
    x = Activation('relu')(x)
    return x


def basic_network(x):
    x = BatchNormalization(momentum=0.9)(x)
    x = conv3x3_relu(x, 16)
    x = max_pool3x3(x)
    x = conv3x3_relu(x, 32)
    x = max_pool2x2(x)
    x = conv3x3_relu(x, 64, 'same')
    x = max_pool2x2(x)
    x = conv3x3_relu(x, 64, 'same')
    x = max_pool3x3(x)
    x = conv3x3_relu(x, 32)
    x = max_pool3x3(x)
    x = Flatten()(x)
    x = dense_relu(x, 256)
    x = dense_relu(x, 128)
    return Dense(units=4, activation='softmax')(x)


###

inputs = Input(shape=(512, 512, 3))
predictions = basic_network(inputs)

model = Model(inputs=inputs, outputs=predictions)

optim = optimizers.Adam(lr=learn_rate)
model.compile(optimizer=optim,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

mu.build_empty_dir('logs')
call1 = TensorBoard(log_dir='logs')

call2=ModelCheckpoint('best_model.h5',monitor='val_acc',verbose=True,save_best_only=True)

###


total_epochs = 100

model.fit_generator(train_data, epochs=total_epochs, validation_data=val_data, callbacks=[call])
