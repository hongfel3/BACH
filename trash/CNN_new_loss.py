from keras.layers import Input, Dense, Conv2D, Activation, MaxPool2D, Flatten, BatchNormalization, Dropout, Layer
from keras import optimizers
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import backend as K
from keras import metrics

import os
from utils import misc_utils as mu

###

dropout_rate = 0.5
initial_learning_rate = 0.001

epochs = 20
batch_size = 64

###

# >> change me << #
root_dir = '/media/peter/HDD 1/ICIAR2018_BACH_Challenge'

###

mini = True

if mini == True:
    # training data
    train_data = ImageDataGenerator().flow_from_directory(os.path.join(root_dir, 'Mini_set'),
                                                          target_size=(256, 256), batch_size=batch_size)
    print(train_data.class_indices)

    # validation data
    val_data = ImageDataGenerator().flow_from_directory(os.path.join(root_dir, 'Mini_set'),
                                                        target_size=(256, 256), batch_size=batch_size)
    print(val_data.class_indices)

elif mini == False:
    # training data
    train_data = ImageDataGenerator(horizontal_flip=True, preprocessing_function=mu.RandRot).flow_from_directory(
        os.path.join(root_dir, 'Train_set'),
        target_size=(256, 256), batch_size=batch_size)
    print(train_data.class_indices)

    # validation data
    val_data = ImageDataGenerator().flow_from_directory(os.path.join(root_dir, 'Val_set'),
                                                        target_size=(256, 256), batch_size=batch_size)
    print(val_data.class_indices)


###

###

class CustomLossLayer(Layer):

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super().__init__(**kwargs)

    def my_loss(self, probabilities, labels):
        loss = metrics.categorical_crossentropy(probabilities, labels)
        return K.mean(loss)

    def call(self, inputs):
        probabilities = inputs[0]
        labels = inputs[1]
        loss = self.my_loss(probabilities, labels)
        self.add_loss(loss, inputs=inputs)
        # We don't use this output.
        return loss


###

def conv5x5_relu(x, num_filters, pad='same'):
    x = Conv2D(filters=num_filters, kernel_size=(5, 5), strides=(1, 1), padding=pad)(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)
    return x


def max_pool2x2(x):
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    return x


def dense(x, num_out):
    x = Dense(units=num_out)(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)
    x = Dropout(rate=dropout_rate)(x)
    return x


def basic_network(x):
    x = BatchNormalization(momentum=0.9)(x)
    x = conv5x5_relu(x, 32)
    x = max_pool2x2(x)
    x = conv5x5_relu(x, 32)
    x = max_pool2x2(x)
    x = conv5x5_relu(x, 64)
    x = max_pool2x2(x)
    x = Flatten()(x)
    x = dense(x, 64)
    return Dense(units=4, activation='softmax', name='probabilities')(x)


###

data = Input(shape=(256, 256, 3))
labels = Input(shape=(4,))
probabilities = basic_network(data)
loss1 = CustomLossLayer()([probabilities, labels])

trainer = Model(inputs=[data, labels], outputs=loss1)
predictor = Model(inputs=data, outputs=probabilities)
print(trainer.summary())

optim = optimizers.Adam(lr=initial_learning_rate)
trainer.compile(optimizer=optim,
                loss=None)

###

for e in range(epochs):
    print('Epoch {} / {}'.format(e + 1, epochs))
    for i in range(len(train_data)):
        X, y = train_data.next()
        trainer.train_on_batch(x=[X, y], y=None)
