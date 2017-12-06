from keras.layers import Input, Dense, Conv2D, Activation, MaxPool2D, Flatten
from keras.models import Model


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


inputs = Input(shape=(512, 512, 3))
predictions = basic_network(inputs)

model = Model(inputs=inputs, outputs=predictions)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# model.fit(data, labels)  # starts training
