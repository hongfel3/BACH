from keras.layers import Input, Dense, Conv2D
from keras.models import Model


def conv3x3_relu(x, num_filters, pad='valid'):
    x = Conv2D(filters=num_filters, kernel_size=(3, 3), strides=(1, 1), padding=pad, activation='relu')(x)
    return x


inputs = Input(shape=(512, 512, 3))

x = conv3x3_relu(inputs, 16)

print(x.shape)


# predictions = Dense(10, activation='softmax')(x)

# model = Model(inputs=inputs, outputs=predictions)
# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# model.fit(data, labels)  # starts training
