import tensorflow as tf

height=512
width=512


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example, features={'image_raw': tf.FixedLenFeature([], tf.string),
                                                                     'label': tf.FixedLenFeature([], tf.int64)
                                                                     })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)

    image = tf.reshape(image, [height, width, 3])

    # Transformations can be put here.

    image.set_shape((height, width, 3))

    images, labels = tf.train.shuffle_batch([image, label], batch_size=16, capacity=30, num_threads=2,
                                            min_after_dequeue=10)

    return images, labels
