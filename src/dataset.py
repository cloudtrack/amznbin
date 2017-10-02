import collections

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes


class DataSet(object):
    def __init__(self,
                 images,
                 labels,
                 dtype=dtypes.float64,
                 reshape=True):
        if reshape:
            assert images.shape[3] == 1
            images = images.reshape(images.shape[0],
                                    images.shape[1] * images.shape[2])

        self._images = images
        self._num_examples = images.shape[0]
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._images[start:end], self._labels[start:end]


DATA_SIZE = 535234
VALIDATION_SIZE = 50000
TEST_SIZE = 10000


def load_dataset(dataset_dir, dtype=dtypes.float64, reshape=True):
    file_reader = tf.WholeFileReader()
    image_files_queue = tf.train.string_input_producer(tf.train.match_filenames_once(dataset_dir + '/bin-images/*.jpg'))
    filename, image_file = file_reader.read(image_files_queue)
    image = tf.image.decode_jpeg(image_file, channels=3)

    num_training = DATA_SIZE - (VALIDATION_SIZE + TEST_SIZE)
    num_validation = VALIDATION_SIZE
    num_test = TEST_SIZE

    print('Load training:{0}, validation:{1}, test:{2}'.format(num_training, num_validation, num_test))

    all_images = image
    all_images = all_images.reshape(all_images.shape[0],
                                    all_images.shape[1], all_images.shape[2], 1)

    all_labels = np.asarray(range(0, DATA_SIZE))

    mask = range(num_training)
    train_images = all_images[mask]
    train_labels = all_labels[mask]

    mask = range(num_training, num_training + num_validation)
    validation_images = all_images[mask]
    validation_labels = all_labels[mask]

    mask = range(num_training + num_validation, num_training + num_validation + num_test)
    test_images = all_images[mask]
    test_labels = all_labels[mask]

    train = DataSet(train_images, train_labels, dtype=dtype, reshape=reshape)
    validation = DataSet(validation_images, validation_labels, dtype=dtype, reshape=reshape)
    test = DataSet(test_images, test_labels, dtype=dtype, reshape=reshape)

    ds = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
    return ds(train=train, validation=validation, test=test)
