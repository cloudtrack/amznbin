import collections
import json
import random
from os import path
from time import time

import tensorflow as tf

from constants import TOTAL_DATA_SIZE, RANDOM_SPLIT_FILE, IMAGE_DIR, RAW_METADATA_FILE, \
    ASIN_INDEX_FILE, METADATA_FILE, INDEX_ASIN_FILE, DATASET_DIR


def _parse_function(example_proto):
    features = {
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    return parsed_features["image"], parsed_features["label"]

class DataSet(object):

    def __init__(self, filename):
        self._filename = filename
        self._num_examples = 1000
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self):
        return self.next_batch(self._num_examples)

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        assert batch_size <= self._num_examples
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            print("epoch completed!")
            # Finished epoch
            self._epochs_completed += 1
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch
        print("load next batch(size {0}) from {1} to {2}".format(batch_size, start, end))
        t0 = time()
        with tf.Session() as sess:
            filename_queue = tf.train.string_input_producer([self._filename])
            reader = tf.TFRecordReader()
            _, serialized_example = reader.read(filename_queue)
            features = tf.parse_single_example(
                serialized_example,
                features={
                    'image': tf.FixedLenFeature([], tf.string),
                    'label': tf.FixedLenFeature([], tf.int64),
                }
            )
            # Convert the image data from string back to the numbers
            image = tf.reshape(tf.decode_raw(features['image'], tf.uint8), [224, 224, 3])
            label = tf.reshape(tf.cast(features['label'], tf.int32), [1])
            min_queue_examples_train = 50
            # Creates batches by randomly shuffling tensors
            images, labels = tf.train.shuffle_batch(
                [image, label], batch_size=batch_size, num_threads=8,
                capacity=min_queue_examples_train + 3 * batch_size,
                min_after_dequeue=min_queue_examples_train
            )
            sess.run(tf.global_variables_initializer())

            # Create a coordinator and run all QueueRunner objects
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            images_value, labels_value = sess.run([images, labels])
            # Stop the threads
            coord.request_stop()

            # Wait for threads to stop
            coord.join(threads)
            sess.close()
        print('get_images finished after ' + str(round(time() - t0, 2)) + 's')
        return images_value, labels_value


def load_dataset(function):
    train = DataSet(path.join(DATASET_DIR, '{0}_{1}.tfrecords'.format(function, 'train')))
    validation = DataSet(path.join(DATASET_DIR, '{0}_{1}.tfrecords'.format(function, 'validation')))
    test = DataSet(path.join(DATASET_DIR, '{0}_{1}.tfrecords'.format(function, 'test')))
    ds = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
    return ds(train=train, validation=validation, test=test)


# Randomly split the whole list into train, validation, and test set.
def make_random_split(train_size, validation_size, test_size):
    print('make new random_split.json for train:{0}, validation:{1}, test:{2}'.format(train_size, validation_size, test_size))
    random_list = list(range(1, TOTAL_DATA_SIZE + 1))
    random.shuffle(random_list)
    result = {
        'train': random_list[:train_size],
        'validation': random_list[train_size:train_size + validation_size],
        'test': random_list[train_size + validation_size:],
    }
    with open(RANDOM_SPLIT_FILE, 'w') as random_split_file:
        json.dump(result, random_split_file)


########################
# Target vector utils
########################
def json2tv(index_list, function):
    print("making target vectors, function: " +function)
    with open(RAW_METADATA_FILE) as raw_metadata_file:
        raw_metadata = json.load(raw_metadata_file)
    with open(ASIN_INDEX_FILE) as asin_index_file:
        asin_index_map = json.load(asin_index_file)
    tv_list = []
    tv = []
    for index in index_list:
        if function == "classify":
            tv = [0] * len(asin_index_map.keys())
            data = raw_metadata[index]
            for asin in data['DATA'].keys():
                tv_index = asin_index_map.get(asin)
                tv[tv_index] = 1
        elif function == "count":
            data = raw_metadata[index]
            tv = [data['TOTAL']]
        else:
            print("Invalid function name")
        tv_list.append(tv)
    return tv_list


def tv2res(tv):
    print("opening " + METADATA_FILE)
    with open(METADATA_FILE) as metadata_file:
        metadata = json.load(metadata_file)
    print("opening " + INDEX_ASIN_FILE)
    with open(INDEX_ASIN_FILE) as index_asin_file:
        index_asin_map = json.load(index_asin_file)
    res = {}
    for i in range(len(tv)):
        if tv[i] != 0:
            asin = index_asin_map[str(i)]
            asin_meta = {
                'name': metadata[asin]['name'],
                'quantity': tv[i],
            }
            res[asin] = asin_meta
    return res
