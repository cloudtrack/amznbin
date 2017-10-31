import collections
import json
import random

import numpy as np
import tensorflow as tf

from time import time
from constants import TOTAL_DATA_SIZE, VALIDATION_SIZE, TEST_SIZE, RANDOM_SPLIT_FILE, IMAGE_DIR, RAW_METADATA_FILE, \
    ASIN_INDEX_FILE, METADATA_FILE, INDEX_ASIN_FILE


class DataSet(object):
    def __init__(self, input_list, function):
        self._input_list = input_list
        self._function = function
        self._num_examples = len(input_list)
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._get_images(0, self._num_examples)

    @property
    def labels(self):
        return self._get_labels(0, self._num_examples)

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        assert batch_size <= self._num_examples
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            print("epoch completed!")
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            random.shuffle(self._input_list)
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch
        print("load next batch(size {0}) from {1} to {2}".format(batch_size, start, end))
        return self._get_images(start, end), self._get_labels(start, end)

    def _get_images(self, start, end):
        t0 = time()
        files = ['%s%05d.jpg' % (IMAGE_DIR, self._input_list[i]) for i in range(start, end)]
        filename_queue = tf.train.string_input_producer(files)
        image_name, image_file = tf.WholeFileReader().read(filename_queue)
        decoded_image = tf.image.decode_jpeg(image_file, channels=3)
        decoded_image.set_shape((224, 224, 3))
        image_batch = tf.train.batch(
            [decoded_image],
            batch_size=len(files),
        )
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            image = sess.run(image_batch)
            coord.request_stop()
            coord.join(threads)
            sess.close()
        print('get_images finished after ' + str(round(time() - t0, 2)) + 's')
        return image

    def _get_labels(self, start, end):
        tv_list = json2tv(self._input_list[start:end], self._function)
        return np.array(tv_list)


def load_dataset(function):
    num_training = TOTAL_DATA_SIZE - (VALIDATION_SIZE + TEST_SIZE)
    num_validation = VALIDATION_SIZE
    num_test = TEST_SIZE

    print('train:{0}, validation:{1}, test:{2}'.format(num_training, num_validation, num_test))

    if not tf.gfile.Exists(RANDOM_SPLIT_FILE):
        make_random_split(num_training, num_validation, num_test)
    with open(RANDOM_SPLIT_FILE, 'r') as random_split_file:
        random_split_json = json.load(random_split_file)

    train = DataSet(random_split_json.get('train'), function)
    validation = DataSet(random_split_json.get('validation'), function)
    test = DataSet(random_split_json.get('test'), function)

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
    print("making target vectors, function: "+function)
    print("opening " + RAW_METADATA_FILE)
    with open(RAW_METADATA_FILE) as raw_metadata_file:
        raw_metadata = json.load(raw_metadata_file)
    print("opening " + ASIN_INDEX_FILE)
    with open(ASIN_INDEX_FILE) as asin_index_file:
        asin_index_map = json.load(asin_index_file)
    tv_list = []
    for index in index_list:
        if(function == "classify"):
            tv = [0] * len(asin_index_map.keys())
            data = raw_metadata[index]
            for asin in data['DATA'].keys():
                tv_index = asin_index_map.get(asin)
                tv[tv_index] = 1
        elif (function == "count"):
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
