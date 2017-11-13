import collections
import json
import random
from os import path

import tensorflow as tf

from constants import TOTAL_DATA_SIZE, RANDOM_SPLIT_FILE, RAW_METADATA_FILE, \
    ASIN_INDEX_FILE, METADATA_FILE, INDEX_ASIN_FILE, DATASET_DIR


class DataSet(object):

    def __init__(self, filename):
        self._filename = filename

    def get_batch_tensor(self, batch_size, num_epochs=1):
        print('load dataset from ' + self._filename)
        filename_queue = tf.train.string_input_producer([self._filename], num_epochs=num_epochs)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'target': tf.FixedLenFeature([], tf.int64),
            }
        )
        # Convert the image data from string back to the numbers
        image = tf.reshape(tf.decode_raw(features['image'], tf.uint8), [224, 224, 3])
        target = tf.reshape(tf.cast(features['target'], tf.int32), [1])
        # Creates batches by randomly shuffling tensors
        images, targets = tf.train.shuffle_batch(
            [image, target],
            batch_size=batch_size,
            capacity=batch_size * 3,
            min_after_dequeue=batch_size,
            num_threads=4
        )
        print(images)
        print(targets)
        return images, targets


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
def json2tv(index_list, function, difficulty):
    print("making target vectors, function: " +function)
    with open(RAW_METADATA_FILE) as raw_metadata_file:
        raw_metadata = json.load(raw_metadata_file)
    with open(ASIN_INDEX_FILE) as asin_index_file:
        asin_index_map = json.load(asin_index_file)
    tv_list = []
    tv = []
    for index in index_list:
        data = raw_metadata[index]
        if function == "classify":
            tv = [0] * len(asin_index_map.keys())
            for asin in data['DATA'].keys():
                tv_index = asin_index_map.get(asin)
                if tv_index != None:
                    tv[tv_index] = 1
        elif function == "count":
            if difficulty == "moderate":
                tv = [0] * 12
                quantity = data['TOTAL']
                if quantity > 10:
                    tv[11] = 1
                else:
                    tv[quantity] = 1
            else:
                tv = [data['TOTAL']]
        tv_list.append(tv)
    return tv_list

"""
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
"""
