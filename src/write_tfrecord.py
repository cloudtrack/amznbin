import json
import random
from os import path

import numpy as np
import tensorflow as tf
from PIL import Image

from constants import TOTAL_DATA_SIZE, VALIDATION_RATIO, TEST_RATIO, IMAGE_DIR, DATASET_DIR, VALID_IMAGES_FILE

IMAGE_CHUNK_SIZE = 1000


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# random_split.json 에 지정된 train, validation, test 데이터 매핑에 따라 tfrecord 를 청크별로 생성한다.
def make_tfrecord(random_split_json):
    for function_type in ['train', 'validation', 'test']:
        index_list = random_split_json.get(function_type)
        index_chunks = [index_list[x:x + IMAGE_CHUNK_SIZE] for x in range(0, len(index_list), IMAGE_CHUNK_SIZE)]
        for i in range(len(index_chunks)):
            filename = path.join(DATASET_DIR, '{0}_{1}.tfrecords'.format(function_type, i))
            print('Start Writing ' + filename)
            current_index_chunk = index_chunks[i]
            writer = tf.python_io.TFRecordWriter(filename)
            for index in current_index_chunk:
                image = np.array(Image.open('%s%05d.jpg' % (IMAGE_DIR, index)))
                # Create a feature
                features = tf.train.Features(feature={
                    'image': _bytes_feature(tf.compat.as_bytes(image.tostring())),
                    'index': _int64_feature(index)
                })
                example = tf.train.Example(features=features)
                # Serialize to string and write on the file
                writer.write(example.SerializeToString())
            writer.close()


# Randomly split the whole list into train, validation, and test set.
def make_random_split(train_size, validation_size, test_size):
    print('make random_split for train:{0}, validation:{1}, test:{2}'.format(train_size, validation_size, test_size))
    # index_list = list(range(1, TOTAL_DATA_SIZE + 1))
    with open(VALID_IMAGES_FILE, 'r') as valid_images_file:
        index_list = json.load(valid_images_file)
    random.shuffle(index_list)
    result = {
        'train': index_list[:train_size],
        'validation': index_list[train_size:train_size + validation_size],
        'test': index_list[train_size + validation_size:],
    }
    return result


if __name__ == '__main__':
    num_validation = int(TOTAL_DATA_SIZE * VALIDATION_RATIO)
    num_test = int(TOTAL_DATA_SIZE * TEST_RATIO)
    num_training = TOTAL_DATA_SIZE - (num_validation + num_test)
    print('train:{0}, validation:{1}, test:{2}'.format(num_training, num_validation, num_test))
    random_split_json = make_random_split(num_training, num_validation, num_test)
    make_tfrecord(random_split_json)
