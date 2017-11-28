import json
import random
from os import path

import numpy as np
import tensorflow as tf
from PIL import Image

from constants import VALIDATION_RATIO, TEST_RATIO, IMAGE_DIR, DATASET_DIR, VALID_IMAGES_FILE

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


if __name__ == '__main__':
    # Randomly split the valid images list into train, validation, and test set.
    with open(VALID_IMAGES_FILE, 'r') as valid_images_file:
        valid_data_list = json.load(valid_images_file)
    valid_data_size = len(valid_data_list)
    print('valid data list size:{0}'.format(valid_data_size))
    num_validation = int(valid_data_size * VALIDATION_RATIO)
    num_test = int(valid_data_size * TEST_RATIO)
    num_train = valid_data_size - (num_validation + num_test)
    print('train:{0}, validation:{1}, test:{2}'.format(num_train, num_validation, num_test))
    random.shuffle(valid_data_list)
    random_split_json = {
        'train': valid_data_list[:num_train],
        'validation': valid_data_list[num_train:num_train + num_validation],
        'test': valid_data_list[num_train + num_validation:],
    }
    # print(random_split_json)
    make_tfrecord(random_split_json)
