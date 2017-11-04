import json

import numpy as np
import tensorflow as tf
from PIL import Image
from os import path

from constants import TOTAL_DATA_SIZE, VALIDATION_SIZE, TEST_SIZE, RANDOM_SPLIT_FILE, IMAGE_DIR, DATASET_DIR
from dataset import json2tv, make_random_split


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

if __name__ == '__main__':
    num_training = TOTAL_DATA_SIZE - (VALIDATION_SIZE + TEST_SIZE)
    num_validation = VALIDATION_SIZE
    num_test = TEST_SIZE
    print('train:{0}, validation:{1}, test:{2}'.format(num_training, num_validation, num_test))
    if not tf.gfile.Exists(RANDOM_SPLIT_FILE):
        make_random_split(num_training, num_validation, num_test)
    with open(RANDOM_SPLIT_FILE, 'r') as random_split_file:
        random_split_json = json.load(random_split_file)

    for training_type in ['count', 'classify']:
        for data_type in ['train', 'validation', 'test']:
            filename = path.join(DATASET_DIR, '{0}_{1}.tfrecords'.format(training_type, data_type))
            print('Start Writing ' + filename)
            writer = tf.python_io.TFRecordWriter(filename)
            file_indices = random_split_json.get(data_type)
            targets = json2tv(file_indices, training_type)

            for i in range(len(file_indices)):
                if not i % 1000:
                    print('processing: {}/{}'.format(i, len(file_indices)))
                # Load the image
                image_file = '%s%05d.jpg' % (IMAGE_DIR, file_indices[i])
                img = np.array(Image.open(image_file))
                target = targets[i]
                # Create a feature
                feature = {
                    'image': _bytes_feature(tf.compat.as_bytes(img.tostring())),
                    'target': _int64_feature(target)
                }
                # Create an example protocol buffer
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                # Serialize to string and write on the file
                writer.write(example.SerializeToString())

            writer.close()
