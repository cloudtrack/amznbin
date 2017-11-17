import json
from os import path

import numpy as np
import tensorflow as tf
from PIL import Image

from constants import TOTAL_DATA_SIZE, VALIDATION_SIZE, TEST_SIZE, RANDOM_SPLIT_FILE, IMAGE_DIR, DATASET_DIR
from dataset import json2tv, make_random_split

IMAGE_CHUNK_SIZE = 1000


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_feature_list(values):
  return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def write_tfrecord(training_type, difficulty=None):
    for data_type in ['train', 'validation', 'test']:
        index_list = random_split_json.get(data_type)
        index_chunks = [index_list[x:x + IMAGE_CHUNK_SIZE] for x in range(0, len(index_list), IMAGE_CHUNK_SIZE)]
        for i in range(len(index_chunks)):
            if difficulty:
                filename = path.join(DATASET_DIR,
                                     '{0}_{1}_{2}_{3}.tfrecords'.format(training_type, difficulty, data_type, i))
            else:
                filename = path.join(DATASET_DIR, '{0}_{1}_{2}.tfrecords'.format(training_type, data_type, i))
            print('Start Writing ' + filename)
            current_chunk = index_chunks[i]
            writer = tf.python_io.TFRecordWriter(filename)
            targets = json2tv(current_chunk, training_type, difficulty)
            for j in range(len(current_chunk)):
                img = np.array(Image.open('%s%05d.jpg' % (IMAGE_DIR, current_chunk[j])))
                target = targets[j]
                if i == 0 and j == 0:
                    print('show the first feature')
                    print('image: {0}'.format(img.tostring()))
                    print('target_size: {0}'.format(len(target)))
                    print('target: {0}'.format(target))
                # Create a feature
                features = tf.train.Features(feature={
                    'image': _bytes_feature(tf.compat.as_bytes(img.tostring())),
                    'target_size': _int64_feature(len(target)),
                })
                feature_lists = tf.train.FeatureLists(feature_list={
                    'target': _int64_feature_list(target)
                })
                # Create an example protocol buffer
                sequence_example = tf.train.SequenceExample(
                    context=features, feature_lists=feature_lists
                )
                # example = tf.train.Example(features=tf.train.Features(feature=feature))
                # Serialize to string and write on the file
                writer.write(sequence_example.SerializeToString())
            writer.close()


if __name__ == '__main__':
    num_training = TOTAL_DATA_SIZE - (VALIDATION_SIZE + TEST_SIZE)
    num_validation = VALIDATION_SIZE
    num_test = TEST_SIZE
    print('train:{0}, validation:{1}, test:{2}'.format(num_training, num_validation, num_test))
    if not tf.gfile.Exists(RANDOM_SPLIT_FILE):
        make_random_split(num_training, num_validation, num_test)
    with open(RANDOM_SPLIT_FILE, 'r') as random_split_file:
        random_split_json = json.load(random_split_file)
    # Count
    for difficulty in ['moderate', 'hard']:
        write_tfrecord('count', difficulty)
    # Classify
    write_tfrecord('classify')
