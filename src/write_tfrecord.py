import json
import random
from os import path

import numpy as np
import tensorflow as tf
from PIL import Image

from constants import VALIDATION_RATIO, TEST_RATIO, IMAGE_DIR, DATASET_DIR, VALID_IMAGES_FILE

IMAGE_CHUNK_SIZE = 50


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# random_split.json 에 지정된 train, validation, test 데이터 매핑에 따라 tfrecord 를 청크별로 생성한다.
def make_tfrecord(random_split_json):
    for function_type in ['train', 'validation', 'test']:
        index_list = random_split_json.get(function_type)
        if function_type == 'train' :
            new_valid_images = []
            print('Input data augmentation mode (0, 1, 2, 3)')
            cmd = int(input())

            for i in index_list:
                if i < 1000000:
                    image = Image.open('%s%05d.jpg' % (IMAGE_DIR, i))
                    if cmd >= 1:
                        new_i = i + 1000000
                        t_image = image.transpose(Image.FLIP_LEFT_RIGHT)
                        t_image.save('%s%05d.jpg' % (IMAGE_DIR, new_i))
                        new_valid_images.append(new_i)
                    if cmd >= 2:
                        new_i = i + 2000000
                        t_image = image.transpose(Image.FLIP_TOP_BOTTOM)
                        t_image.save('%s%05d.jpg' % (IMAGE_DIR, new_i))
                        new_valid_images.append(new_i)

                        new_i = i + 3000000
                        t_image = image.transpose(Image.ROTATE_180)
                        t_image.save('%s%05d.jpg' % (IMAGE_DIR, new_i))
                        new_valid_images.append(new_i)
                    if cmd >= 3:
                        new_i = i + 4000000
                        t_image = image.transpose(Image.ROTATE_90)
                        t_image.save('%s%05d.jpg' % (IMAGE_DIR, new_i))
                        new_valid_images.append(new_i)

                        new_i = i + 5000000
                        t_image = image.transpose(Image.ROTATE_270)
                        t_image.save('%s%05d.jpg' % (IMAGE_DIR, new_i))
                        new_valid_images.append(new_i)

                        new_i = i + 6000000
                        t_image = image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90)
                        t_image.save('%s%05d.jpg' % (IMAGE_DIR, new_i))
                        new_valid_images.append(new_i)

                        new_i = i + 7000000
                        t_image = image.transpose(Image.FLIP_LEFT_RIGHT).transpose(IMAGE.ROTATE_270)
                        t_image.save('%s%05d.jpg' % (IMAGE_DIR, new_i))
                        new_valid_images.append(new_i)
            for i in new_valid_images:
                index_list.append(i)
            random.shuffle(index_list)

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
