import collections
import json
import random

import tensorflow as tf

from constants import TOTAL_DATA_SIZE, RANDOM_SPLIT_FILE, RAW_METADATA_FILE, \
    ASIN_INDEX_FILE, DATASET_DIR, MAXIMUM_COUNT


def prefetch_input_data(
        reader,
        filename_format,
        num_epochs,
        is_training,
        batch_size,
        values_per_shard,
        input_queue_capacity_factor=16,
        num_reader_threads=1,
        shard_queue_name="filename_queue",
        value_queue_name="input_queue"):
    """
    Prefetches string values from disk into an input queue.
    In training the capacity of the queue is important because a larger queue
    means better mixing of training examples between shards. The minimum number of
    values kept in the queue is values_per_shard * input_queue_capacity_factor,
    where input_queue_memory factor should be chosen to trade-off better mixing
    with memory usage.

    Args:
      reader: Instance of tf.ReaderBase.
      is_training: Boolean; whether prefetching for training or eval.
      batch_size: Model batch size used to determine queue capacity.
      values_per_shard: Approximate number of values per shard.
      input_queue_capacity_factor: Minimum number of values to keep in the queue
      in multiples of values_per_shard. See comments above.
      num_reader_threads: Number of reader threads to fill the queue.
    Returns:
      A Queue containing prefetched string values.
    """
    if is_training:
        print("is_training == True : RandomShuffleQueue")
        filename_queue = tf.train.string_input_producer(
            tf.train.match_filenames_once(filename_format),
            shuffle=True, capacity=16, name=shard_queue_name, num_epochs=num_epochs
        )
        min_queue_examples = values_per_shard * input_queue_capacity_factor
        capacity = min_queue_examples + 100 * batch_size
        values_queue = tf.RandomShuffleQueue(
            capacity=capacity,
            min_after_dequeue=min_queue_examples,
            dtypes=[tf.string],
            name="random_" + value_queue_name
        )
    else:
        print("is_training == False : FIFOQueue")
        filename_queue = tf.train.string_input_producer(
            tf.train.match_filenames_once(filename_format),
            shuffle=False, capacity=1, name=shard_queue_name, num_epochs=num_epochs
        )
        capacity = values_per_shard + 3 * batch_size
        values_queue = tf.FIFOQueue(
            capacity=capacity, dtypes=[tf.string], name="fifo_" + value_queue_name
        )
    enqueue_ops = []
    for _ in range(num_reader_threads):
        _, value = reader.read(filename_queue)
        enqueue_ops.append(values_queue.enqueue([value]))
    tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(values_queue, enqueue_ops))
    return values_queue


class DataSet(object):
    def __init__(self, function, type, difficulty, output_len):
        self._function = function
        self._type = type
        self._difficulty = difficulty
        self._output_len = output_len

    def get_batch_tensor(self, batch_size, num_epochs=1):
        print('load dataset for {0} {1} {2}'.format(self._function, self._difficulty, self._type))
        if self._difficulty:
            filename_format = '{0}{1}_{2}_{3}_*.tfrecords'.format(DATASET_DIR, self._function, self._difficulty,
                                                                  self._type)
        else:
            filename_format = '{0}{1}_{2}_*.tfrecords'.format(DATASET_DIR, self._function, self._type)
        print('use filename format {0}'.format(filename_format))
        filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(filename_format),
                                                        num_epochs=num_epochs)
        reader = tf.TFRecordReader()
        # _, serialized_example = reader.read(filename_queue)
        input_queue = prefetch_input_data(
            reader,
            filename_format=filename_format,
            num_epochs=num_epochs,
            is_training=True,  # if training, shuffle and random choice
            batch_size=batch_size,
            values_per_shard=2300,  # mixing between shards in training.
            input_queue_capacity_factor=2,  # minimum number of shards to keep in the input queue.
            num_reader_threads=1  # number of threads for prefetching SequenceExample protos.
        )
        serialized_example = input_queue.dequeue()
        context, sequence = tf.parse_single_sequence_example(
            serialized_example,
            context_features={
                'image': tf.FixedLenFeature([], tf.string),
            },
            sequence_features={
                'target': tf.FixedLenSequenceFeature([], tf.int64),
            }
        )
        # Convert the image data from string back to the numbers
        image = tf.reshape(tf.decode_raw(context['image'], tf.uint8), [224, 224, 3])
        target_size = self._output_len
        target = tf.reshape(sequence['target'], [target_size])
        # Creates batches by randomly shuffling tensors
        images, targets = tf.train.batch(
            [image, target],
            batch_size=batch_size,
            capacity=batch_size * 3,
            num_threads=4
        )
        print(images)
        print(targets)
        return images, targets


def load_dataset(function, difficulty, output_len):
    train = DataSet(function, 'train', difficulty, output_len)
    validation = DataSet(function, 'validation', difficulty, output_len)
    test = DataSet(function, 'test', difficulty, output_len)
    ds = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
    return ds(train=train, validation=validation, test=test)


# Randomly split the whole list into train, validation, and test set.
def make_random_split(train_size, validation_size, test_size):
    print('make new random_split.json for train:{0}, validation:{1}, test:{2}'.format(train_size, validation_size,
                                                                                      test_size))
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
    print("making target vectors, function: " + function)
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
                tv = [0] * (MAXIMUM_COUNT + 2)
                quantity = data['TOTAL']
                if quantity > MAXIMUM_COUNT:
                    tv[MAXIMUM_COUNT + 1] = 1
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
