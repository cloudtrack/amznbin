import collections
import json

import tensorflow as tf

from constants import RAW_METADATA_FILE, ASIN_INDEX_FILE, DATASET_DIR, MAXIMUM_COUNT


def prefetch_input_data(
        reader,
        filename_queue,
        is_training,
        batch_size,
        values_per_shard,
        input_queue_capacity_factor=16,
        num_reader_threads=1):
    """
    Prefetches string values from disk into an input queue.
    In training the capacity of the queue is important because a larger queue
    means better mixing of training examples between shards. The minimum number of
    values kept in the queue is values_per_shard * input_queue_capacity_factor,
    where input_queue_memory factor should be chosen to trade-off better mixing
    with memory usage.

    Args:
      reader: Instance of tf.ReaderBase.
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
        min_queue_examples = values_per_shard * input_queue_capacity_factor
        capacity = min_queue_examples + 100 * batch_size
        values_queue = tf.RandomShuffleQueue(
            capacity=capacity,
            min_after_dequeue=min_queue_examples,
            dtypes=[tf.string],
        )
    else:
        print("is_training == False : FIFOQueue")
        capacity = values_per_shard + 3 * batch_size
        values_queue = tf.FIFOQueue(capacity=capacity, dtypes=[tf.string])
    enqueue_ops = []
    for _ in range(num_reader_threads):
        _, value = reader.read(filename_queue)
        enqueue_ops.append(values_queue.enqueue([value]))
    tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(values_queue, enqueue_ops))
    return values_queue


class DataSet(object):
    def __init__(self, data_type):
        self._data_type = data_type
        self.raw_metadata = None
        self.asin_index_map = None

    def get_batch_tensor(self, batch_size, num_epochs=1):
        print('load dataset for {0}'.format(self._data_type))
        filename_format = '{0}{1}_*.tfrecords'.format(DATASET_DIR, self._data_type)
        print('use filename format {0}'.format(filename_format))
        reader = tf.TFRecordReader()
        is_training = (self._data_type == 'train')
        filename_queue = tf.train.string_input_producer(
            tf.train.match_filenames_once(filename_format),
            shuffle=is_training,
            num_epochs=num_epochs
        )
        input_queue = prefetch_input_data(
            reader,
            filename_queue=filename_queue,
            is_training=is_training,
            batch_size=batch_size,
            values_per_shard=2000,  # mixing between shards in training.
            input_queue_capacity_factor=2,  # minimum number of shards to keep in the input queue.
            num_reader_threads=4  # number of threads for prefetching SequenceExample protos.
        )
        serialized = input_queue.dequeue()
        features = tf.parse_single_example(
            serialized,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'index': tf.FixedLenFeature([], tf.int64),
            },
            name='features'
        )
        image = tf.reshape(tf.decode_raw(features['image'], tf.uint8), [224, 224, 3])
        index = features['index']

        min_after_dequeue = batch_size * 3
        capacity = min_after_dequeue + 10 * batch_size
        images, indices = tf.train.shuffle_batch(
            [image, index],
            batch_size=batch_size,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue,
            num_threads=4
        )
        print(images)
        print(indices)
        return images, indices

    def get_labels_from_indices(self, index_list, function, difficulty):
        if not self.raw_metadata:
            with open(RAW_METADATA_FILE) as raw_metadata_file:
                self.raw_metadata = json.load(raw_metadata_file)
        tv_list = []
        tv = []
        for index in index_list:
            data = self.raw_metadata[index % 1000000]
            if function == "classify":
                if not self.asin_index_map:
                    with open(ASIN_INDEX_FILE) as asin_index_file:
                        self.asin_index_map = json.load(asin_index_file)
                tv = [0] * (len(self.asin_index_map.keys()) + 1)
                if data['TOTAL'] == 0:
                    tv[0] = 1
                else:
                    for asin in data['DATA'].keys():
                        tv_index = self.asin_index_map.get(asin)
                        if tv_index:
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


def load_dataset():
    train = DataSet('train')
    validation = DataSet('validation')
    test = DataSet('test')
    ds = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
    return ds(train=train, validation=validation, test=test)


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
