import argparse
import json
import time

import tensorflow as tf
from numpy.distutils.fcompiler import str2bool

from constants import VALIDATION_SIZE, TEST_SIZE
from dataset import load_dataset
from models import ALEXNET, VGGNET, LENET


def train(model, sess, saver, train_data, valid_data, batch_size, max_iters, use_early_stop, early_stop_max_iter, function):
    """
    Trainer 
    """
    t0 = time.time()
    # Optimize
    prev_valid_rmse = float("Inf")
    early_stop_iters = 0
    train_image_tensor, train_target_tensor = train_data.get_batch_tensor(batch_size)
    valid_image_tensor, valid_target_tensor = valid_data.get_batch_tensor(batch_size=VALIDATION_SIZE)
    for i in range(max_iters):
        print('==== New epoch started ====')
        # Training
        with tf.Session() as _sess:
            _sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=_sess, coord=coord)
            try:
                while not coord.should_stop():
                    t2 = time.time()
                    print('train - get next batch')
                    images, labels = _sess.run([train_image_tensor, train_target_tensor])
                    model.train_iteration(images, labels)
                    train_error = model.eval_loss(images, labels)
                    train_rmse, train_acc, train_pred = model.eval_metric(images, labels)
                    print(model.model_filename)
                    print("train accuracy: %.4f, train rmse: %.4f,  train loss: %.4f in %ds"
                          % (train_acc, train_rmse, train_error, time.time() - t2))
            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            finally:
                coord.request_stop()
                coord.join(threads)

        # Validation
        with tf.Session() as _sess:
            _sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=_sess, coord=coord)
            try:
                while not coord.should_stop():
                    images, labels = _sess.run([valid_image_tensor, valid_target_tensor])
                    valid_rmse, valid_acc, valid_pred = model.eval_metric(images, labels)
                    print(model.model_filename)
                    print('validation accuracy: %.4f, validation rmse: %.4f' % (valid_acc, valid_rmse))
            except tf.errors.OutOfRangeError:
                print('Done validation -- epoch limit reached')
            finally:
                coord.request_stop()
                coord.join(threads)

        # Checkpointing/early stopping
        if use_early_stop:
            print("%d/%d chances left" % (early_stop_max_iter - early_stop_iters, early_stop_max_iter))
            early_stop_iters += 1
            if valid_rmse < prev_valid_rmse:
                prev_valid_rmse = valid_rmse
                early_stop_iters = 0
                saver.save(sess, model.model_filename)
            elif early_stop_iters == early_stop_max_iter:
                print("Early stopping ({} vs. {})...".format(prev_valid_rmse, valid_rmse))
                traintime = (time.time() - t0)
                print("total training time %ds" % traintime)
                return traintime
        else:
            saver.save(sess, model.model_filename)


def test(model, sess, saver, test_data, log=True):
    """
    Tester
    """
    batch_image, batch_target = test_data.get_batch_tensor(batch_size=TEST_SIZE)
    test_rmse, _ = model.eval_metric(batch_image, batch_target)
    if log:
        print("Final test RMSE: {}".format(test_rmse))
    return test_rmse


if __name__ == '__main__':
    """ Main function. Parses arguments """
    # Set up command line params
    parser = argparse.ArgumentParser(description='Trains/evaluates model.')

    # Required
    parser.add_argument('--model', metavar='MODEL_NAME', type=str, choices=['VGGNET', 'ALEXNET', 'LENET'],
                        help='the name of the model to use', required=True)
    parser.add_argument('--mode', metavar='MODE', type=str, choices=['train', 'test'],
                        help='the mode to run the program in', default='train', required=True)
    parser.add_argument('--function', metavar='FUNCTION', type=str, choices=['classify', 'count'], default='count', required=True)

    # Optional
    parser.add_argument('--batch', metavar='BATCH_SIZE', type=int, default=5000,
                        help='the batch size to use when doing gradient descent')
    parser.add_argument('--learning-rate', metavar='LEARNING-RATE', type=float, default=0.0025)
    parser.add_argument('--no-early', type=str2bool, default=False, help='disable early stopping')
    parser.add_argument('--early-stop-max-iter', metavar='EARLY_STOP_MAX_ITER', type=int, default=300,
                        help='the maximum number of iterations to let the model continue training after reaching a '
                             'minimum validation error')
    parser.add_argument('--max-iters', metavar='MAX_ITERS', type=int, default=100,
                        help='the maximum number of iterations to allow the model to train for')
    parser.add_argument('--outfile', type=str, default='modelname', help='output file name')
    parser.add_argument('--difficulty', type=str, default='moderate', choices=['moderate', 'hard'], help='difficulty of task')    

    # Parse args
    args = parser.parse_args()
    # Global args
    model_name = args.model
    mode = args.mode
    function = args.function
    batch_size = args.batch
    learning_rate = args.learning_rate
    use_early_stop = not (args.no_early)
    early_stop_max_iter = args.early_stop_max_iter
    max_iters = args.max_iters
    difficulty = args.difficulty

    with tf.Session() as sess:
        # Process data
        print("Load dataset")
        dataset = load_dataset(function)
        train_data, validation_data, test_data = dataset.train, dataset.validation, dataset.test

        # Define computation graph & Initialize
        print('Building network & initializing variables')
        if model_name == 'ALEXNET':
            model = ALEXNET(function, learning_rate, difficulty)
        elif model_name == 'VGGNET' :
            model = VGGNET(function, learning_rate, difficulty)
        else :
            model = LENET(function, learning_rate, difficulty)

        model.init_sess(sess)
        saver = tf.train.Saver()

        # Train
        traintime = 0
        if mode == 'train':
            traintime = train(
                model, sess, saver, train_data, validation_data,
                batch_size=batch_size, max_iters=max_iters,
                use_early_stop=use_early_stop, early_stop_max_iter=early_stop_max_iter, function=function
            )
        elif mode == 'test':
            print('Loading best checkpointed model')
            saver.restore(sess, model.model_filename)
            test_rmse = test(model, sess, saver, test_data)

        # if(args.outfile == 'modelname') :
        #     outfile = model.model_filename
        # else :
        #     outfile = args.outfile
        # if os.path.exists('out/'+outfile+'.txt') == False:
        #     with open('out/'+outfile+'.txt', 'w') as myfile:
        #         myfile.close()
        #     os.chmod('out/'+outfile+'.txt', 0o777)
        # with open('out/'+outfile+'.txt', "a") as myfile:
        #     myfile.write(model.model_filename+(' %.4f %.4f %.4f %ds\n' % (TRAIN, VALID, TEST, traintime)))
        #     myfile.close()
