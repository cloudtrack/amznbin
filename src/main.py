import argparse
import json
import time

import tensorflow as tf
from numpy.distutils.fcompiler import str2bool

from constants import DATASET_DIR
from dataset import load_dataset
from models import ALEXNET  # , VGG16, INCEPTION


def train(model, sess, saver, train_data, valid_data, batch_size, max_iters, use_early_stop, early_stop_max_iter):
    """
    Trainer 
    """
    t0 = time.time()
    batch_image, batch_target = train_data.next_batch(batch_size)
    train_error = model.eval_loss(batch_image, batch_target)
    train_rmse, train_pred = model.eval_rmse(batch_image, batch_target)
    valid_rmse, valid_pred = model.eval_rmse(valid_data.images, valid_data.labels)

    print("train loss: %.3f, train rmse: %.3f, valid rmse: %.3f" % (train_error, train_rmse, valid_rmse))

    # Optimize
    prev_valid_rmse = float("Inf")
    early_stop_iters = 0
    for i in range(max_iters):
        t1 = time.time()
        batch_image, batch_target = train_data.next_batch(batch_size)
        model.train_iteration(batch_image, batch_target)

        # Evaluate
        train_error, train_rsme = model.eval_loss(batch_image, batch_target)
        valid_rmse, _ = model.eval_rmse(valid_data.images, valid_data.labels)
        print(model.model_filename)
        print("train loss: %.4f, train rmse: %.4f, valid rmse: %.4f in %ds" % (
            train_error, train_rmse, valid_rmse, time.time() - t1))

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


def test(model, sess, saver, test_data, train_data, valid_data, log=True):
    """
    Tester
    """
    train_rmse, _ = model.eval_rmse(train_data.images, train_data.labels)
    valid_rmse, _ = model.eval_rmse(valid_data.images, valid_data.labels)
    if log:
        print("Final train RMSE: {}".format(train_rmse))
        print("Final valid RMSE: {}".format(valid_rmse))

    test_rmse, _ = model.eval_rmse(test_data.images, test_data.labels)
    if log:
        print("Final test RMSE: {}".format(test_rmse))

    print('%.4f %.4f %.4f' % (train_rmse, valid_rmse, test_rmse))

    return train_rmse, valid_rmse, test_rmse


if __name__ == '__main__':
    """ Main function. Parses arguments """
    # Set up command line params
    parser = argparse.ArgumentParser(description='Trains/evaluates model.')

    # Required
    parser.add_argument('--model', metavar='MODEL_NAME', type=str, choices=['VGG16', 'ALEXNET', 'INCEPTION'],
                        help='the name of the model to use', required=True)
    parser.add_argument('--mode', metavar='MODE', type=str, choices=['train', 'test'],
                        help='the mode to run the program in', default='train', required=True)

    # Optional
    parser.add_argument('--model-params', metavar='MODEL_PARAMS_JSON', type=str, default='{}',
                        help='JSON string containing model params: D, Dprime, hidden_layer_num, hidden_units_per_layer, learning_rate, dropoutkeep, lam, alpha, model name')
    parser.add_argument('--batch', metavar='BATCH_SIZE', type=int, default=128,
                        help='the batch size to use when doing gradient descent')
    parser.add_argument('--no-early', type=str2bool, default=False, help='disable early stopping')
    parser.add_argument('--early-stop-max-iter', metavar='EARLY_STOP_MAX_ITER', type=int, default=300,
                        help='the maximum number of iterations to let the model continue training after reaching a '
                             'minimum validation error')
    parser.add_argument('--max-iters', metavar='MAX_ITERS', type=int, default=10000,
                        help='the maximum number of iterations to allow the model to train for')
    parser.add_argument('--outfile', type=str, default='modelname', help='output file name')

    # Parse args
    args = parser.parse_args()
    # Global args
    model_name = args.model
    mode = args.mode
    model_params = json.loads(args.model_params)
    batch_size = args.batch
    use_early_stop = not (args.no_early)
    early_stop_max_iter = args.early_stop_max_iter
    max_iters = args.max_iters

    with tf.Session() as sess:
        # Process data
        print("Load dataset")
        dataset = load_dataset(DATASET_DIR)
        train_data, validation_data, test_data = dataset.train, dataset.validation, dataset.test

        # Define computation graph & Initialize
        print('Building network & initializing variables')
        if model_name == 'ALEXNET':
            model = ALEXNET()
        # elif model_name == 'VGG16' :
        #     model = VGG16()
        # else :
        #     model = INCEPTION()

        model.init_sess(sess)
        saver = tf.train.Saver()
        # Train
        traintime = 0
        if mode == 'train':
            traintime = train(model, sess, saver, train_data, validation_data, batch_size=batch_size,
                              max_iters=max_iters, use_early_stop=use_early_stop,
                              early_stop_max_iter=early_stop_max_iter)
        elif mode == 'test':
            print('Loading best checkpointed model')
            saver.restore(sess, model.filename)
            TRAIN, VALID, TEST = test(model, sess, saver, test_data, train_data, validation_data)

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
