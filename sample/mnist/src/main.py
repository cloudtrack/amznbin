import argparse
import time

import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
from numpy.distutils.fcompiler import str2bool

from models import ALEXNET, VGGNET, LENET

from tensorflow.examples.tutorials.mnist import input_data

def train(model, sess, saver, train_data, valid_data, batch_size, max_iters, use_early_stop, early_stop_max_iter,
          function, difficulty):
    """
    Trainer 
    """
    t0 = time.time()
    # Optimize
    prev_valid_metric = float("Inf")
    early_stop_iters = 0

    if function == 'count' and difficulty == 'hard':
        metric = 'rmse'
        prev_train_metric = float("Inf")
        prev_valid_metric = float("Inf")
    else:
        metric = 'accuracy'
        prev_train_metric = 0
        prev_valid_metric = 0

    #train_log = open("train_log.txt", 'w')

    for i in range(max_iters):
        print('==== New epoch started ====')
        # Training
        final_train_metric = 0
        for batch_cnt in range(100) :
            t2 = time.time()
            
            batch = train_data.next_batch(batch_size)
            images = [img.reshape(28,28,1) for img in batch[0]]
            labels = batch[1]

            model.train_iteration(images, labels)
            train_loss = model.eval_loss(images, labels)
            train_metric, train_pred, train_pred_one = model.eval_metric(images, labels)
            # train_metric = train_metric[0]
            print_string = "epoch: " + str(i) + "\tbatch: " + str(batch_cnt) +"\ttrain " + metric + ": %.4f \tloss: %.4f in %ds" % (train_metric, train_loss, time.time() - t2)
            print(print_string)
            # plt.imshow(images[0].reshape(28,28), interpolation='nearest')
            # plt.axis('off')
            # plt.show()
            print(train_pred[0])
            print('predicted: ' + str(train_pred_one[0]) + ' by %.2f percent' % (train_pred[0][train_pred_one[0]] * 100))
            print('target:    ' + str(np.argmax(labels[0])))
            # plt.close()
            # plt.imshow(images[1].reshape(28,28), interpolation='nearest')
            # plt.axis('off')
            # plt.show()
            print(train_pred[1])
            print('predicted: ' + str(train_pred_one[1]) + ' by %.2f percent' % (train_pred[1][train_pred_one[1]] * 100))
            print('target:    ' + str(np.argmax(labels[1])))
            print('------------------------------------------------------------------------------')
            final_train_metric = final_train_metric + train_metric
            #train_log.write(print_string + "\n")

        final_train_metric = final_train_metric/100
        
        # Validation
        batch = valid_data.next_batch(batch_size*100)
        images = [img.reshape(28,28,1) for img in batch[0]]
        labels = batch[1]
        valid_metric, valid_pred, valid_pred_one = model.eval_metric(images, labels)
        # valid_metric = valid_metric[0]
        print('validation ' + metric + ': %.4f' % (valid_metric))
        print(valid_pred[0])
        print('predicted: ' + str(valid_pred_one[0]) + ' by %.2f percent' % (valid_pred[0][valid_pred_one[0]] * 100))
        print('target:    ' + str(np.argmax(labels[0])))
        
        # Checkpointing/early stopping
        if use_early_stop:
            print("%d/%d chances left" % (early_stop_max_iter - early_stop_iters, early_stop_max_iter))
            print("previous: {} vs. current: {})...".format(prev_valid_metric, valid_metric))
            early_stop_iters += 1

            if metric == 'accuracy' :
                if valid_metric > prev_valid_metric:
                    prev_valid_metric = valid_metric
                    prev_train_metric = final_train_metric
                    early_stop_iters = 0
                    saver.save(sess, model.model_filename)
                elif early_stop_iters == early_stop_max_iter:
                    print("Early stopping ({} vs. {})...".format(prev_valid_metric, final_valid_metric))
                    traintime = (time.time() - t0)
                    print("total training time %ds" % traintime)
                    return prev_train_metric, prev_valid_metric, traintime
            else :
                if valid_metric < prev_valid_metric:
                    prev_valid_metric = valid_metric
                    prev_train_metric = final_train_metric
                    early_stop_iters = 0
                    saver.save(sess, model.model_filename)
                elif early_stop_iters == early_stop_max_iter:
                    print("Early stopping ({} vs. {})...".format(prev_valid_metric, final_valid_metric))
                    traintime = (time.time() - t0)
                    print("total training time %ds" % traintime)
                    return prev_train_metric, prev_valid_metric, traintime

        else:
            saver.save(sess, model.model_filename)
    
    prev_train_metric = final_train_metric
    return prev_train_metric, prev_valid_metric, time.time() - t0

    # train_log.close()


def test(model, sess, saver, test_data, function, difficulty, batch_size, log=True):
    """
    Tester
    """
    if function == 'count' and difficulty == 'hard':
        metric = 'rmse'
    else:
        metric = 'accuracy'

    with tf.Session() as _sess:
        _sess.run(tf.local_variables_initializer())

        batch = test_data.next_batch(batch_size*100)
        images = [img.reshape(28,28,1) for img in batch[0]]
        labels = batch[1]
        test_metric, _, _ = model.eval_metric(images, labels)
        print('test ' + metric + ': %.4f' % (test_metric))

        return test_metric


if __name__ == '__main__':
    """ Main function. Parses arguments """
    # Set up command line params
    parser = argparse.ArgumentParser(description='Trains/evaluates model.')

    # Required
    parser.add_argument('--model', metavar='MODEL_NAME', type=str, choices=['VGGNET', 'ALEXNET', 'LENET'],
                        help='the name of the model to use', default = 'LENET')
    parser.add_argument('--mode', metavar='MODE', type=str, choices=['train', 'test'],
                        help='the mode to run the program in', default='train')
    parser.add_argument('--function', metavar='FUNCTION', type=str, choices=['classify', 'count'], default='classify')

    # Optional
    parser.add_argument('--batch', metavar='BATCH_SIZE', type=int, default=10,
                        help='the batch size to use when doing gradient descent')
    parser.add_argument('--learning-rate', metavar='LEARNING-RATE', type=float, default=0.0001)
    parser.add_argument('--no-early', type=str2bool, default=False, help='disable early stopping')
    parser.add_argument('--early-stop-max-iter', metavar='EARLY_STOP_MAX_ITER', type=int, default=10,
                        help='the maximum number of iterations to let the model continue training after reaching a '
                             'minimum validation error')
    parser.add_argument('--max-iters', metavar='MAX_ITERS', type=int, default=50,
                        help='the maximum number of iterations to allow the model to train for')
    parser.add_argument('--outfile', type=str, default='modelname', help='output file name')
    parser.add_argument('--difficulty', type=str, default='hard', choices=['moderate', 'hard'],
                        help='difficulty of task')
    parser.add_argument('--continue-train', type=str2bool, default=False, help='whether to continue training from previously trained model')

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
    continue_train = args.continue_train

    with tf.Session() as sess:
        # Define computation graph & Initialize
        print('Building network & initializing variables')
        if model_name == 'ALEXNET':
            model = ALEXNET(function, learning_rate, difficulty)
        elif model_name == 'VGGNET':
            model = VGGNET(function, learning_rate, difficulty)
        else:
            model = LENET(function, learning_rate, difficulty)

        model.init_sess(sess)
        saver = tf.train.Saver()

        if continue_train :
            saver.restore(sess, model.model_filename)

        # Process data
        print("Load dataset")
        dataset = input_data.read_data_sets('MNIST_data', one_hot=True)
        train_data, validation_data, test_data = dataset.train, dataset.validation, dataset.test

        # Train
        train_metric=0
        valid_metric=0
        traintime=0
        if mode == 'train':
            train_metric, valid_metric, traintime = train(model, sess, saver, train_data, validation_data, batch_size=batch_size, max_iters=max_iters,
                use_early_stop=use_early_stop, early_stop_max_iter=early_stop_max_iter, function=function, difficulty=difficulty)
        
        print('Loading best checkpointed model')
        saver.restore(sess, model.model_filename)
        test_metric = test(model, sess, saver, test_data, function, difficulty, batch_size)

        results = open("results.txt", 'w')
        results.write("train: %.4f\t valid: %.4f\t test: %.4f\t in %ds \n" % (train_metric, valid_metric, test_metric, traintime))
        results.close()

