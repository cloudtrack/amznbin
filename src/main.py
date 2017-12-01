import argparse
import time

import numpy as np
import tensorflow as tf
from PIL import ImageDraw, Image
from numpy.distutils.fcompiler import str2bool

from dataset import load_dataset
from models import ALEXNET, VGGNET, LENET

def color_scaler(img) :
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    img[:,:,0] = gray
    img[:,:,1] = 0
    img[:,:,2] = b
    return img

def train(model, sess, saver, train_data, valid_data, batch_size, max_iters, use_early_stop, early_stop_max_iter,
          function, difficulty):
    """
    Trainer 
    """
    t0 = time.time()
    # Optimize
    prev_valid_metric = float("Inf")
    early_stop_iters = 0
    train_image_tensor, train_image_index_tensor = train_data.get_batch_tensor(batch_size)
    valid_image_tensor, valid_image_index_tensor = valid_data.get_batch_tensor(batch_size)

    if function == 'count' and difficulty == 'hard':
        metric = 'rmse'
    else:
        metric = 'accuracy'

    train_log = open("train_log.txt", 'w')
    for i in range(max_iters):
        print('==== New epoch started ====')
        # Training
        with tf.Session() as _sess:
            _sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=_sess, coord=coord)
            batch_cnt = 0
            try:
                while not coord.should_stop():
                    t2 = time.time()
                    images, indices = _sess.run([train_image_tensor, train_image_index_tensor])
                    images = [color_scaler(image) for image in images]
                    labels = train_data.get_labels_from_indices(indices, function, difficulty)
                    # Debug
                    # img = Image.fromarray(images[0])
                    # draw = ImageDraw.Draw(img)
                    # draw.text((0, 0), 'label: {0}'.format(labels[0]), (255, 255, 255))
                    # img.show()
                    model.train_iteration(images, labels)
                    model.eval_loss(images, labels)
                    train_metric, train_pred = model.eval_metric(images, labels)
                    print_string = "iter: " + str(i) + "\tbatch: "+str(batch_cnt)+"\ttrain " + metric + ": %.4f in %ds" % (train_metric, time.time() - t2)
                    print(print_string)
                    train_log.write(print_string + "\n")
                    batch_cnt = batch_cnt + 1
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
            final_valid_metric = 0
            iterations = 0
            try:
                while not coord.should_stop():
                    images, indices = _sess.run([valid_image_tensor, valid_image_index_tensor])
                    labels = valid_data.get_labels_from_indices(indices, function, difficulty)
                    valid_metric, valid_pred = model.eval_metric(images, labels)
                    print('validation ' + metric + ': %.4f' % (valid_metric))
                    final_valid_metric = final_valid_metric + valid_metric
                    iteration = iterations + 1
            except tf.errors.OutOfRangeError:
                print('final validation ' + metric + ': %.4f' % (final_valid_metric/iterations))
                print('Done validation -- epoch limit reached')
            finally:
                coord.request_stop()
                coord.join(threads)

        # Checkpointing/early stopping
        if use_early_stop:
            print("%d/%d chances left" % (early_stop_max_iter - early_stop_iters, early_stop_max_iter))
            early_stop_iters += 1
            if valid_metric < prev_valid_metric:
                prev_valid_metric = valid_metric
                early_stop_iters = 0
                saver.save(sess, model.model_filename)
            elif early_stop_iters == early_stop_max_iter:
                print("Early stopping ({} vs. {})...".format(prev_valid_metric, valid_metric))
                traintime = (time.time() - t0)
                print("total training time %ds" % traintime)
                return traintime
        else:
            saver.save(sess, model.model_filename)
    train_log.close()


def test(model, sess, saver, test_data, function, difficulty, batch_size, log=True):
    """
    Tester
    """
    batch_image, batch_image_index = test_data.get_batch_tensor(batch_size=batch_size)

    if function == 'count' and difficulty == 'hard':
        metric = 'rmse'
    else:
        metric = 'accuracy'

    with tf.Session() as _sess:
        _sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=_sess, coord=coord)
        try:
            while not coord.should_stop():
                images, indices = _sess.run([batch_image, batch_image_index])
                labels = test_data.get_labels_from_indices(indices, function, difficulty)
                test_metric, _ = model.eval_metric(images, labels)
                print('test ' + metric + ': %.4f' % (test_metric))
        except tf.errors.OutOfRangeError:
            print('Something went wrong while testing')
        finally:
            coord.request_stop()
            coord.join(threads)
        return test_metric


if __name__ == '__main__':
    """ Main function. Parses arguments """
    # Set up command line params
    parser = argparse.ArgumentParser(description='Trains/evaluates model.')

    # Required
    parser.add_argument('--model', metavar='MODEL_NAME', type=str, choices=['VGGNET', 'ALEXNET', 'LENET'],
                        help='the name of the model to use', required=True)
    parser.add_argument('--mode', metavar='MODE', type=str, choices=['train', 'test'],
                        help='the mode to run the program in', default='train', required=True)
    parser.add_argument('--function', metavar='FUNCTION', type=str, choices=['classify', 'count'], default='count',
                        required=True)

    # Optional
    parser.add_argument('--batch', metavar='BATCH_SIZE', type=int, default=5,
                        help='the batch size to use when doing gradient descent')
    parser.add_argument('--learning-rate', metavar='LEARNING-RATE', type=float, default=0.01)
    parser.add_argument('--no-early', type=str2bool, default=False, help='disable early stopping')
    parser.add_argument('--early-stop-max-iter', metavar='EARLY_STOP_MAX_ITER', type=int, default=300,
                        help='the maximum number of iterations to let the model continue training after reaching a '
                             'minimum validation error')
    parser.add_argument('--max-iters', metavar='MAX_ITERS', type=int, default=500,
                        help='the maximum number of iterations to allow the model to train for')
    parser.add_argument('--outfile', type=str, default='modelname', help='output file name')
    parser.add_argument('--difficulty', type=str, default='moderate', choices=['moderate', 'hard'],
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
        dataset = load_dataset()
        train_data, validation_data, test_data = dataset.train, dataset.validation, dataset.test

        # Train
        traintime = 0
        if mode == 'train':
            traintime = train(model, sess, saver, train_data, validation_data, batch_size=batch_size, max_iters=max_iters,
                use_early_stop=use_early_stop, early_stop_max_iter=early_stop_max_iter, function=function, difficulty=difficulty)
        
        print('Loading best checkpointed model')
        saver.restore(sess, model.model_filename)
        test_metric = test(model, sess, saver, test_data, function, difficulty, batch_size)

        if(args.outfile == 'modelname') :
            outfile = model.model_filename
        else :
            outfile = args.outfile
        if os.path.exists('out/'+outfile+'.txt') == False:
            with open('out/'+outfile+'.txt', 'w') as myfile:
                myfile.close()
            os.chmod('out/'+outfile+'.txt', 0o777)
        with open('out/'+outfile+'.txt', "a") as myfile:
            myfile.write(model.model_filename+(' %.4f %ds\n' % (test_metric, traintime)))
            myfile.close()
