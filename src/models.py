import numpy as np
import tensorflow as tf

from constants import IMAGE_SIZE, CLASS_SIZE


def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())


class _Base(object):

    """ Base structure """
    def __init__(self, function, learning_rate):
        # Internal counter to keep track of current iteration
        self._iters = 0

        # Input
        self.learning_rate = learning_rate

        self.image = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 3])
        if function == 'count':
            self.target = tf.placeholder(tf.float32, [None, 1])
        else:
            self.target = tf.placeholder(tf.float32, [None, CLASS_SIZE])

        self.function = function

        self.model_filename = 'model/' + function + '.ckpt'

        # Call methods to initialize variables and operations 
        self._init_vars()
        self._init_ops()

        # RMSE
        self.rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.target, self.pred))))
        #self.rmse, _ = tf.metrics.root_mean_squared_error(labels=self.target, predictions=self.pred)
        # Accuracy
        pred_labels = tf.greater_equal(self.pred, 0.2)
        acc, _ = tf.metrics.accuracy(labels=self.target, predictions=pred_labels)
        self.accuracy = tf.divide(tf.multiply(acc, CLASS_SIZE), tf.cast(tf.count_nonzero(pred_labels), tf.float32))
        # self.accuracy = tf.subtract(tf.cast(1.0, tf.float64), tf.divide(tf.count_nonzero(tf.subtract(self.target, self.pred)), CLASS_SIZE))
        

    @property
    def filename(self):
        raise NotImplementedError()

    def _init_vars(self):
        raise NotImplementedError()

    def _init_ops(self):
        raise NotImplementedError()

    def init_sess(self, sess):
        """ 
        Initializes tensorflow session 
        :param sess: tensorflow session to execute tensorflow operations
        """
        self.sess = sess
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        self.sess.run(init_g)
        self.sess.run(init_l)        

    def train_iteration(self, imagedata, targetdata):
        """
        Runs each train iteration
        """
        print('train_iteration')
        feed_dict = {self.image: imagedata, self.target: targetdata}
        self.sess.run(self.optimize_steps, feed_dict=feed_dict)
        self._iters += 1

    def eval_metric(self, imagedata, targetdata):
        """ 
        Calculates RMSE 
        """
        print('eval_metric')
        feed_dict = {self.image: imagedata, self.target: targetdata}
        return self.sess.run([self.rmse, self.accuracy, self.pred], feed_dict=feed_dict)

    def eval_loss(self, imagedata, targetdata):
        """
        Calculates loss
        """
        print('eval_loss')
        feed_dict = {self.image: imagedata, self.target: targetdata}
        return self.sess.run(self.loss, feed_dict=feed_dict)


class ALEXNET(_Base):
    """ AlexNet model structrue """

    def __init__(self, function, learning_rate):
        super(ALEXNET, self).__init__(function, learning_rate)

    @property
    def filename(self):
        return 'alexnet'

    def _init_vars(self):
        """ Build layers of the model """
        # self.pred = tf.squeeze(self.build_layers(self.image), squeeze_dims=[1])
        self.pred = tf.squeeze(self.build_layers(self.image))


    def _init_ops(self):
        """ Calculates loss and performs gradient descent """
        # Loss     
        if self.function == 'classify':
            self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.target, logits=self.pred)
        else:
            self.loss = tf.reduce_sum(tf.square(tf.subtract(self.target, self.pred)))

        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        # Optimize the weights
        self.optimize_steps = self.optimizer.minimize(self.loss, var_list=self.parameters)

    def build_layers(self, image):
        """
        Builds layers 
        """
        self.parameters = []

        # conv1
        with tf.name_scope('conv1') as scope:
            kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(image, kernel, [1, 4, 4, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(bias, name=scope)
            # print_activations(conv1)
            self.parameters += [kernel, biases]

        # lrn1
        with tf.name_scope('lrn1') as scope:
            lrn1 = tf.nn.local_response_normalization(conv1,
                                                      alpha=1e-4,
                                                      beta=0.75,
                                                      depth_radius=2,
                                                      bias=2.0)

            # pool1
            pool1 = tf.nn.max_pool(lrn1,
                                 ksize=[1, 3, 3, 1],
                                 strides=[1, 2, 2, 1],
                                 padding='VALID',
                                 name='pool1')
            # print_activations(pool1)

        # conv2
        with tf.name_scope('conv2') as scope:
            kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32), trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(bias, name=scope)
            self.parameters += [kernel, biases]
            # print_activations(conv2)

        # lrn2
        with tf.name_scope('lrn2') as scope:
            lrn2 = tf.nn.local_response_normalization(conv2,
                                                      alpha=1e-4,
                                                      beta=0.75,
                                                      depth_radius=2,
                                                      bias=2.0)

            # pool2
            pool2 = tf.nn.max_pool(lrn2,
                                 ksize=[1, 3, 3, 1],
                                 strides=[1, 2, 2, 1],
                                 padding='VALID',
                                 name='pool2')
            # print_activations(pool2)

        # # conv3
        # with tf.name_scope('conv3') as scope:
        #     kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384], dtype=tf.float32, stddev=1e-1), name='weights')
        #     conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        #     biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), trainable=True, name='biases')
        #     bias = tf.nn.bias_add(conv, biases)
        #     conv3 = tf.nn.relu(bias, name=scope)
        #     self.parameters += [kernel, biases]
        #     # print_activations(conv3)

        # # conv4
        # with tf.name_scope('conv4') as scope:
        #     kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype=tf.float32, stddev=1e-1), name='weights')
        #     conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        #     biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        #     bias = tf.nn.bias_add(conv, biases)
        #     conv4 = tf.nn.relu(bias, name=scope)
        #     self.parameters += [kernel, biases]
        #     # print_activations(conv4)

        # conv5
        with tf.name_scope('conv5') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 256], dtype=tf.float32, stddev=1e-1), name='weights')
            # kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1), name='weights')
            # conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
            conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv5 = tf.nn.relu(bias, name=scope)
            self.parameters += [kernel, biases]
            # print_activations(conv5)

        # pool5
        pool5 = tf.nn.max_pool(conv5,
                             ksize=[1, 3, 3, 1],
                             strides=[1, 2, 2, 1],
                             padding='VALID',
                             name='pool5')
        # print_activations(pool5)

        # fullyconnected6
        fc6W = tf.Variable(tf.constant(0.0, shape=[9216, 512], dtype=tf.float32), trainable=True, name='weights')
        fc6b = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
        fc6 = tf.nn.relu_layer(tf.reshape(pool5, [-1, int(np.prod(pool5.get_shape()[1:]))]), fc6W, fc6b)
        # print_activations(fc6)

        self.parameters += [fc6W, fc6b]
        
        # fullyconnected7
        fc7W = tf.Variable(tf.constant(0.0, shape=[512, 512], dtype=tf.float32), trainable=True, name='weights')
        fc7b = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
        fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)
        # print_activations(fc7)
        self.parameters += [fc7W, fc7b]
        
        if self.function == 'classify' :
            OUTPUT = CLASS_SIZE
        else : 
            OUTPUT = 1

        # fullyconnected8
        fc8W = tf.Variable(tf.constant(0.0, shape=[512, OUTPUT], dtype=tf.float32), trainable=True, name='weights')
        fc8b = tf.Variable(tf.constant(0.0, shape=[OUTPUT], dtype=tf.float32), trainable=True, name='biases')
        fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
        # print_activations(fc8)

        self.parameters += [fc8W, fc8b]
        
        if self.function == 'classify' :
            # fc8 = tf.nn.softmax(fc8)
            fc8 = tf.sigmoid(fc8)

        return fc8
