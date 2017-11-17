import json

import numpy as np
import tensorflow as tf

from constants import IMAGE_SIZE, CLASS_SIZE, PARAM_DIR


def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())

class _Base(object):

    """ Base structure """
    def __init__(self, function, learning_rate, difficulty):
        # Internal counter to keep track of current iteration
        self._iters = 0

        # Input
        self.learning_rate = learning_rate
        self.difficulty = difficulty

        self.image = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 3])
        
        if function == 'classify':
            self.param = json.loads(open(PARAM_DIR+'model_parameters_classify.json').read())
            self.OUTPUT = CLASS_SIZE
        else :
            self.param = json.loads(open(PARAM_DIR+'model_parameters_count.json').read())
            if difficulty == 'moderate' :
                self.OUTPUT = 12   
            else:
                self.OUTPUT = 1

        self.target = tf.placeholder(tf.float32, [None, self.OUTPUT])

        self.function = function

        self.model_filename = 'model/' + function + '_'+ difficulty +'.ckpt'

        # Call methods to initialize variables and operations 
        self._init_vars()
        self._init_ops()

        if self.function == 'classify':
            # Accuracy
            pred_labels = tf.cast(tf.greater_equal(self.pred, 0.2), tf.int32)

            def pred_longer() :
                total = tf.cast(tf.count_nonzero(pred_labels), tf.float32) 
                where = tf.equal(pred_labels, tf.constant(1, dtype=tf.int32))
                indices = tf.reshape(tf.where(where), [-1])
                count = tf.map_fn(lambda x: tf.cast(tf.equal(pred_labels[x], tf.constant(1, dtype=tf.int32)), tf.int32), indices, tf.int32)
                return tf.cast(tf.reduce_sum(count), tf.float32), total

            def target_longer() :
                total = tf.cast(tf.count_nonzero(self.target), tf.float32)
                where = tf.equal(self.target, tf.constant(1, dtype=tf.float32))
                indices = tf.reshape(tf.where(where), [-1])
                count = tf.map_fn(lambda x: tf.cast(tf.equal(pred_labels[x], tf.constant(1, dtype=tf.int32)), tf.int32), indices, tf.int32)
                return tf.cast(tf.reduce_sum(count), tf.float32), total

            count, total = tf.cond(tf.greater(tf.cast(tf.count_nonzero(pred_labels), tf.float32), tf.cast(tf.count_nonzero(self.target), tf.float32)), lambda: pred_longer(), lambda: target_longer())

            self.metric = tf.multiply(tf.divide(count, total), 100)
        
        elif (self.function == 'count') and (self.difficulty == 'moderate') :
            # Accuracy 
            self.metric = tf.multiply(tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.target, 1)), tf.float32)), 100)

        else :
            # RMSE
            self.metric = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.target, self.pred))))
        
    @property
    def filename(self):
        raise NotImplementedError()

    def _init_vars(self):
        """ Build layers of the model """
        self.pred = tf.squeeze(self.build_layers(self.image))

    def _init_ops(self):
        """ Calculates loss and performs gradient descent """
        # Loss     
        if self.function == 'classify':
            self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.target, logits=self.pred)
        elif self.function == 'count' and self.difficulty == 'moderate':
            self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.target, logits=self.pred)
        else:
            self.loss = tf.reduce_sum(tf.square(tf.subtract(self.target, self.pred)))

        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        # Optimize the weights
        self.optimize_steps = self.optimizer.minimize(self.loss, var_list=self.variables)

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

    def train_iteration(self, image_data, target_data):
        """
        Runs each train iteration
        """
        print('train_iteration')
        feed_dict = {self.image: image_data, self.target: target_data}
        self.sess.run(self.optimize_steps, feed_dict=feed_dict)
        self._iters += 1

    def eval_metric(self, image_data, target_data):
        """ 
        Calculates RMSE 
        """
        print('eval_metric')
        feed_dict = {self.image: image_data, self.target: target_data}
        return self.sess.run([self.metric, self.pred], feed_dict=feed_dict)

    def eval_loss(self, image_data, target_data):
        """
        Calculates loss
        """
        print('eval_loss')
        feed_dict = {self.image: image_data, self.target: target_data}
        return self.sess.run(self.loss, feed_dict=feed_dict)


class ALEXNET(_Base):
    """ AlexNet model structrue """

    def __init__(self, function, learning_rate, difficulty):
        super(ALEXNET, self).__init__(function, learning_rate, difficulty)

    @property
    def filename(self):
        return 'alexnet'

    def _init_vars(self):
        super(ALEXNET, self)._init_vars()

    def _init_ops(self):
        super(ALEXNET, self)._init_ops()

    def build_layers(self, image):
        """
        Builds layers 
        """
        self.variables = []

        # conv1
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, self.param['alexnet_conv1_kernel']], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(image, kernel, [1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[self.param['alexnet_conv1_kernel']], dtype=tf.float32), trainable=True)
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias)
        self.variables += [kernel, biases]

        # lrn1
        lrn1 = tf.nn.local_response_normalization(conv1, alpha=1e-4, beta=0.75, depth_radius=2, bias=2.0)

        # pool1
        pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        # conv2
        kernel = tf.Variable(tf.truncated_normal([5, 5, self.param['alexnet_conv1_kernel'], self.param['alexnet_conv2_kernel']], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[self.param['alexnet_conv2_kernel']], dtype=tf.float32), trainable=True)
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias)
        self.variables += [kernel, biases]

        # lrn2
        lrn2 = tf.nn.local_response_normalization(conv2, alpha=1e-4, beta=0.75, depth_radius=2, bias=2.0)

        # pool2
        pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        # conv3
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, self.param['alexnet_conv3_kernel']], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[self.param['alexnet_conv3_kernel']], dtype=tf.float32), trainable=True)
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias)
        self.variables += [kernel, biases]

        # conv4
        kernel = tf.Variable(tf.truncated_normal([3, 3, self.param['alexnet_conv3_kernel'], self.param['alexnet_conv4_kernel']], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[self.param['alexnet_conv4_kernel']], dtype=tf.float32), trainable=True)
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias)
        self.variables += [kernel, biases]

        # conv5
        kernel = tf.Variable(tf.truncated_normal([3, 3, self.param['alexnet_conv4_kernel'], self.param['alexnet_conv5_kernel']], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[self.param['alexnet_conv5_kernel']], dtype=tf.float32), trainable=True)
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias)
        self.variables += [kernel, biases]

        # pool5
        pool5 = tf.nn.max_pool(conv5,  ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        # fullyconnected6
        kernel = tf.Variable(tf.constant(0.0, shape=[self.param['alexnet_fc6_kernel1'], self.param['alexnet_fc6_kernel2']], dtype=tf.float32), trainable=True)
        biases = tf.Variable(tf.constant(0.0, shape=[self.param['alexnet_fc6_kernel2']], dtype=tf.float32), trainable=True)
        fc6 = tf.nn.relu_layer(tf.reshape(pool5, [-1, int(np.prod(pool5.get_shape()[1:]))]), kernel, biases)
        self.variables += [kernel, biases]
        
        # fullyconnected7
        kernel = tf.Variable(tf.constant(0.0, shape=[self.param['alexnet_fc6_kernel2'], self.param['alexnet_fc7_kernel']], dtype=tf.float32), trainable=True)
        biases = tf.Variable(tf.constant(0.0, shape=[self.param['alexnet_fc7_kernel']], dtype=tf.float32), trainable=True)
        fc7 = tf.nn.relu_layer(fc6, kernel, biases)
        self.variables += [kernel, biases]

        # fullyconnected8
        kernel = tf.Variable(tf.constant(0.0, shape=[self.param['alexnet_fc7_kernel'], self.OUTPUT], dtype=tf.float32), trainable=True)
        biases = tf.Variable(tf.constant(0.0, shape=[self.OUTPUT], dtype=tf.float32), trainable=True)
        fc8 = tf.nn.xw_plus_b(fc7, kernel, biases)
        self.variables += [kernel, biases]
        
        if self.function == 'classify' :
            fc8 = tf.sigmoid(fc8)
        elif self.difficulty == 'moderate' :
            fc8 = tf.nn.softmax(fc8)

        return fc8


class VGGNET(_Base):
    """ VGGNet model structrue """

    def __init__(self, function, learning_rate, difficulty):
        super(VGGNET, self).__init__(function, learning_rate, difficulty)

    @property
    def filename(self):
        return 'vggnet'

    def _init_vars(self):
        super(VGGNET, self)._init_vars()

    def _init_ops(self):
        super(VGGNET, self)._init_ops()

    def build_layers(self, image):
        """
        Builds layers 
        """
        self.variables = []

        # conv1
        kernel = tf.Variable(tf.truncated_normal([3, 3, 3, vggnet_conv1_kernel], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(image, kernel, [1, 1, 1, 1], padding='VALID')
        biases = tf.Variable(tf.constant(0.0, shape=[vggnet_conv1_kernel], dtype=tf.float32), trainable=True)
        bias = tf.nn.bias_add(conv, biases)
        conv11 = tf.nn.relu(bias)
        self.variables += [kernel, biases]

        kernel = tf.Variable(tf.truncated_normal([3, 3, vggnet_conv1_kernel, vggnet_conv1_kernel], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(conv11, kernel, [1, 1, 1, 1], padding='VALID')
        biases = tf.Variable(tf.constant(0.0, shape=[vggnet_conv1_kernel], dtype=tf.float32), trainable=True)
        bias = tf.nn.bias_add(conv, biases)
        conv12 = tf.nn.relu(bias)
        self.variables += [kernel, biases]

        # pool1
        pool1 = tf.nn.max_pool(conv12, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        # conv2
        kernel = tf.Variable(tf.truncated_normal([3, 3, vggnet_conv1_kernel, vggnet_conv2_kernel], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='VALID')
        biases = tf.Variable(tf.constant(0.0, shape=[vggnet_conv2_kernel], dtype=tf.float32), trainable=True)
        bias = tf.nn.bias_add(conv, biases)
        conv21 = tf.nn.relu(bias)
        self.variables += [kernel, biases]

        kernel = tf.Variable(tf.truncated_normal([3, 3, vggnet_conv2_kernel, vggnet_conv2_kernel], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(conv21, kernel, [1, 1, 1, 1], padding='VALID')
        biases = tf.Variable(tf.constant(0.0, shape=[vggnet_conv2_kernel], dtype=tf.float32), trainable=True)
        bias = tf.nn.bias_add(conv, biases)
        conv22 = tf.nn.relu(bias)
        self.variables += [kernel, biases]

        # pool2
        pool2 = tf.nn.max_pool(conv22,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')

        # conv3
        kernel = tf.Variable(tf.truncated_normal([3, 3, vggnet_conv2_kernel, vggnet_conv3_kernel], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='VALID')
        biases = tf.Variable(tf.constant(0.0, shape=[vggnet_conv3_kernel], dtype=tf.float32), trainable=True)
        bias = tf.nn.bias_add(conv, biases)
        conv31 = tf.nn.relu(bias)
        self.variables += [kernel, biases]

        kernel = tf.Variable(tf.truncated_normal([3, 3, vggnet_conv3_kernel, vggnet_conv3_kernel], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(conv31, kernel, [1, 1, 1, 1], padding='VALID')
        biases = tf.Variable(tf.constant(0.0, shape=[vggnet_conv3_kernel], dtype=tf.float32), trainable=True)
        bias = tf.nn.bias_add(conv, biases)
        conv32 = tf.nn.relu(bias)
        self.variables += [kernel, biases]

        kernel = tf.Variable(tf.truncated_normal([3, 3, vggnet_conv3_kernel, vggnet_conv3_kernel], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(conv32, kernel, [1, 1, 1, 1], padding='VALID')
        biases = tf.Variable(tf.constant(0.0, shape=[vggnet_conv3_kernel], dtype=tf.float32), trainable=True)
        bias = tf.nn.bias_add(conv, biases)
        conv33 = tf.nn.relu(bias)
        self.variables += [kernel, biases]

        # pool3
        pool3 = tf.nn.max_pool(conv33, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # conv4
        kernel = tf.Variable(tf.truncated_normal([3, 3, vggnet_conv3_kernel, vggnet_conv4_kernel], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='VALID')
        biases = tf.Variable(tf.constant(0.0, shape=[vggnet_conv4_kernel], dtype=tf.float32), trainable=True)
        bias = tf.nn.bias_add(conv, biases)
        conv41 = tf.nn.relu(bias)
        self.variables += [kernel, biases]

        kernel = tf.Variable(tf.truncated_normal([3, 3, vggnet_conv4_kernel, vggnet_conv4_kernel], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(conv41, kernel, [1, 1, 1, 1], padding='VALID')
        biases = tf.Variable(tf.constant(0.0, shape=[vggnet_conv4_kernel], dtype=tf.float32), trainable=True)
        bias = tf.nn.bias_add(conv, biases)
        conv42 = tf.nn.relu(bias)
        self.variables += [kernel, biases]

        kernel = tf.Variable(tf.truncated_normal([3, 3, vggnet_conv4_kernel, vggnet_conv4_kernel], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(conv42, kernel, [1, 1, 1, 1], padding='VALID')
        biases = tf.Variable(tf.constant(0.0, shape=[vggnet_conv4_kernel], dtype=tf.float32), trainable=True)
        bias = tf.nn.bias_add(conv, biases)
        conv43 = tf.nn.relu(bias)
        self.variables += [kernel, biases]

        # pool4
        pool4 = tf.nn.max_pool(conv43, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # conv5
        kernel = tf.Variable(tf.truncated_normal([3, 3, vggnet_conv4_kernel, vggnet_conv5_kernel], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='VALID')
        biases = tf.Variable(tf.constant(0.0, shape=[vggnet_conv5_kernel], dtype=tf.float32), trainable=True)
        bias = tf.nn.bias_add(conv, biases)
        conv51 = tf.nn.relu(bias)
        self.variables += [kernel, biases]

        kernel = tf.Variable(tf.truncated_normal([3, 3, vggnet_conv5_kernel, vggnet_conv5_kernel], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(conv51, kernel, [1, 1, 1, 1], padding='VALID')
        biases = tf.Variable(tf.constant(0.0, shape=[vggnet_conv5_kernel], dtype=tf.float32), trainable=True)
        bias = tf.nn.bias_add(conv, biases)
        conv52 = tf.nn.relu(bias)
        self.variables += [kernel, biases]

        kernel = tf.Variable(tf.truncated_normal([3, 3, vggnet_conv5_kernel, vggnet_conv5_kernel], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(conv52, kernel, [1, 1, 1, 1], padding='VALID')
        biases = tf.Variable(tf.constant(0.0, shape=[vggnet_conv5_kernel], dtype=tf.float32), trainable=True)
        bias = tf.nn.bias_add(conv, biases)
        conv53 = tf.nn.relu(bias)
        self.variables += [kernel, biases]

        # pool5
        pool5 = tf.nn.max_pool(conv53, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # fullyconnected6
        kernel = tf.Variable(tf.constant(0.0, shape=[vggnet_fc6_kernel1, vggnet_fc6_kernel2], dtype=tf.float32), trainable=True)
        biases = tf.Variable(tf.constant(0.0, shape=[vggnet_fc6_kernel2], dtype=tf.float32), trainable=True)
        fc6 = tf.nn.relu_layer(tf.reshape(pool5, [-1, int(np.prod(pool5.get_shape()[1:]))]), kernel, biases)
        self.variables += [kernel, biases]
        
        # fullyconnected7
        kernel = tf.Variable(tf.constant(0.0, shape=[vggnet_fc6_kernel2, vggnet_fc7_kernel], dtype=tf.float32), trainable=True)
        biases = tf.Variable(tf.constant(0.0, shape=[vggnet_fc7_kernel], dtype=tf.float32), trainable=True)
        fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)
        self.variables += [kernel, biases]

        # fullyconnected8
        kernel = tf.Variable(tf.constant(0.0, shape=[vggnet_fc7_kernel, self.OUTPUT], dtype=tf.float32), trainable=True)
        biases = tf.Variable(tf.constant(0.0, shape=[self.OUTPUT], dtype=tf.float32), trainable=True)
        fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
        self.variables += [kernel, biases]
        
        if self.function == 'classify' :
            fc8 = tf.sigmoid(fc8)
        elif self.difficulty == 'moderate' :
            fc8 = tf.nn.softmax(fc8)

        return fc8

class LENET(_Base):
    """ LeNet model structrue """

    def __init__(self, function, learning_rate, difficulty):
        super(LENET, self).__init__(function, learning_rate, difficulty)

    @property
    def filename(self):
        return 'lenet'

    def _init_vars(self):
        super(LENET, self)._init_vars()

    def _init_ops(self):
        super(LENET, self)._init_ops()

    def build_layers(self, image):
        """
        Builds layers 
        """
        self.variables = []

        # conv1
        kernel = tf.Variable(tf.truncated_normal([33, 33, 3, lenet_conv1_kernel], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(image, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[lenet_conv1_kernel], dtype=tf.float32), trainable=True)
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.tanh(bias)
        self.variables += [kernel, biases]

        # pool1
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='VALID')

        # conv2
        kernel = tf.Variable(tf.truncated_normal([33, 33, lenet_conv1_kernel, lenet_conv2_kernel], dtype=tf.float32, stddev=1e-1))
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[lenet_conv2_kernel], dtype=tf.float32), trainable=True)
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.tanh(bias)
        self.variables += [kernel, biases]

        # pool2
        pool2 = tf.nn.max_pool(lrn2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='VALID')
 
        # fullyconnected3
        kernel = tf.Variable(tf.constant(0.0, shape=[lenet_fc3_kernel1, lenet_fc3_kernel2], dtype=tf.float32), trainable=True)
        biases = tf.Variable(tf.constant(0.0, shape=[lenet_fc3_kernel2], dtype=tf.float32), trainable=True)
        fc3 = tf.nn.tanh(tf.nn.xw_plus_b(tf.reshape(pool2, [-1, int(np.prod(pool2.get_shape()[1:]))]), kernel, biases))
        self.variables += [kernel, biases]

        # fullyconnected4
        kernel = tf.Variable(tf.constant(0.0, shape=[lenet_fc3_kernel2, self.OUTPUT], dtype=tf.float32), trainable=True)
        biases = tf.Variable(tf.constant(0.0, shape=[self.OUTPUT], dtype=tf.float32), trainable=True)
        fc4 = tf.nn.xw_plus_b(fc3, kernel, biases)
        self.variables += [kernel, biases]
        
        if self.function == 'classify' :
            fc4 = tf.sigmoid(fc4)
        elif self.difficulty == 'moderate' :
            fc4 = tf.nn.softmax(fc4)

        return fc4
