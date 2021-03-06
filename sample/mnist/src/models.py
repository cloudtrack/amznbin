import json

import numpy as np
import tensorflow as tf

from constants import IMAGE_SIZE, PARAM_DIR


def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())

class _Base(object):

    """ Base structure """
    def __init__(self, model, function, learning_rate, difficulty):
        # Internal counter to keep track of current iteration
        self._iters = 0

        # Input
        self.learning_rate = learning_rate
        self.difficulty = difficulty

        # self.image = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 3])
        self.image = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 1])
            

        if function == 'classify':
            self.param = json.loads(open(PARAM_DIR+'model_parameters_classify.json').read())
            #with open(ASIN_INDEX_FILE, 'r') as asin_index_file:
            #    asin_index = json.load(asin_index_file)
            #self.OUTPUT = len(asin_index.keys())
            self.OUTPUT = 10
        else :
            self.param = json.loads(open(PARAM_DIR+'model_parameters_count.json').read())
            if difficulty == 'moderate' :
                self.OUTPUT = 12   
            else:
                self.OUTPUT = 1

        self.target = tf.placeholder(tf.float32, [None, self.OUTPUT])

        self.function = function

        self.model_filename = 'model/' + model + '_' + function + '_'+ difficulty +'.ckpt'

        # Call methods to initialize variables and operations 
        self._init_vars()
        self._init_ops()

        if self.function == 'classify':
            # For multilabel accuracy
            # pred_labels = tf.cast(tf.greater_equal(self.pred, 0.2), tf.float32)
            # tp = tf.reduce_sum(tf.multiply(self.target, pred_labels), 1)
            # fn = tf.reduce_sum(tf.multiply(self.target, 1-pred_labels), 1)
            # fp = tf.reduce_sum(tf.multiply(1-self.target, pred_labels), 1)
            # self.metric = tf.multiply(tf.reduce_mean(1 - (tp / (tp + fn + fp))), 100)

            # For single label accuracy
            self.pred_one = tf.argmax(self.pred, 1)
            self.metric = tf.multiply(tf.reduce_mean(tf.cast(tf.equal(self.pred_one, tf.argmax(self.target, 1)), tf.float32)), 100)
 
        elif (self.function == 'count') and (self.difficulty == 'moderate') :
            # Accuracy 
            # batch 1
            # self.pred_one = tf.argmax(self.pred, 0)
            # self.metric = tf.multiply(tf.cast(tf.equal(self.pred_one, tf.argmax(self.target, 1)), tf.float32), 100)
            self.pred_one = tf.argmax(self.pred, 1)
            self.metric = tf.multiply(tf.reduce_mean(tf.cast(tf.equal(self.pred_one, tf.argmax(self.target, 1)), tf.float32)), 100)

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
            # multilabel
            # self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.target, logits=self.pred))
            
            # single label
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.target, logits=self.pred))

        elif self.function == 'count' and self.difficulty == 'moderate':
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.target, logits=self.pred))
        else:
            self.loss = tf.reduce_sum(tf.square(tf.subtract(self.target, self.pred)))

        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        # self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9)
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
        feed_dict = {self.image: image_data, self.target: target_data}
        self.sess.run(self.optimize_steps, feed_dict=feed_dict)
        self._iters += 1

    def eval_metric(self, image_data, target_data):
        """ 
        Calculates RMSE 
        """
        feed_dict = {self.image: image_data, self.target: target_data}

        if self.function == 'classify' :
            # multilabel
            # return self.sess.run([self.metric, tf.nn.sigmoid(self.pred), self.pred], feed_dict=feed_dict)

            # single label
            return self.sess.run([self.metric, tf.nn.softmax(self.pred), self.pred_one], feed_dict=feed_dict)
        elif self.difficulty == 'moderate' :
            return self.sess.run([self.metric, tf.nn.softmax(self.pred), self.pred_one], feed_dict=feed_dict)
        else : 
            return self.sess.run([self.metric, self.pred, self.pred_one], feed_dict=feed_dict)

    def eval_loss(self, image_data, target_data):
        """
        Calculates loss
        """
        feed_dict = {self.image: image_data, self.target: target_data}
        return self.sess.run(self.loss, feed_dict=feed_dict)


class ALEXNET(_Base):
    """ AlexNet model structrue """

    def __init__(self, function, learning_rate, difficulty):
        super(ALEXNET, self).__init__('ALEX', function, learning_rate, difficulty)

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
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, self.param['alexnet_conv1_kernel']], dtype=tf.float32, mean=1e-1, stddev=5e-1))
        conv = tf.nn.conv2d(image, kernel, [1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(1e-4, shape=[self.param['alexnet_conv1_kernel']], dtype=tf.float32), trainable=True)
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias)
        self.variables += [kernel, biases]

        # lrn1
        lrn1 = tf.nn.local_response_normalization(conv1, alpha=1e-4, beta=0.75, depth_radius=2, bias=2.0)

        # pool1
        pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        # conv2
        kernel = tf.Variable(tf.truncated_normal([5, 5, self.param['alexnet_conv1_kernel'], self.param['alexnet_conv2_kernel']], dtype=tf.float32, mean=1e-1, stddev=5e-1))
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(1e-4, shape=[self.param['alexnet_conv2_kernel']], dtype=tf.float32), trainable=True)
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias)
        self.variables += [kernel, biases]

        # lrn2
        lrn2 = tf.nn.local_response_normalization(conv2, alpha=1e-4, beta=0.75, depth_radius=2, bias=2.0)

        # pool2
        pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        # conv3
        kernel = tf.Variable(tf.truncated_normal([3, 3, self.param['alexnet_conv2_kernel'], self.param['alexnet_conv3_kernel']], dtype=tf.float32, mean=1e-1, stddev=5e-1))
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(1e-4, shape=[self.param['alexnet_conv3_kernel']], dtype=tf.float32), trainable=True)
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias)
        self.variables += [kernel, biases]

        # conv4
        kernel = tf.Variable(tf.truncated_normal([3, 3, self.param['alexnet_conv3_kernel'], self.param['alexnet_conv4_kernel']], dtype=tf.float32, mean=1e-1, stddev=5e-1))
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(1e-4, shape=[self.param['alexnet_conv4_kernel']], dtype=tf.float32), trainable=True)
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias)
        self.variables += [kernel, biases]

        # conv5
        kernel = tf.Variable(tf.truncated_normal([3, 3, self.param['alexnet_conv4_kernel'], self.param['alexnet_conv5_kernel']], dtype=tf.float32, mean=1e-1, stddev=5e-1))
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(1e-4, shape=[self.param['alexnet_conv5_kernel']], dtype=tf.float32), trainable=True)
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias)
        self.variables += [kernel, biases]

        # pool5
        pool5 = tf.nn.max_pool(conv5,  ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        # fullyconnected6
        kernel = tf.Variable(tf.constant(0.0, shape=[self.param['alexnet_fc6_kernel1'], self.param['alexnet_fc6_kernel2']], dtype=tf.float32), trainable=True)
        biases = tf.Variable(tf.constant(1e-4, shape=[self.param['alexnet_fc6_kernel2']], dtype=tf.float32), trainable=True)
        fc6 = tf.nn.relu_layer(tf.reshape(pool5, [-1, int(np.prod(pool5.get_shape()[1:]))]), kernel, biases)
        self.variables += [kernel, biases]
        
        # fullyconnected7
        kernel = tf.Variable(tf.constant(0.0, shape=[self.param['alexnet_fc6_kernel2'], self.param['alexnet_fc7_kernel']], dtype=tf.float32), trainable=True)
        biases = tf.Variable(tf.constant(1e-4, shape=[self.param['alexnet_fc7_kernel']], dtype=tf.float32), trainable=True)
        fc7 = tf.nn.relu_layer(fc6, kernel, biases)
        self.variables += [kernel, biases]

        # fullyconnected8
        kernel = tf.Variable(tf.constant(0.0, shape=[self.param['alexnet_fc7_kernel'], self.OUTPUT], dtype=tf.float32), trainable=True)
        biases = tf.Variable(tf.constant(1e-4, shape=[self.OUTPUT], dtype=tf.float32), trainable=True)
        fc8 = tf.nn.xw_plus_b(fc7, kernel, biases)
        self.variables += [kernel, biases]

        return fc8


class VGGNET(_Base):
    """ VGGNet model structrue """

    def __init__(self, function, learning_rate, difficulty):
        super(VGGNET, self).__init__('VGG', function, learning_rate, difficulty)

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
        kernel = tf.Variable(tf.truncated_normal([3, 3, 3, self.param['vggnet_conv1_kernel']], dtype=tf.float32, mean=1e-1, stddev=5e-1))
        conv = tf.nn.conv2d(image, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(1e-4, shape=[self.param['vggnet_conv1_kernel']], dtype=tf.float32), trainable=True)
        bias = tf.nn.bias_add(conv, biases)
        conv11 = tf.nn.relu(bias)
        self.variables += [kernel, biases]

        kernel = tf.Variable(tf.truncated_normal([3, 3, self.param['vggnet_conv1_kernel'], self.param['vggnet_conv1_kernel']], dtype=tf.float32, mean=1e-1, stddev=5e-1))
        conv = tf.nn.conv2d(conv11, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(1e-4, shape=[self.param['vggnet_conv1_kernel']], dtype=tf.float32), trainable=True)
        bias = tf.nn.bias_add(conv, biases)
        conv12 = tf.nn.relu(bias)
        self.variables += [kernel, biases]

        # pool1
        pool1 = tf.nn.max_pool(conv12, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # conv2
        kernel = tf.Variable(tf.truncated_normal([3, 3, self.param['vggnet_conv1_kernel'], self.param['vggnet_conv2_kernel']], dtype=tf.float32, mean=1e-1, stddev=5e-1))
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(1e-4, shape=[self.param['vggnet_conv2_kernel']], dtype=tf.float32), trainable=True)
        bias = tf.nn.bias_add(conv, biases)
        conv21 = tf.nn.relu(bias)
        self.variables += [kernel, biases]

        kernel = tf.Variable(tf.truncated_normal([3, 3, self.param['vggnet_conv2_kernel'], self.param['vggnet_conv2_kernel']], dtype=tf.float32, mean=1e-1, stddev=5e-1))
        conv = tf.nn.conv2d(conv21, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(1e-4, shape=[self.param['vggnet_conv2_kernel']], dtype=tf.float32), trainable=True)
        bias = tf.nn.bias_add(conv, biases)
        conv22 = tf.nn.relu(bias)
        self.variables += [kernel, biases]

        # pool2
        pool2 = tf.nn.max_pool(conv22,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='VALID')

        # conv3
        kernel = tf.Variable(tf.truncated_normal([3, 3, self.param['vggnet_conv2_kernel'], self.param['vggnet_conv3_kernel']], dtype=tf.float32, mean=1e-1, stddev=5e-1))
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(1e-4, shape=[self.param['vggnet_conv3_kernel']], dtype=tf.float32), trainable=True)
        bias = tf.nn.bias_add(conv, biases)
        conv31 = tf.nn.relu(bias)
        self.variables += [kernel, biases]

        kernel = tf.Variable(tf.truncated_normal([3, 3, self.param['vggnet_conv3_kernel'], self.param['vggnet_conv3_kernel']], dtype=tf.float32, mean=1e-1, stddev=5e-1))
        conv = tf.nn.conv2d(conv31, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(1e-4, shape=[self.param['vggnet_conv3_kernel']], dtype=tf.float32), trainable=True)
        bias = tf.nn.bias_add(conv, biases)
        conv32 = tf.nn.relu(bias)
        self.variables += [kernel, biases]

        kernel = tf.Variable(tf.truncated_normal([3, 3, self.param['vggnet_conv3_kernel'], self.param['vggnet_conv3_kernel']], dtype=tf.float32, mean=1e-1, stddev=5e-1))
        conv = tf.nn.conv2d(conv32, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(1e-4, shape=[self.param['vggnet_conv3_kernel']], dtype=tf.float32), trainable=True)
        bias = tf.nn.bias_add(conv, biases)
        conv33 = tf.nn.relu(bias)
        self.variables += [kernel, biases]

        # pool3
        pool3 = tf.nn.max_pool(conv33, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # conv4
        kernel = tf.Variable(tf.truncated_normal([3, 3, self.param['vggnet_conv3_kernel'], self.param['vggnet_conv4_kernel']], dtype=tf.float32, mean=1e-1, stddev=5e-1))
        conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(1e-4, shape=[self.param['vggnet_conv4_kernel']], dtype=tf.float32), trainable=True)
        bias = tf.nn.bias_add(conv, biases)
        conv41 = tf.nn.relu(bias)
        self.variables += [kernel, biases]

        kernel = tf.Variable(tf.truncated_normal([3, 3, self.param['vggnet_conv4_kernel'], self.param['vggnet_conv4_kernel']], dtype=tf.float32, mean=1e-1, stddev=5e-1))
        conv = tf.nn.conv2d(conv41, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(1e-4, shape=[self.param['vggnet_conv4_kernel']], dtype=tf.float32), trainable=True)
        bias = tf.nn.bias_add(conv, biases)
        conv42 = tf.nn.relu(bias)
        self.variables += [kernel, biases]

        kernel = tf.Variable(tf.truncated_normal([3, 3, self.param['vggnet_conv4_kernel'], self.param['vggnet_conv4_kernel']], dtype=tf.float32, mean=1e-1, stddev=5e-1))
        conv = tf.nn.conv2d(conv42, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(1e-4, shape=[self.param['vggnet_conv4_kernel']], dtype=tf.float32), trainable=True)
        bias = tf.nn.bias_add(conv, biases)
        conv43 = tf.nn.relu(bias)
        self.variables += [kernel, biases]

        # pool4
        pool4 = tf.nn.max_pool(conv43, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # conv5
        kernel = tf.Variable(tf.truncated_normal([3, 3, self.param['vggnet_conv4_kernel'], self.param['vggnet_conv5_kernel']], dtype=tf.float32, mean=1e-1, stddev=5e-1))
        conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(1e-4, shape=[self.param['vggnet_conv5_kernel']], dtype=tf.float32), trainable=True)
        bias = tf.nn.bias_add(conv, biases)
        conv51 = tf.nn.relu(bias)
        self.variables += [kernel, biases]

        kernel = tf.Variable(tf.truncated_normal([3, 3, self.param['vggnet_conv5_kernel'], self.param['vggnet_conv5_kernel']], dtype=tf.float32, mean=1e-1, stddev=5e-1))
        conv = tf.nn.conv2d(conv51, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(1e-4, shape=[self.param['vggnet_conv5_kernel']], dtype=tf.float32), trainable=True)
        bias = tf.nn.bias_add(conv, biases)
        conv52 = tf.nn.relu(bias)
        self.variables += [kernel, biases]

        kernel = tf.Variable(tf.truncated_normal([3, 3, self.param['vggnet_conv5_kernel'], self.param['vggnet_conv5_kernel']], dtype=tf.float32, mean=1e-1, stddev=5e-1))
        conv = tf.nn.conv2d(conv52, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(1e-4, shape=[self.param['vggnet_conv5_kernel']], dtype=tf.float32), trainable=True)
        bias = tf.nn.bias_add(conv, biases)
        conv53 = tf.nn.relu(bias)
        self.variables += [kernel, biases]

        # pool5
        pool5 = tf.nn.max_pool(conv53, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # fullyconnected6
        kernel = tf.Variable(tf.constant(0.0, shape=[self.param['vggnet_fc6_kernel1'], self.param['vggnet_fc6_kernel2']], dtype=tf.float32), trainable=True)
        biases = tf.Variable(tf.constant(1e-4, shape=[self.param['vggnet_fc6_kernel2']], dtype=tf.float32), trainable=True)
        fc6 = tf.nn.relu_layer(tf.reshape(pool5, [-1, int(np.prod(pool5.get_shape()[1:]))]), kernel, biases)
        self.variables += [kernel, biases]
        
        # fullyconnected7
        kernel = tf.Variable(tf.constant(0.0, shape=[self.param['vggnet_fc6_kernel2'], self.param['vggnet_fc7_kernel']], dtype=tf.float32), trainable=True)
        biases = tf.Variable(tf.constant(1e-4, shape=[self.param['vggnet_fc7_kernel']], dtype=tf.float32), trainable=True)
        fc7 = tf.nn.relu_layer(fc6, kernel, biases)
        self.variables += [kernel, biases]

        # fullyconnected8
        kernel = tf.Variable(tf.constant(0.0, shape=[self.param['vggnet_fc7_kernel'], self.OUTPUT], dtype=tf.float32), trainable=True)
        biases = tf.Variable(tf.constant(1e-4, shape=[self.OUTPUT], dtype=tf.float32), trainable=True)
        fc8 = tf.nn.xw_plus_b(fc7, kernel, biases)
        self.variables += [kernel, biases]

        return fc8

class LENET(_Base):
    """ LeNet model structrue """

    def __init__(self, function, learning_rate, difficulty):
        super(LENET, self).__init__('LE', function, learning_rate, difficulty)

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
        kernel = tf.Variable(tf.truncated_normal([5, 5, 1, self.param['lenet_conv1_kernel']], dtype=tf.float32, stddev=0.1), trainable=True)
        conv = tf.nn.conv2d(image, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.1, shape=[self.param['lenet_conv1_kernel']], dtype=tf.float32), trainable=True)
        conv1 = tf.nn.relu(conv + biases)
        self.variables += [kernel, biases]

        # pool1
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # conv2
        kernel = tf.Variable(tf.truncated_normal([5, 5, self.param['lenet_conv1_kernel'], self.param['lenet_conv2_kernel']], dtype=tf.float32, stddev=0.1), trainable=True)
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.1, shape=[self.param['lenet_conv2_kernel']], dtype=tf.float32), trainable=True)
        conv2 = tf.nn.relu(conv + biases)
        self.variables += [kernel, biases]

        # pool2
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
 
        # fullyconnected3
        kernel = tf.Variable(tf.truncated_normal([self.param['lenet_fc3_kernel1'], self.param['lenet_fc3_kernel2']], dtype=tf.float32, stddev=0.1), trainable=True)
        biases = tf.Variable(tf.constant(0.1, shape=[self.param['lenet_fc3_kernel2']], dtype=tf.float32), trainable=True)
        fc3 = tf.nn.relu(tf.nn.xw_plus_b(tf.reshape(pool2, [-1, int(np.prod(pool2.get_shape()[1:]))]), kernel, biases))
        self.variables += [kernel, biases]

        # fullyconnected4
        kernel = tf.Variable(tf.truncated_normal([self.param['lenet_fc3_kernel2'], self.OUTPUT], dtype=tf.float32, stddev=0.1), trainable=True)
        biases = tf.Variable(tf.constant(0.1, shape=[self.OUTPUT], dtype=tf.float32), trainable=True)
        fc4 = tf.nn.xw_plus_b(fc3, kernel, biases)
        self.variables += [kernel, biases]

        return fc4
