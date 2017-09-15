import math
import tensorflow as tf

def print_activations(t):
  print(t.op.name, ' ', t.get_shape().as_list())

class _Base(object):
    """ Base structrue """
    def __init__(self) :
        # Internal counter to keep track of current iteration
        self._iters = 0

        # Input
        # TODO: change target format 
        image_size = 224
        self.image = tf.Variable(tf.random_normal([FLAGS.batch_size,
                                                   image_size,
                                                   image_size, 3],
                                                   dtype=tf.float32,
                                                   stddev=1e-1))
        self.target = tf.placeholder(tf.float32, [None])

        # Call methods to initialize variables and operations 
        self._init_vars()
        self._init_ops()

        # RMSE
        # TODO: rmse format of classification is probably different 
        self.rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.target, self.pred))))

    def init_sess(self, sess):
        """ 
        Initializes tensorflow session 
        :param sess: tensorflow session to execute tensorflow operations
        """
        self.sess = sess
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def train_iteration(self, imagedata, targetdata):
        """ 
        Runs each train iteration
        """
        feed_dict = {self.image=imagedata, self.target=targetdata}

        self.sess.run(self.optimize_steps, feed_dict=feed_dict)

        self._iters += 1

    def eval_rmse(self, imagedata, targetdata):
        """ 
        Calculates RMSE 
        """
        feed_dict = {self.image=imagedata, self.target=targetdata}
        return self.sess.run([self.rmse, self.pred], feed_dict=feed_dict)

class ALEXNET(_Base):
    """ AlexNet model structrue """
    def __init__(self, *args, **kwargs):
        super(ALEXNET, self).__init__(*args, **kwargs)

    def _init_vars(self):
        """ Build layers of the model """

        _pred = self.build_layers(self.image)
        self.pred = tf.squeeze(_pred, squeeze_dims=[1])

    def _init_ops(self):
        """ Calculates loss and performs gradient descent """
        # Loss     
        loss = tf.reduce_sum(tf.square(tf.subtract(self.target, self.pred)), reduction_indices=[0])
                    
        self.loss = 

        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        # Optimize the weights
        self.optimize_steps= self.optimizer.minimize(self.loss, var_list=self.parameters)

    def eval_loss(self, imagedata, targetdata):
        """ 
        Calculates loss
        """
        feed_dict = {self.image: imagedata, self.target: targetdata}

        return self.sess.run(self.loss, feed_dict=feed_dict)

    def build_layers(image):
        """
        Builds layers 
        """
        self.parameters = []
        
        # conv1
        with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(image, kernel, [1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        print_activations(conv1)
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
        print_activations(pool1)

        # conv2
        with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        self.parameters += [kernel, biases]
        print_activations(conv2)

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
        print_activations(pool2)

        # conv3
        with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        self.parameters += [kernel, biases]
        print_activations(conv3)

        # conv4
        with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        self.parameters += [kernel, biases]
        print_activations(conv4)

        # conv5
        with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        self.parameters += [kernel, biases]
        print_activations(conv5)

        # pool5
        pool5 = tf.nn.max_pool(conv5,
                             ksize=[1, 3, 3, 1],
                             strides=[1, 2, 2, 1],
                             padding='VALID',
                             name='pool5')
        print_activations(pool5)

        return pool5


