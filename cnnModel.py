import tensorflow as tf
import numpy as np
from PIL import Image

class CnnModel(object):

    def __init__(self, particle_size, model_input_size, num_class):
        self.particle_size = particle_size
        self.batch_size = model_input_size[0]
        self.num_col = model_input_size[1]
        self.num_row = model_input_size[2]
        self.num_channel = model_input_size[3]
        self.num_class = num_class

    def init_learning_rate(self, learning_rate = 0.01, learning_rate_decay_factor = 0.95, decay_steps = 400, staircase = True):
        self.learning_rate = learning_rate
        self.learning_rate_decay_factor = learning_rate_decay_factor
        self.decay_steps = decay_steps
        self.staircase = staircase
        # define a global step variable
        self.global_step = tf.Variable(0, trainable = False)

    def init_momentum(self, momentum = 0.9):
        self.momentum = momentum

    def __variable_with_weight_decay(self, name, shape, stddev, wd):
        var = tf.get_variable(name, shape, initializer= tf.truncated_normal_initializer(stddev=stddev, seed = 123))
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('lossed', weight_decay)
        return var

    def __inference(self, data, train=True):
        conv1 = tf.nn.conv2d(data, self.kernel1, strides=[1, 1, 1, 1], padding='VALID')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, self.bias1))
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        conv2 = tf.nn.conv2d(pool1, self.kernel2, strides=[1, 1, 1, 1], padding='VALID')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, self.bias2))
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        conv3 = tf.nn.conv2d(pool2, self.kernel3, strides=[1, 1, 1, 1], padding='VALID')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, self.bias3))
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        conv4 = tf.nn.conv2d(pool3, self.kernel4, strides=[1, 1, 1, 1], padding='VALID')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, self.bias4))
        pool4 = tf.nn.max_pool(relu4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        hidden = tf.reshape(pool4, [self.batch_size, -1])
        # print(hidden.get_shape())
        if train:
            hidden = tf.nn.dropout(hidden, 0.5, seed=6543)

        fc1 = tf.nn.relu(tf.matmul(hidden, self.weights_fc1) + self.bias_fc1)
        sotfmax = tf.add(tf.matmul(fc1, self.weights_fc2), self.bias_fc2)
        return (sotfmax)

    def __loss(self, logits):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.train_label_node, name ='cross_entropy_all')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)
        all_loss = tf.add_n(tf.get_collection('losses'), name='all_loss')
        return all_loss

    # def __preprosses_particle(self, batch_data):
    #     batch_data_shape = batch_data.get_shape().as_list()
    #     batch_data_list = tf.unstack(batch_data)
    #     for i in xrange(batch_data_shape[0]):
    #         image = Image.fromarray(batch_data_list[i].eval())
    #         random_degree = np.random.randint(0,360)
    #         rotated = Image.Image.rotate(image, random_degree)

    def init_model_graph_train(self):
        self.kernel1 = self.__variable_with_weight_decay('weights1', shape = [9, 9, 1, 8], stddev = 0.05, wd = 0.0)
        self.bias1 = tf.get_variable('bias1', [8], initializer=tf.constant_initializer(0.0))

        self.kernel2 = self.__variable_with_weight_decay('weights2', shape=[5, 5, 8, 16], stddev=0.05, wd = 0.0)
        self.bias2 = tf.get_variable('bias2', [16], initializer=tf.constant_initializer(0.0))

        self.kernel3 = self.__variable_with_weight_decay('weights3', shape=[3, 3, 16, 32], stddev=0.05, wd = 0.0)
        self.bias3 = tf.get_variable('bias3', [32], initializer=tf.constant_initializer(0.0))

        self.kernel4 = self.__variable_with_weight_decay('weights4', shape=[2, 2, 32, 64], stddev=0.05, wd = 0.0)
        self.bias4 = tf.get_variable('bias4', [64], initializer=tf.constant_initializer(0.0))

        dim = 64*2*2

        self.weights_fc1 = self.__variable_with_weight_decay('weights_fc1', shape=[dim, 128], stddev=0.05, wd=0.0005)
        self.bias_fc1 = tf.get_variable('bias_fc1', [128], initializer=tf.constant_initializer(0.0))

        self.weights_fc2 = self.__variable_with_weight_decay('weights_fc2', shape=[128, self.num_class], stddev=0.05, wd=0.0005)
        self.bias_fc2 = tf.get_variable('bias_fc2', [self.num_class], initializer=tf.constant_initializer(0.0))


        self.train_data_node = tf.placeholder(tf.float32,
                                          shape=(self.batch_size, self.num_col, self.num_row, self.num_channel))
        self.train_label_node = tf.placeholder(tf.int64, shape=(self.batch_size,))
        self.eval_data_node = tf.placeholder(tf.float32,
                                         shape=(self.batch_size, self.num_col, self.num_row, self.num_channel))#diff

        logits = self.__inference(self.train_data_node, train=True)

        self.train_prediction_operation = tf.nn.softmax(logits)
        self.loss_operation = self.__loss(logits)
        # define the learning rate decay during training
        self.learningRate_operation = tf.train.exponential_decay(self.learning_rate,
                                                                 self.global_step,
                                                                 self.decay_steps,
                                                                 self.learning_rate_decay_factor,
                                                                 staircase=self.staircase)
        # define the Optimizer
        self.optimizer_operation = tf.train.MomentumOptimizer(self.learningRate_operation, self.momentum).minimize(
            self.loss_operation, global_step=self.global_step)

        # define the evaluation procedure
        evaluation_logits = self.__inference(self.eval_data_node, train=False)
        self.evaluation_prediction_operation = tf.nn.softmax(evaluation_logits)

    def train_batch(self, batch_data, batch_label, sess):
         # do the computation
         # feed_dict = {self.train_data_node: batch_data, self.train_label_node: batch_label}
         _, loss_value, learning_rate, prediction = sess.run(
                       [self.optimizer_operation, self.loss_operation, self.learningRate_operation, self.train_prediction_operation],
                       feed_dict={self.train_data_node: batch_data, self.train_label_node: batch_label})
         return loss_value, learning_rate, prediction

    def evaluation(self, data, sess):
        data_size = data.shape[0]
        predictions = np.ndarray(shape=(data_size, self.num_class), dtype=np.float32)
        for begin in xrange(0, data_size, self.batch_size):
            end = begin + self.batch_size
            if end <= data_size:
                batch_data = data[begin:end, ...]
                predictions[begin:end,:] = sess.run(self.evaluation_prediction_operation, feed_dict={self.eval_data_node: batch_data})
            else:
                batch_data = data[-self.batch_size:, ...]
                batch_predictions = sess.run(self.evaluation_prediction_operation, feed_dict={self.eval_data_node: batch_data})
                predictions[begin:,:] = batch_predictions[begin - data_size:, :]

        return predictions

    def init_model_graph_evaluate(self):
        self.kernel1 = self.__variable_with_weight_decay('weights1', shape=[9, 9, 1, 8], stddev=0.05, wd = 0.0)
        self.bias1 = tf.get_variable('bias1', [8], initializer=tf.constant_initializer(0.0))

        self.kernel2 = self.__variable_with_weight_decay('weights2', shape=[5, 5, 8, 16], stddev=0.05, wd = 0.0)
        self.bias2 = tf.get_variable('bias2', [16], initializer=tf.constant_initializer(0.0))

        self.kernel3 = self.__variable_with_weight_decay('weights3', shape=[3, 3, 16, 32], stddev=0.05, wd = 0.0)
        self.bias3 = tf.get_variable('bias3', [32], initializer=tf.constant_initializer(0.0))

        self.kernel4 = self.__variable_with_weight_decay('weights4', shape=[2, 2, 32, 64], stddev=0.05, wd = 0.0)
        self.bias4 = tf.get_variable('bias4', [64], initializer=tf.constant_initializer(0.0))

        dim = 64 * 2 * 2

        self.weights_fc1 = self.__variable_with_weight_decay('weights_fc1', shape=[dim, 128], stddev=0.05, wd = 0.0005)
        self.bias_fc1 = tf.get_variable('bias_fc1', [128], initializer=tf.constant_initializer(0.0))

        self.weights_fc2 = self.__variable_with_weight_decay('weights_fc2', shape=[128, self.num_class], stddev=0.05, wd = 0.0005)
        self.bias_fc2 = tf.get_variable('bias_fc2', [self.num_class], initializer=tf.constant_initializer(0.0))

        self.eval_data_node = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_col, self.num_row, self.num_channel))

        evaluation_logits = self.__inference(self.eval_data_node, train=False)
        self.evaluation_prediction_operation = tf.nn.softmax(evaluation_logits)