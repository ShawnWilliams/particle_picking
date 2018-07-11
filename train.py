from __future__ import division

import os.path
import numpy as np
import tensorflow as tf
import time

from optparse import OptionParser
from mrcDataLoader import MrcDataLoader
from cnnModel import CnnModel

def shuffle_related_data(data, label):
    assert len(data) == len(label)
    shuffle_index = np.random.permutation(len(data))
    return data[shuffle_index], label[shuffle_index]

def resize_data(data):
    # for i in range(len(data)):
    #     data[i] = np.reshape(data[i], (64, 64, 1))
    data = np.reshape(data,(len(data), 64, 64, 1))
    return data


def train():
    parser = OptionParser()
    parser.add_option("--train_inputDir", dest="train_inputDir", help="Input Directory", metavar="DIRECTORY")
    parser.add_option("--train_inputFile", dest="train_input_file", help="Input File", metavar="FILE")
    parser.add_option("--mrc_number", dest="mrc_number", help="Number of mrc files to be trained.", metavar="VALUE", default=-1)
    parser.add_option("--positive_particle_number", dest="positive_particle_number", help="Number of positive samples to train.", metavar="VALUE", default=-1)
    parser.add_option("--coordinate_symbol", dest="coordinate_symbol", help="The symbol of the coordinate file, like '_manualPick'", metavar="STRING")
    parser.add_option("--particle_size", dest="particle_size", help="the size of the particle.", metavar="VALUE", default=-1)
    parser.add_option("--validation_ratio", dest="validation_ratio", help="the ratio.", metavar="VALUE", default=0.2)
    parser.add_option("--model_save_dir", dest="model_save_dir", help="save the model to this directory", metavar="DIRECTORY", default="./trained_model")
    parser.add_option("--model_save_file", dest="model_save_file", help="save the model to file", metavar="FILE")
    (opt, args) = parser.parse_args()

    #load mrc data
    train_inputDir = opt.train_inputDir
    train_input_file = opt.train_input_file
    mrc_number = int(opt.mrc_number)
    positive_particle_number = int(opt.positive_particle_number)
    coordinate_symbol = opt.coordinate_symbol
    particle_size = int(opt.particle_size)
    validation_ratio = float(opt.validation_ratio)
    # model_save_dir = opt.model_save_dir
    model_save_file = os.path.join(opt.model_save_dir, opt.model_save_file)

    # print(train_inputDir)
    # print(type(train_inputDir))
    #
    # print(train_input_file)
    # print(type(train_input_file))
    model_input_size = [100, 64, 64, 1]

    # dataLoader = MrcDataLoader()
    train_data, train_label, eval_data, eval_label = MrcDataLoader.load_trainData_From_mrcFileDir(train_inputDir,
                                                                                                  particle_size, model_input_size, validation_ratio, coordinate_symbol, mrc_number, positive_particle_number)

    try:
        train_data
    except NameError:
        print("ERROR: in function load.loadInputTrainData.")
        return None
    else:
        print("Load training data successfully!")

    train_data, train_label = shuffle_related_data(train_data, train_label)
    eval_data, eval_label = shuffle_related_data(eval_data, eval_label)


    learning_rate = 0.01
    learning_rate_decay_factor = 0.95
    learning_rate_staircase = True
    momentum = 0.9
    train_size = len(train_data)
    eval_size = len(eval_data)

    train_data = resize_data(train_data)
    eval_data = resize_data(eval_data)

    batch_size = model_input_size[0]

    learning_rate_decay_steps = 10 * (train_size // batch_size)

    #positive particle and negative particle
    num_class = 2
    cnnModel = CnnModel(particle_size, model_input_size, num_class)
    batch_size = model_input_size[0]

    cnnModel.init_learning_rate(learning_rate = learning_rate, learning_rate_decay_factor = learning_rate_decay_factor,
                                decay_steps = learning_rate_decay_steps, staircase = learning_rate_staircase)

    cnnModel.init_momentum(momentum = momentum)

    cnnModel.init_model_graph_train()

    saver = tf.train.Saver(tf.all_variables())

    start_time = time.time()

    init = tf.initialize_all_variables()

    init_toleration_patience = 5
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        sess.run(init)
        max_epochs = 200
        best_eval_error_rate = 100
        toleration_patience = init_toleration_patience
        eval_frequency = train_size // batch_size   # the frequency to evaluate the evaluation dataset
        for step in xrange( int(max_epochs * train_size) // batch_size):
            offset =  (step * batch_size) % (train_size - batch_size)
            batch_data = train_data[offset:offset+batch_size, ...]
            batch_label = train_label[offset:offset+batch_size]
            loss_value, lr, train_prediction = cnnModel.train_batch(batch_data, batch_label, sess)

            if step % eval_frequency == 0:
                stop_time = time.time() - start_time
                start_time = time.time()
                eval_prediction = cnnModel.evaluation(eval_data, sess)
                eval_error_rate = error_rate(eval_prediction, eval_label)
                print('epoch: %.2f , %.2f ms' % (step * batch_size / train_size, 1000 * stop_time / eval_frequency))
                print('train loss: %.6f,\t learning rate: %.6f' % (loss_value, lr))
                print('train error: %.6f%%,\t valid error: %.6f%%' % (
                error_rate(train_prediction, batch_label), eval_error_rate))
                if eval_error_rate < best_eval_error_rate:
                    best_eval_error_rate = eval_error_rate
                    toleration_patience = init_toleration_patience
                else:
                    toleration_patience -= 1
            if toleration_patience == 0:
                saver.save(sess, model_save_file)
                break


def error_rate(prediction, label):
    return 100.0 - (100.0 * np.sum(np.argmax(prediction, 1) == label) / prediction.shape[0])

def main(argv=None):
    train()

if __name__ == '__main__':
    main()