import os.path
import numpy as np
import mrcDataLoader

from optparse import OptionParser
from mrcDataLoader import MrcDataLoader

def shuffle_related_data(data, label):
    assert len(data) == len(label)
    shuffle_index = np.random.permutation(len(data))
    return data[shuffle_index], label[shuffle_index]

def train():
    parser = OptionParser()
    parser.add_option("--train_inputDir", dest="train_inputDir", help="Input Directory", metavar="DIRECTORY")
    parser.add_option("--train_inputFile", dest="train_input_file", help="Input File", metavar="FILE")
    parser.add_option("--mrc_number", dest="mrc_number", help="Number of mrc files to be trained.", metavar="VALUE", default=-1)
    parser.add_option("--positive_particle_number", dest="positive_particle_number", help="Number of positive samples to train.", metavar="VALUE", default=-1)
    parser.add_option("--coordinate_symbol", dest="coordinate_symbol", help="The symbol of the coordinate file, like '_manualPick'", metavar="STRING")
    parser.add_option("--particle_size", dest="particle_size", help="the size of the particle.", metavar="VALUE", default=-1)
    parser.add_option("--validation_ratio", dest="validation_ratio", help="the ratio.", metavar="VALUE", default=0.1)
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
    model_save_dir = opt.model_save_dir
    model_save_file = opt.model_save_file

    # print(train_inputDir)
    # print(type(train_inputDir))
    #
    # print(train_input_file)
    # print(type(train_input_file))
    model_input_size = [100, 64, 64, 1]

    # dataLoader = MrcDataLoader()
    train_data, train_label, eval_data, eval_label = MrcDataLoader.load_trainData_From_mrcFileDir(train_inputDir, particle_size, model_input_size, validation_ratio, coordinate_symbol, mrc_number, positive_particle_number)

    try:
        train_data
    except NameError:
        print("ERROR: in function load.loadInputTrainData.")
        return None
    else:
        print("Load training data successfully!")

    train_data, train_label = shuffle_related_data(train_data, train_label)

def main(argv=None):
    train()

if __name__ == '__main__':
    main()