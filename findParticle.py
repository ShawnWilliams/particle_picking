from optparse import OptionParser
import os
import re
import time
import tensorflow as tf
import numpy as np
from cnnModel import CnnModel
from particleFinder import ParticleFinder

def find_particle():
    parser = OptionParser()
    parser.add_option("--inputDir", dest="inputDir", help="Input directory", metavar="DIRECTORY")
    parser.add_option("--trained_model", dest="trained_model", help="Input the trained model", metavar="FILE")
    parser.add_option("--mrc_number", dest="mrc_number", help="Number of mrc files to be picked.", metavar="VALUE", default=-1)
    parser.add_option("--particle_size", dest="particle_size", help="the size of the particle.", metavar="VALUE", default=-1)
    parser.add_option("--outputDir", dest="outputDir", help="Output directory, the coordinates file will be saved here.", metavar="DIRECTORY")
    parser.add_option("--coordinate_symbol", dest="coordinate_symbol", help="The symbol of the saveed coordinate file, like '_cnnPick'", metavar="STRING")
    parser.add_option("--threshold", dest="threshold", help="Pick the particles, the prediction value is larger than the threshold..", metavar="VALUE", default=0.5)
    (opt, args) = parser.parse_args()

    tf.set_random_seed(123)
    np.random.seed(123)

    model_input_size = [100, 64, 64, 1]
    num_class = 2
    batch_size = model_input_size[0]

    particle_size = int(opt.particle_size)

    trained_model = opt.trained_model
    input_dir = opt.inputDir
    output_dir = opt.outputDir
    threshold = float(opt.threshold)
    coordinate_symbol = opt.coordinate_symbol
    mrc_number = int(opt.mrc_number)

    if not os.path.isfile(trained_model):
        print("Error: %s is not a valid model file." %trained_model)

    if not os.path.isdir(input_dir):
        print("Error: %s is not a valid input directory." %input_dir)

    if not os.path.isdir(output_dir):
        print("Error: %s is not a valid output directory." %output_dir)

    cnnModel = CnnModel(particle_size, model_input_size, num_class)
    cnnModel.init_model_graph_evaluate()

    mrc_file_all = []
    files = os.listdir(input_dir)
    for f in files:
        if re.search('\.mrc$', f):
            filename = os.path.join(input_dir, f)
            mrc_file_all.append(filename)

    mrc_file_all.sort()
    if mrc_number <= 0 or mrc_number > len(mrc_file_all):
        mrc_number = len(mrc_file_all)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, trained_model)

        particleFinder = ParticleFinder(sess, model_input_size, cnnModel, particle_size)
        start_time = time.time()
        candidate_particle_all = []
        for i in range(mrc_number):
            coordinate = particleFinder.pick(mrc_file_all[i])


def main():
    find_particle()

if __name__ == '__main__':
    main()