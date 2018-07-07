import numpy as np
from mrcDataLoader import MrcDataLoader
import mrcfile


class ParticleFinder(object):
    def __init__(self, sess, model_input_size, cnnModel, particle_size):
        """Initialize the Autopicker.

        Args:
            sess: an instance of tensorflow session.
            model_input_size: a list of length 4, it is the input size of a placeholder of tensorflow.
            cnnModel: an instance of class cnnModel
            particle_size: the particle size of the molecular

        """
        self.sess = sess
        self.model_input_size = model_input_size
        self.cnnModel = cnnModel
        self.particle_size = particle_size


    def pick(self, mrc_filename):
        """Do the picking job through tensorflow.

        This function read the micrograph data information based on the given filename of micrograph.
        Then do the picking based on trained CNN model.

        Args:
            mrc_filename: string, the filename of the target micrograph.

        Returns:
            return list_coordinate
            list_coordinate: a list, the length of this list stands for the number of picked particles.
                                Each element in the list is also a list, the length is 4, the first one is y-axis,
                                the second one is x-axis, the third one is the predicted score, the fourth is the micrograph filename.
        """

        print(mrc_filename)
        mrc = mrcfile.open(mrc_filename, mode='r+')

        MrcDataLoader.preprocess_particle(mrc.data)
        body_2d, bin_size = MrcDataLoader.preprocess_mrcFile(mrc.data)
        num_col = int(mrc.data.shape[0]/bin_size)
        num_row = int(mrc.data.shape[1]/bin_size)







        mrc.close()