import mrcfile
import os
import re


class MrcDataLoader(object):
    # def __init__(self):

    @staticmethod
    def load_Particle_From_mrcFileDir(trainInputDir, particle_size, model_input_size, coordinate_symbol, mrc_number,
                                      produce_negative=True, negative_distance_ratio=0.5):
        """Read the particles data from mrc file dir.

        Based on the coordinates information and the corresponding mrc file information,
        extarct the particle patch when given the particle size.
        At the same time, select some negative particles randomly.
        The coordinates of the negative particles are enforced to be away from positive particles,
        the threshold is set to negative_distance_ratio*particle_size.

        Args:
            trainInputDir: string, the dir of mrc files as well as the coordinate files.
            particle_size: int, the size of the particle.
            model_input_size: the size of Placeholder to fit the model input, like [100, 64, 64, 1]
            coordinate_symbol: symbol of the coordinate file like '_manual'.
            mrc_number: number of mrc files to be used.
            produce_negative: bool, whether to produce the negative particles.
            negative_distance_ratio: float, a value between 0~1. It stands for the minimum distance between a positive sample
                                     and negative sample compared to the particle_size.
        Returns:
            return particle_array_positive,particle_array_negative
            particle_array_positive: numpy.array, a 4-dim array,the shape is (number_particles, particle_size, particle_size, 1).
            particle_array_negative: numpy.array, a 4-dim array,the shape is (number_particles, particle_size, particle_size, 1).

        Raises:
            None
        """
        if not os.path.isdir(trainInputDir):
            print("Invalid directory:", trainInputDir)

        files = os.listdir(trainInputDir)
        mrc_file_all = []
        valid_mrc_file = []
        coordinate_file = []
        for f in files:
            if re.search('\.mrc$', f):
                file_name = os.path.join(trainInputDir, f)
                mrc_file_all.append(file_name)

        mrc_file_all.sort()
        for i in range(len(mrc_file_all)):
            filename_mrc = mrc_file_all[i]
            filename_coordinate = filename_mrc.replace('.mrc', coordinate_symbol + '.star')
            if os.path.isfile(filename_coordinate):
                valid_mrc_file.append(mrc_file_all[i])
                coordinate_file.append(filename_coordinate)

        if mrc_number <= 0 | mrc_number > len(valid_mrc_file):
            mrc_number = len(valid_mrc_file)

        positive_particles = []
        negative_particles = []



    @staticmethod
    def load_trainData_From_mrcFileDir(train_inputDir, particle_size, model_input_size, validation_ratio, coordinate_symbol, mrc_number, positive_particle_number):
        """read train_data and validation data from a directory of mrc files

        Train a CNN model based on mrc files and corresponding coordinates.

        Args:
            trainInputDir: the directory of mrc files
            particle_size: particle size
            model_input_size: the size of Placeholder to fit the model input, like [100, 64, 64, 1]
            validation_rate: divide the total samples into training dataset and validation dataset.
                             This is the ratio of validation dataset compared to the total samples.
            mrc_number: number of mrc files to be used.
            train_number: number of positive particles to be used for training.

        Returns:
            return train_data,train_labels,validation_data,validation_labels
            train_data: numpy.array, np.float32, the shape is (number_samples, particle_size, particle_size, 1)
            train_labels: numpy.array, int64, the shape is (number_samples)
            validation_data: numpy.array, np.float32, the shape is (number_samples, particle_size, particle_size, 1)
            validation_labels: numpy.array, int64, the shape is (number_samples)

        Raises:
            None
        """
        particle_array_positive, particle_array_negative = MrcDataLoader.load_Particle_From_mrcFileDir(train_inputDir, particle_size, model_input_size, mrc_number)


        return particle_array_positive, particle_array_negative

