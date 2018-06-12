import mrcfile
import os
import re
import numpy as np
import scipy.ndimage
import scipy.misc
from starReader import starRead
from pylab import *


class MrcDataLoader(object):
    # def __init__(self):

    @staticmethod
    def bin_2d(body_2d, bin_size):
        """Do the bin process to the 2D array.

        This function can make bin the image based on the bin_size.
        bin_size is a int value. if it was set to 2, then the 4 points in a small patch 2x2 of the body_2d
               are summed to one value. It likes an average pooling operation.

        Args:
            body_2d: numpy.array, it is a 2d array, the dim is 2.
            bin_size: int value.

        Returns:
            return pool_result
            pool_result: numpy.array, the shape of it is (body_2d.shape[0]/bin_size, body_2d.shape[1]/bin_size)

        """
        # based on the numpy operation to do the bin process
        col = body_2d.shape[0]
        row = body_2d.shape[1]
        scale_col = col // bin_size
        scale_row = row // bin_size
        patch = np.copy(body_2d[0:scale_col * bin_size, 0:scale_row * bin_size])
        patch_view = patch.reshape(scale_col, bin_size, scale_row, bin_size)
        body_2d_bin = patch_view.mean(axis=3).mean(axis=1)
        return body_2d_bin

    @staticmethod
    def preprocess_mrcFile(micrograph):
        """Do preprocess to the micrograph after the micrograph data is loaded into a numpy.array.

        Define this function to make sure that the same process is done to the micrograph
            during the training process and picking process.

        Args:
            micrograph: numpy.array, the shape is (micrograph_col, micrograph_row)

        Returns:
            return micrograph
            micrograph: numpy.array
        """
        # mrc_col = micrograph.shape[0]
        # mrc_row = micrograph.shape[1]
        # lowpass
        micrograph = scipy.ndimage.filters.gaussian_filter(micrograph, 0.1)
        # do the bin process
        pooling_size = 3
        micrograph = MrcDataLoader.bin_2d(micrograph, pooling_size)

        # low pass the micrograph
        # micrograph_lowpass = scipy.ndimage.filters.gaussian_filter(micrograph, 0.1)
        # f = np.fft.fft2(micrograph)
        # fshift = np.fft.fftshift(f)
        # magnitude_spectrum = 20*np.log(np.abs(fshift))

        # plt.subplot(121),plt.imshow(micrograph, cmap = 'gray')
        # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        # plt.subplot(122),plt.imshow(micrograph_lowpass, cmap = 'gray')
        # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
        # plt.show()

        # nomalize the patch
        # max_value = micrograph.max()
        # min_value = micrograph.min()
        # particle = (micrograph - min_value) / (max_value - min_value)
        mean_value = micrograph.mean()
        std_value = micrograph.std()
        micrograph = (micrograph - mean_value) / std_value
        #
        return micrograph, pooling_size

    @staticmethod
    def read_coordinate_from_star(starfile):
        """ Read the coordinate from star file.
        return a list

        Args:
            starfile: string, the input coodinate star file.

        Returns:
            return coordinate_list
            coordinate_list: list, the length of the list stands for the number of particles.
                             Each particle is a list too, which contains two elements.
                             The first one is the x coordinate, the second one is y coordinate.
        """
        particle_star = starRead(starfile)
        table_star = particle_star.getByName('data_')
        coordinateX_list = table_star.getByName('_rlnCoordinateX')
        coordinateY_list = table_star.getByName('_rlnCoordinateY')
        coordinate_list = []
        for i in range(len(coordinateX_list)):
            coordinate = []
            coordinate.append(int(float(coordinateX_list[i])))
            coordinate.append(int(float(coordinateY_list[i])))
            coordinate_list.append(coordinate)
        return coordinate_list

    @staticmethod
    def preprocess_particle(particle, model_input_size):
        """Do preprocess to the particle patch after the particle data is extracted from the micrograph.

        Define this function to make sure that the same process is done to the particle
                    during the training process and picking process.

        Args:
            particle: numpy.array, the shape is (particle_col, particle_row)
            model_input_size: a list with length 4. The size is to fit with the model input.
                                      model_input_size[0] stands for the batchsize.
                                      model_input_size[1] stands for the input col.
                                      model_input_size[2] stands for the input row.
                                      model_input_size[3] stands for the input channel.
        Returns:
            return particle
            particle: numpy.array
        """
        particle = scipy.misc.imresize(particle, (model_input_size[1], model_input_size[2]), interp='bilinear', mode = 'L')
        mean_value = particle.mean()
        std_value = particle.std()
        particle = (particle - mean_value)/std_value
        return particle

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
        positive_particle_number_sum = 0
        for i in range(mrc_number):
            print(valid_mrc_file[i])
            mrc = mrcfile.open(valid_mrc_file[i], mode='r+')
            body_2d, bin_size = MrcDataLoader.preprocess_mrcFile(mrc.data)

            coordinate = MrcDataLoader.read_coordinate_from_star(coordinate_file[i])
            particle_size = int(particle_size // bin_size)
            # print type(mrc.header)
            sha = mrc.data.shape
            n_col = int(sha[0]/bin_size)
            n_row = int(sha[1]/bin_size)
            for i in range(len(coordinate)):
                coordinate[i][0] //= bin_size
                coordinate[i][1] //= bin_size
            radius = int(particle_size/2)
            i = 0
            while i < len(coordinate):
                coord_x = coordinate[i][0]
                coord_y = coordinate[i][1]
                if coord_x < radius or coord_x + radius > n_row or coord_y < radius or coord_y + radius > n_col:
                    coordinate.pop(i)
                else:
                    # extract positive particles
                    coord_x = coordinate[i][0]
                    coord_y = coordinate[i][1]
                    valid_positive_particle = body_2d[coord_x - radius : coord_x + radius, coord_y - radius : coord_y + radius]
                    valid_positive_particle = MrcDataLoader.preprocess_particle(valid_positive_particle, model_input_size)
                    positive_particles.append(valid_positive_particle)
                    i += 1

            positive_particle_number_sum += len(coordinate)
            print 'number of positive particles:', positive_particle_number_sum

            if produce_negative:
                for i in range(len(coordinate)):
                    while True:
                        isLegal = True
                        coord_x = np.random.randint(radius, n_row - radius)
                        coord_y = np.random.randint(radius, n_col - radius)
                        for j in range(len(coordinate)):
                            distance = ((coord_x - coordinate[j][0]) ** 2 + (coord_y - coordinate[j][1]) ** 2) ** 0.5
                            if distance < negative_distance_ratio * particle_size:
                                isLegal = False
                                break
                        if isLegal:
                            valid_negative_particle = body_2d[coord_x - radius: coord_x + radius,
                                                      coord_y - radius: coord_y + radius]
                            valid_negative_particle = MrcDataLoader.preprocess_particle(valid_negative_particle,
                                                                                        model_input_size)
                            negative_particles.append(valid_negative_particle)
                            break

            #close the mrc file
            mrc.close()

        if produce_negative:
            return positive_particles, negative_particles
        else:
            return positive_particles
            # for i in range(coordinate):
            #     coord_x = coordinate[i]


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
        particle_array_positive, particle_array_negative = MrcDataLoader.load_Particle_From_mrcFileDir(train_inputDir, particle_size, model_input_size, coordinate_symbol, mrc_number)

        #todo: we can just extract positive_particle_number particles rather than extract len(particle_array_positive) particles than throw some part.
        if positive_particle_number > 0 & positive_particle_number < len(particle_array_positive):
            particle_array_positive = particle_array_positive[:positive_particle_number]
            particle_array_negative = particle_array_negative[:positive_particle_number]



        validation_size = int(validation_ratio * len(particle_array_positive))
        validation_data = particle_array_positive[:validation_size, ...]
        validation_data = concatenate((validation_data, particle_array_negative[:validation_size, ...]))
        validation_labels = concatenate((ones(validation_size, dtype=int64), zeros(validation_size, dtype=int64)))

        train_size = positive_particle_number - validation_size
        train_data = particle_array_positive[validation_size:, ...]
        train_labels = concatenate((ones(train_size, dtype=int64), zeros(train_size, dtype=int64)))

        return train_data, train_labels, validation_data, validation_labels

