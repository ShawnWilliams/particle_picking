import numpy as np
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from mrcDataLoader import MrcDataLoader
import mrcfile
import time
import math
import os
import pickle

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

    def peak_detection(self, image_2D, local_window_size):
        """Do the local peak dection to get the best coordinate of molecular center.

        This function does a local peak dection to the score map to get the best coordinates.

        Args:
            image_2d: numpy.array, it is a 2d array, the dim is 2, the value of it was a prediction score given by the CNN model.
            local_window_size: this is the distance threshold between two particles. The peak detection is done in the local window.

        Returns:
            return list_coordinate_clean
            list_coordinate_clean: a list, the length of this list stands for the number of picked particles.
                                   Each element in the list is also a list, the length is 3.
                                   The first one is x-axis, the second one is y-axis, the third one is the predicted score.
        """
        col = image_2D.shape[0]
        row = image_2D.shape[1]
        # filter the array in local, the values are replaced by local max value.
        data_max = filters.maximum_filter(image_2D, local_window_size)
        # compare the filter array to the original one, the same value in the same location is the local maximum.
        # maxima is a bool 2D array, true stands for the local maximum
        maxima = (image_2D == data_max)
        data_min = filters.minimum_filter(image_2D, local_window_size)
        diff = ((data_max - data_min) > 0)
        maxima[diff == 0] = 0

        labeled, num_objects = ndimage.label(maxima)
        # get the coordinate of the local maximum
        # the shape of the array_y_x is (number, 2)
        array_y_x = np.array(ndimage.center_of_mass(image_2D, labeled, range(1, num_objects + 1)))
        array_y_x = array_y_x.astype(int)
        list_y_x = array_y_x.tolist()
        # print("number of local maximum:%d"%len(list_y_x))
        for i in range(len(list_y_x)):
            # add the prediction score to the list
            list_y_x[i].append(image_2D[array_y_x[i][0]][array_y_x[i][1]])
            # add a symbol to the list, and it is used to remove crowded candidate
            list_y_x[i].append(0)

        # remove close candidate
        for i in range(len(list_y_x) - 1):
            if list_y_x[i][3] == 1:
                continue

            for j in range(i + 1, len(list_y_x)):
                if list_y_x[i][3] == 1:
                    break
                if list_y_x[j][3] == 1:
                    continue
                d_y = list_y_x[i][0] - list_y_x[j][0]
                d_x = list_y_x[i][1] - list_y_x[j][1]
                d_distance = math.sqrt(d_y ** 2 + d_x ** 2)
                if d_distance < local_window_size / 2:
                    if list_y_x[i][2] >= list_y_x[j][2]:
                        list_y_x[j][3] = 1
                    else:
                        list_y_x[i][3] = 1

        list_coordinate_clean = []
        for i in range(len(list_y_x)):
            if list_y_x[i][3] == 0:
                # remove the symbol element
                list_x_y = []
                list_x_y.append(list_y_x[i][1])
                list_x_y.append(list_y_x[i][0])
                list_x_y.append(list_y_x[i][2])
                list_coordinate_clean.append(list_x_y)

        return list_coordinate_clean

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

        MrcDataLoader.preprocess_particle(mrc.data, self.model_input_size)
        body_2d, bin_size = MrcDataLoader.preprocess_mrcFile(mrc.data)


        num_col = int(mrc.data.shape[0]/bin_size)
        num_row = int(mrc.data.shape[1]/bin_size)

        step_size = 4
        num_total_patch = 0
        patch_size = int(self.particle_size/bin_size)

        # the size to do peak detection
        local_window_size = int(0.6*patch_size/step_size)

        map_col = int((body_2d.shape[0]-patch_size)/step_size)
        map_row = int((body_2d.shape[1]-patch_size)/step_size)

        start_time = time.time()
        particle_candidate_all = []
        map_index_col = 0
        for col in range(0, body_2d.shape[0] - patch_size + 1, step_size):
            for row in range(0, body_2d.shape[1] - patch_size + 1, step_size):
                # extract the particle patch
                patch = np.copy(body_2d[col:(col + patch_size), row:(row + patch_size)])
                # do preprocess to the particle
                patch = MrcDataLoader.preprocess_particle(patch, self.model_input_size)
                particle_candidate_all.append(patch)
                num_total_patch = num_total_patch + 1
            map_index_col = map_index_col + 1

        map_index_row = map_index_col - map_col + map_row
        particle_candidate_all = np.array(particle_candidate_all).reshape(num_total_patch, self.model_input_size[1], self.model_input_size[2], 1)

        # predict
        predictions = self.cnnModel.evaluation(particle_candidate_all, self.sess)
        predictions = predictions[:, 1:2]
        predictions = predictions.reshape(map_index_col, map_index_row)

        time_cost = time.time() - start_time
        print("time cost: %d s"%time_cost)

        # do a local peak detection to get the best coordinate
        # list_coordinate is a 2D list of shape (number_particle, 3)
        # element in list_coordinate is [x_coordinate, y_coordinate, prediction_value]
        list_coordinate = self.peak_detection(predictions, local_window_size)

        for i in range(len(list_coordinate)):
            list_coordinate[i].append(mrc_filename)
            # transform the coordinates to the original size
            list_coordinate[i][0] = (list_coordinate[i][0]*step_size+patch_size/2)*bin_size
            list_coordinate[i][1] = (list_coordinate[i][1]*step_size+patch_size/2)*bin_size

        #can move towards
        mrc.close()

        return list_coordinate

    @staticmethod
    def write_coordinate(coordinate, mrc_filename, coordinate_symbol, threshold, output_dir):
        """Write the picking results in the Relion '.star' format.

        This function selects the particles based on the given threshold and saves these particles in Relion '.star' file.

        Args:
            coordinate: a list, all the coordinates in it are come from the same micrograph.
                        The length of the list stands for the number of the particles.
                        And each element in the list is a small list of length of 3 at least.
                        The first element in the small list is the coordinate x-aixs.
                        The second element in the small list is the coordinate y-aixs.
                        The third element in the small list is the prediction score.
                        The fourth element in the small list is the micrograph name.
            mrc_filename: string, the corresponding micrograph file.
            coordinate_symbol: the symbol is used in the output star file name, like '_manualPick', '_cnnPick'.
            threshold: particles over the threshold are stored, a default value is 0.5.
            output_dir: the directory to store the coordinate file.
        """
        mrc_basename = os.path.basename(mrc_filename)
        print(mrc_basename)
        coordinate_name = os.path.join(output_dir, mrc_basename[:-4] + coordinate_symbol + ".star")
        print(coordinate_name)
        f = open(coordinate_name, 'w')
        f.write('data_\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n')
        for i in range(len(coordinate)):
            if coordinate[i][2] > threshold:
                f.write(str(coordinate[i][0]) + ' ' + str(coordinate[i][1]) + '\n')

        f.close()

    @staticmethod
    def write_pick_results(coordinate, output_file):
        """Write the picking results in a file of binary format.

        This function writes the coordinates of all micrographs into a binary file.

        Args:
            coordinate: a list, the length of it stands for the number of picked micrograph file.
                        Each element is a list too, which contains all coordinates from the same micrograph.
                        The length of the list stands for the number of the particles.
                        And each element in the list is a small list of length of 4.
                        The first element in the small list is the coordinate x-aixs.
                        The second element in the small list is the coordinate y-aixs.
                        The third element in the small list is the prediction score.
                        The fourth element in the small list is the micrograh name.
            output_file: string, the output file.
        """
        with open(output_file, 'wb') as f:
            pickle.dump(coordinate, f)