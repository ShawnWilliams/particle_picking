from __future__ import division
from optparse import OptionParser
from operator import itemgetter
from mrcDataLoader import MrcDataLoader
import os
import math
import pickle

def calculate_tp(coordinate_pick, coordinate_reference, threshold):
    if len(coordinate_pick) < 1 or len(coordinate_reference) < 1:
        print("Invalid coordinate parameters in function calculate_tp()!")

    # add a symbol to index whether the coordinate is matched with a reference coordinate
    for i in range(len(coordinate_pick)):
        coordinate_pick[i].append(0)

    tp = 0
    average_distance = 0

    for i in range(len(coordinate_reference)):
        coordinate_reference[i].append(0)
        coor_x = coordinate_reference[i][0]
        coor_y = coordinate_reference[i][1]
        neighbour = []
        for k in range(len(coordinate_pick)):
            if coordinate_pick[k][4] == 0:
                coor_mx = coordinate_pick[k][0]
                coor_my = coordinate_pick[k][1]
                abs_x = math.fabs(coor_mx - coor_x)
                abs_y = math.fabs(coor_my - coor_y)
                length = math.sqrt(math.pow(abs_x, 2) + math.pow(abs_y, 2))
                if length < threshold:
                    same_n = []
                    same_n.append(k)
                    same_n.append(length)
                    neighbour.append(same_n)
        if len(neighbour) >= 1:
            if len(neighbour) > 1:
                neighbour = sorted(neighbour, key=itemgetter(1))
            index = neighbour[0][0]
            # change the symbol to 1, means it matchs with a reference coordinate
            coordinate_pick[index][4] = 1
            # add the distance to the list
            coordinate_pick[index].append(neighbour[0][1])
            coordinate_pick[index].append(coor_x)
            coordinate_pick[index].append(coor_y)
            tp = tp + 1
            average_distance = average_distance + neighbour[0][1]
            coordinate_reference[i][2] = 1
    average_distance = average_distance / tp
    return tp, average_distance

def analysis_pick_results(pick_results_file, reference_coordinate_dir, reference_coordinate_symbol, particle_size,
                          minimum_distance_rate):
    """Load the picking results from a file of binary format and compare it with the reference coordinate.

    This function analysis the picking results with reference coordinate and calculate the recall, precision and the deviation from the center.

    Args:
        pick_results_file: string, the file name of the pre-picked results.
        reference_mrc_dir: string, the directory of the mrc file dir.
        reference_coordinate_symbol: the symbol of the coordinate, like '_manualpick'
        particle_size: int, the size of particle
        minimum_distance_rate: float, the default is 0.2, a picked coordinate is considered to be a true positive only when the distance between the picked coordinate and the reference coordinate is less than minimum_distance_rate mutiplicate particle_size.
    """
    with open(pick_results_file, 'rb') as f:
        coordinate = pickle.load(f)
        """
        coordinate: a list, the length of it stands for the number of picked micrograph file.
                    Each element is a list too, which contains all coordinates from the same micrograph. 
                    The length of the list stands for the number of the particles.
                    And each element in the list is a small list of length of 4.
                    The first element in the small list is the coordinate x-aixs. 
                    The second element in the small list is the coordinate y-aixs. 
                    The third element in the small list is the prediction score. 
                    The fourth element in the small list is the micrograh name. 
        """
    tp = 0
    total_pick = 0
    total_reference = 0
    coordinate_total = []
    threshold = 0.98
    for i in range(len(coordinate)):
        mrc_filename = os.path.basename(coordinate[i][0][3])
        # print(mrc_filename)
        reference_coordinate_file = mrc_filename.replace('.mrc', reference_coordinate_symbol + '.star')
        reference_coordinate_file = os.path.join(reference_coordinate_dir, reference_coordinate_file)
        # print(reference_coordinate_file)
        if os.path.isfile(reference_coordinate_file):
            reference_coordinate = MrcDataLoader.read_coordinate_from_star(reference_coordinate_file)
            """
            reference_coordinate: a list, the length of it stands for the number of picked particles.
                        And each element in the list is a small list of length of 2.
                        The first element in the small list is the coordinate x-aixs. 
                        The second element in the small list is the coordinate y-aixs. 
            """
            tp_sigle, average_distance = calculate_tp(coordinate[i], reference_coordinate,
                                                                 particle_size * minimum_distance_rate)
            # print("tp:",tp_sigle)
            # print("average_distance:",average_distance)
            # calculate the number of true positive, when the threshold is set to 0.5
            tp_sigle = 0
            total_reference = total_reference + len(reference_coordinate)
            for j in range(len(coordinate[i])):
                coordinate_total.append(coordinate[i][j])
                # if coordinate[i][j][2] > 0.5:
                if coordinate[i][j][2] > threshold:
                    total_pick = total_pick + 1
                    if coordinate[i][j][4] == 1:
                        tp = tp + 1
                        tp_sigle = tp_sigle + 1
            #print( float(tp_sigle) / len(reference_coordinate))
        else:
            print("Can not find the reference coordinate:" + reference_coordinate_file)
    precision = float(tp) / total_pick
    recall = float(tp) / total_reference
    print("(threshold %f)precision:%f recall:%f" % (threshold, precision, recall))
    # sort the coordinate based on prediction score in a descending order.
    coordinate_total = sorted(coordinate_total, key=itemgetter(2), reverse=True)
    total_tp = []
    total_recall = []
    total_precision = []
    total_probability = []
    total_average_distance = []
    total_distance = 0
    tp_tem = 0
    for i in range(len(coordinate_total)):
        if coordinate_total[i][4] == 1:
            tp_tem = tp_tem + 1
            total_distance = total_distance + coordinate_total[i][5]
        precision = tp_tem / (i + 1)
        recall = tp_tem / total_reference
        total_tp.append(tp_tem)
        total_recall.append(recall)
        total_precision.append(precision)
        total_probability.append(coordinate_total[i][2])
        if tp_tem == 0:
            average_distance = 0
        else:
            average_distance = total_distance / tp_tem
        total_average_distance.append(average_distance)
    # write the list results in file
    directory_pick = os.path.dirname(pick_results_file)
    total_results_file = os.path.join(directory_pick, 'results.txt')
    f = open(total_results_file, 'w')
    # write total_tp
    f.write(','.join(map(str, total_tp)) + '\n')
    f.write(','.join(map(str, total_recall)) + '\n')
    f.write(','.join(map(str, total_precision)) + '\n')
    f.write(','.join(map(str, total_probability)) + '\n')
    f.write(','.join(map(str, total_average_distance)) + '\n')
    f.write('#total autopick number:%d\n' % (len(coordinate_total)))
    f.write('#total manual pick number:%d\n' % (total_reference))
    f.write('#the first row is number of true positive\n')
    f.write('#the second row is recall\n')
    f.write('#the third row is precision\n')
    f.write('#the fourth row is probability\n')
    f.write('#the fiveth row is distance\n')

    # show the recall and precision
    times_of_manual = len(coordinate_total) // total_reference + 1
    for i in range(times_of_manual):
        print('autopick_total sort, take the head number of total_manualpick * ratio %d' % (i + 1))
        f.write('#autopick_total sort, take the head number of total_manualpick * ratio %d \n' % (i + 1))
        if i == times_of_manual - 1:
            print('precision:%f \trecall:%f' % (total_precision[-1], total_recall[-1]))
            f.write('precision:%f \trecall:%f \n' % (total_precision[-1], total_recall[-1]))
        else:
            print('precision:%f \trecall:%f' % (
            total_precision[(i + 1) * total_reference - 1], total_recall[(i + 1) * total_reference - 1]))
            f.write('precision:%f \trecall:%f \n' % (
            total_precision[(i + 1) * total_reference - 1], total_recall[(i + 1) * total_reference - 1]))
    f.close()


def result_analysis():
    parser = OptionParser()
    parser.add_option("--inputFile", dest="inputFile",
                      help="Input picking results file, like '/PATH/autopick_results.list'", metavar="FILE")
    parser.add_option("--inputDir", dest="inputDir", help="Reference coordinate directory", metavar="DIRECTORY")
    parser.add_option("--coordinate_symbol", dest="coordinate_symbol",
                      help="The symbol of the coordinate file, like '_manualPick'", metavar="STRING")
    parser.add_option("--particle_size", dest="particle_size", help="the size of the particle.", metavar="VALUE",
                      default=-1)
    parser.add_option("--minimum_distance_rate", dest="minimum_distance_rate",
                      help="Use the value particle_size*minimum_distance_rate as the distance threshold for estimate the number of true positive samples, the default value is 0.2",
                      metavar="VALUE", default=0.2)
    (opt, args) = parser.parse_args()

    pick_results_file = opt.inputFile
    reference_mrc_file_dir = opt.inputDir
    reference_coordinate_symbol = opt.coordinate_symbol
    particle_size = int(opt.particle_size)
    minimum_distance_rate = float(opt.minimum_distance_rate)
    analysis_pick_results(pick_results_file, reference_mrc_file_dir, reference_coordinate_symbol, particle_size, minimum_distance_rate)


def main(argv=None):
    result_analysis()


if __name__ == '__main__':
    main()