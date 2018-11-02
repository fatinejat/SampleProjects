from _codecs import latin_1_encode
import numpy as np
import OwnITK as oitk


def point_is_valid(min_dimensions, max_dimensions, x, y, z):
    if x < min_dimensions[0] or y < min_dimensions[1] or z < min_dimensions[2] or x >= max_dimensions[0] or y >= \
            max_dimensions[1] or z >= max_dimensions[2]:
        return False
    return True


def get_valid_neighbours(min_dimensions, max_dimensions, x, y, z):
    result = list()
    if point_is_valid(min_dimensions, max_dimensions, x, y, z + 1):
        result.append([x, y, z + 1])
    if point_is_valid(min_dimensions, max_dimensions, x, y, z - 1):
        result.append([x, y, z - 1])
    if point_is_valid(min_dimensions, max_dimensions, x, y + 1, z):
        result.append([x, y + 1, z])
    if point_is_valid(min_dimensions, max_dimensions, x, y - 1, z):
        result.append([x, y - 1, z])
    if point_is_valid(min_dimensions, max_dimensions, x + 1, y, z):
        result.append([x + 1, y, z])
    if point_is_valid(min_dimensions, max_dimensions, x - 1, y, z):
        result.append([x - 1, y, z])
    return result


def get_connected_component(image, min_dimensions, max_dimensions, x, y, z, threshold=0.1):
    tmp_array = np.zeros((max_dimensions[0] - min_dimensions[0], max_dimensions[1] - min_dimensions[1],
                          max_dimensions[2] - min_dimensions[2]))
    tmp_array[x-min_dimensions[0]][y-min_dimensions[1]][z-min_dimensions[2]] = 1
    bfs_array = [[x, y, z]]
    for bfs_point in bfs_array:
        for neighbour in get_valid_neighbours(min_dimensions, max_dimensions, bfs_point[0], bfs_point[1], bfs_point[2]):
            if tmp_array[neighbour[0] - min_dimensions[0]][neighbour[1] - min_dimensions[1]][
                        neighbour[2] - min_dimensions[2]] == 0:
                if np.abs(image[neighbour[0]][neighbour[1]][neighbour[2]] - image[x][y][z]) < threshold:
                    tmp_array[neighbour[0] - min_dimensions[0]][neighbour[1] - min_dimensions[1]][
                        neighbour[2] - min_dimensions[2]] = 2
                    bfs_array.append(neighbour)
                else:
                    tmp_array[neighbour[0] - min_dimensions[0]][neighbour[1] - min_dimensions[1]][
                        neighbour[2] - min_dimensions[2]] = 1

    for pixel in bfs_array:
        if tmp_array[pixel[0]-min_dimensions[0]][pixel[1]-min_dimensions[1]][pixel[2]-min_dimensions[2]] != 2:
            bfs_array.remove(pixel)
    return bfs_array


# image, dimensions, no_need = oitk.getItkImageData('Data/VSD.Brain.XX.O.MR_Flair.2.nii')
# get_connected_component(image, [0, 0, 0], [30, 30, 30], 0, 0, 0, 0.00000001)
