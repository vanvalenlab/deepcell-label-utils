from utils import SpatialLabelConverter
from shapely.geometry import Point

import numpy as np
import pandas as pd


def test_get_object_ids():
    '''Test that the list of objects is obtained correctly.'''

    # Simple example
    X = np.array([[[[0], [0]], [[0], [0]]]])
    y = np.array([[[[0], [0]], [[0], [0]]]])
    segments = pd.DataFrame([{'cell': 1, 'value': 1, 'c': 0, 't': 0},
                             {'cell': 2, 'value': 2, 'c': 0, 't': 0}])
    SLC = SpatialLabelConverter(X, y, segments)
    assert SLC.get_object_ids() == [1, 2]

    # Test cases where a cell maps to multiple values
    segments = pd.DataFrame([{'cell': 1, 'value': 1, 'c': 0, 't': 0},
                             {'cell': 1, 'value': 2, 'c': 0, 't': 0}])
    SLC = SpatialLabelConverter(X, y, segments)
    assert SLC.get_object_ids() == [1]


def test_segments_to_dict():
    '''
    Test that a segment dataframe is correctly converted
    into list of dictionaries.
    '''
    X = np.array([[[[0], [0]], [[0], [0]]]])
    y = np.array([[[[0], [0]], [[0], [0]]]])
    segments = pd.DataFrame([{'cell': 1, 'value': 1, 'c': 0, 't': 0},
                             {'cell': 2, 'value': 2, 'c': 0, 't': 0}])
    SLC = SpatialLabelConverter(X, y, segments)
    assert SLC.labels[1]['segment'] == [{'value': 1, 'c': 0, 't': 0}]
    assert SLC.labels[2]['segment'] == [{'value': 2, 'c': 0, 't': 0}]


def test_dcl_to_binary_mask():
    '''
    Test that binary masks are generated correctly
    '''

    # Set up overlapping segments (cells 1 and 3)
    X = np.array([[[[0], [0]], [[0], [0]]]])
    y = np.array([[[[1], [2]], [[3], [4]]]])
    segments = pd.DataFrame([{'cell': 1, 'value': 1, 'c': 0, 't': 0},
                             {'cell': 2, 'value': 2, 'c': 0, 't': 0},
                             {'cell': 3, 'value': 3, 'c': 0, 't': 0},
                             {'cell': 1, 'value': 4, 'c': 0, 't': 0},
                             {'cell': 3, 'value': 4, 'c': 0, 't': 0}])
    SLC = SpatialLabelConverter(X, y, segments)

    # Binary masks 1 and 3 should both include bottom right pixel
    assert(np.array_equal(SLC.dcl_to_binary_mask(1),
                          np.array([[[[1], [0]], [[0], [1]]]])))
    assert(np.array_equal(SLC.dcl_to_binary_mask(2),
                          np.array([[[[0], [1]], [[0], [0]]]])))
    assert(np.array_equal(SLC.dcl_to_binary_mask(3),
                          np.array([[[[0], [0]], [[1], [1]]]])))

    # Test case where num_channels > 1
    X = np.array([[[[0, 0], [0, 0]], [[0, 0], [0, 0]]]])
    y = np.array([[[[1, 1], [2, 1]], [[3, 1], [4, 1]]]])
    segments = pd.DataFrame([{'cell': 1, 'value': 1, 'c': 0, 't': 0},
                             {'cell': 2, 'value': 2, 'c': 0, 't': 0},
                             {'cell': 3, 'value': 3, 'c': 0, 't': 0},
                             {'cell': 1, 'value': 4, 'c': 0, 't': 0},
                             {'cell': 3, 'value': 4, 'c': 0, 't': 0},
                             {'cell': 1, 'value': 1, 'c': 1, 't': 0}])
    SLC = SpatialLabelConverter(X, y, segments)
    assert(np.array_equal(SLC.dcl_to_binary_mask(1),
                          np.array([[[[1, 1], [0, 1]], [[0, 1], [1, 1]]]])))
    assert(np.array_equal(SLC.dcl_to_binary_mask(2),
                          np.array([[[[0, 0], [1, 0]], [[0, 0], [0, 0]]]])))
    assert(np.array_equal(SLC.dcl_to_binary_mask(3),
                          np.array([[[[0, 0], [0, 0]], [[1, 0], [1, 0]]]])))


def test_binary_mask_to_centroid():
    '''Test that centroids stored are correctly converted.'''
    for i in range(2, 102, 2):
        y = np.ones((1, i, i, 1), dtype=np.int8)
        halved = i // 2
        y[:, :, halved:i, :] = 2
        X = np.zeros((1, i, i, 1))
        segments = pd.DataFrame([{'cell': 1, 'value': 1, 'c': 0, 't': 0},
                                 {'cell': 2, 'value': 2, 'c': 0, 't': 0}])
        SLC = SpatialLabelConverter(X, y, segments)
        OFFSET = 0.5  # 1,1 coordinate for skimage starts at 0.5, 0.5
        assert(SLC.labels[1]['coordinate'][0][0] == Point(i / 2 - OFFSET,
                                                          i / 4 - OFFSET))
        assert(SLC.labels[2]['coordinate'][0][0] == Point(i / 2 - OFFSET,
                                                          3 * i / 4 - OFFSET))

    # Test case where num_channels > 1
    i = 100
    y = np.ones((1, i, i, 2), dtype=np.int8)
    y[:, :, i//2:i, 0] = 2
    y[:, :, i//2:i, 1] = 3
    X = np.zeros((1, i, i, 2))
    segments = pd.DataFrame([{'cell': 1, 'value': 1, 'c': 0, 't': 0},
                             {'cell': 2, 'value': 2, 'c': 0, 't': 0},
                             {'cell': 1, 'value': 1, 'c': 1, 't': 0},
                             {'cell': 3, 'value': 3, 'c': 1, 't': 0}])
    SLC = SpatialLabelConverter(X, y, segments)
    OFFSET = 0.5  # 1,1 coordinate for skimage starts at 0.5, 0.5
    assert(SLC.labels[1]['coordinate'][0][0] == Point(i / 2 - OFFSET,
                                                      i / 4 - OFFSET))
    assert(SLC.labels[2]['coordinate'][0][0] == Point(i / 2 - OFFSET,
                                                      3 * i / 4 - OFFSET))
    assert(SLC.labels[1]['coordinate'][0][1] == Point(i / 2 - OFFSET,
                                                      i / 4 - OFFSET))
    assert(SLC.labels[3]['coordinate'][0][1] == Point(i / 2 - OFFSET,
                                                      3 * i / 4 - OFFSET))


def test_binary_mask_to_bbox():
    '''Test that bboxes are correctly converted from masks.'''
    y = np.array([[[[1], [1], [1]], [[1], [1], [1]], [[2], [2], [2]]]])
    X = np.zeros(y.shape)
    segments = pd.DataFrame([{'cell': 1, 'value': 1, 'c': 0, 't': 0},
                             {'cell': 2, 'value': 2, 'c': 0, 't': 0}])
    SLC = SpatialLabelConverter(X, y, segments)
    assert(SLC.labels[1]['bbox'][0][0] == [0, 0, 2, 3])
    assert(SLC.labels[2]['bbox'][0][0] == [2, 0, 3, 3])

    # Test the num_channels > 1 case
    y = np.array([[[[1, 0], [1, 0], [1, 0]], [[1, 0],
                 [1, 0], [1, 0]], [[1, 0], [1, 0], [1, 1]]]])
    X = np.zeros(y.shape)
    segments = pd.DataFrame([{'cell': 1, 'value': 1, 'c': 0, 't': 0},
                             {'cell': 1, 'value': 1, 'c': 1, 't': 0}])
    SLC = SpatialLabelConverter(X, y, segments)
    assert(SLC.labels[1]['bbox'][0][0] == [0, 0, 3, 3])
    assert(SLC.labels[1]['bbox'][0][1] == [2, 2, 3, 3])

# def test_binary_mask_to_polygon():
#     '''Test that polygons are correctly converted from masks.'''
#     y = np.array([[[[1], [1], [1]], [[1], [1], [1]],
#                  [[2], [2], [2]], [[2], [2], [2]]]])
#     X = np.zeros(y.shape)
#     segments = pd.DataFrame([{'cell': 1, 'value': 1, 'c': 0, 't': 0},
#                              {'cell': 2, 'value': 2, 'c': 0, 't': 0}])
#     SLC = SpatialLabelConverter(X, y, segments)
#     print(SLC.labels[1]['polygon'][0][0])

#     # # Test the num_channels > 1 case
#     # y = np.array([[[[1, 0], [1, 0], [1, 0]],
#                    [[1, 0], [1, 0], [1, 0]], [[1, 0], [1, 0], [1, 1]]]])
#     # X = np.zeros(y.shape)
#     # segments = pd.DataFrame([{'cell': 1, 'value': 1, 'c': 0, 't': 0},
#     #                          {'cell': 1, 'value': 1, 'c': 1, 't': 0}])
#     # SLC = SpatialLabelConverter(X, y, segments)
#     # assert(SLC.labels[1]['bbox'][0][0] == [0, 0, 3, 3])
#     # assert(SLC.labels[1]['bbox'][0][1] == [2, 2, 3, 3])

if __name__ == '__main__':
    test_get_object_ids()
    test_segments_to_dict()
    test_dcl_to_binary_mask()
    test_binary_mask_to_centroid()
    test_binary_mask_to_bbox()
    # test_binary_mask_to_polygon()
