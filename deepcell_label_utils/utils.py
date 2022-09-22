# Copyright 2016-2022 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/deepcell-label-utils/LICENSE
#
# The Work provided may be used for non-commercial academic purposes only.
# For any other use of the Work, including commercial use, please contact:
# vanvalenlab@gmail.com
#
# Neither the name of Caltech nor the names of its contributors may be used
# to endorse or promote products derived from this software without specific
# prior written permission.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Label utils"""

# import json
# import geojson
import numpy as np
import cv2

from shapely.geometry import MultiPolygon, Polygon, Point  # , mapping
from collections import defaultdict
from skimage.measure import regionprops


def mask_to_polygons(mask, epsilon=1e-3, min_area=10., approx=True):
    """Convert a mask ndarray (binarized image) to Multipolygons"""

    # first, find contours with cv2: it's much faster than shapely
    contours, hierarchy = cv2.findContours(mask,
                                           cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_SIMPLE)

    if contours and approx:
        contours = [cv2.approxPolyDP(cnt,
                                     epsilon * cv2.arcLength(cnt, True), True)
                    for cnt in contours]

    if not contours:
        return MultiPolygon()

    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1

    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(contours[idx])

    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly)
    all_polygons = MultiPolygon(all_polygons)

    return all_polygons


def polygons_to_mask(polygons, im_size):
    """Convert a polygon or multipolygon list back to
       an image mask ndarray"""
    img_mask = np.zeros(im_size, np.uint8)
    if not polygons:
        return img_mask

    # function to round and convert to int
    def int_coords(x):
        return np.array(x).round().astype(np.int32)

    exteriors = [int_coords(poly.exterior.coords) for poly in polygons.geoms]
    interiors = [int_coords(pi.coords) for poly in polygons.geoms
                 for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask


class SpatialLabelConverter(object):
    """Label converter class for DeepCell Label label format. Converts
    DCL labels into binary masks, centroids, bboxes, and polygons"""

    def __init__(self, X, y, segments, test_no_poly=False):
        self.X = X
        self.y = y
        self.segments = segments

        # Get list of segments
        object_ids = self.get_object_ids()
        labels = {}

        # Iterate over all segments and convert to the new label format
        for object_id in object_ids:
            seg = self.segments_to_dict(object_id)
            mask = self.dcl_to_binary_mask(object_id)
            centroid = self.binary_mask_to_centroid(mask)
            bbox = self.binary_mask_to_bbox(mask)
            if not test_no_poly:
                polygon = self.binary_mask_to_polygon(mask)

            # Save converted labels to dictionary
            labels[object_id] = {'segment': seg,
                                 'coordinate': centroid,
                                 'bbox': bbox}
            if not test_no_poly:
                labels[object_id]['polygon'] = polygon

        self.labels = labels

    def get_object_ids(self):
        return list(set(self.segments['cell']))

    def segments_to_dict(self, object_id):
        object_info = self.segments.loc[self.segments['cell'] == object_id]
        list_of_dicts = []

        for i in range(object_info.shape[0]):
            value = object_info.iloc[i]['value']
            time = object_info.iloc[i]['t']
            channel = object_info.iloc[i]['c']
            list_of_dicts.append({'value': value, 't': time, 'c': channel})

        return list_of_dicts

    def dcl_to_binary_mask(self, object_id):
        object_info = self.segments.loc[self.segments['cell'] == object_id]
        binary_mask = np.zeros(self.y.shape, dtype=self.y.dtype)

        for i in range(object_info.shape[0]):
            value = object_info.iloc[i]['value']
            time = object_info.iloc[i]['t']
            channel = object_info.iloc[i]['c']
            binary_mask[time, ..., channel] = np.where(
                self.y[time, ..., channel] == value,
                1, binary_mask[time, ..., channel])

        return binary_mask

    def binary_mask_to_centroid(self, mask):
        centroids = {}
        for t in range(mask.shape[0]):
            channels = {}
            for c in range(mask.shape[3]):
                mt = mask[t, ..., c]
                if np.sum(mt.flatten()) > 0:
                    prop = regionprops(mt)[0]
                    centroid = Point(prop.centroid[0], prop.centroid[1])
                    channels[c] = centroid
            centroids[t] = channels

        return centroids

    def binary_mask_to_bbox(self, mask):
        bboxes = {}
        for t in range(mask.shape[0]):
            channels = {}
            for c in range(mask.shape[3]):
                mt = mask[t, ..., c]
                if np.sum(mt.flatten()) > 0:
                    prop = regionprops(mt)[0]
                    bbox = list(prop.bbox)
                    channels[c] = bbox
            bboxes[t] = channels

        return bboxes

    def binary_mask_to_polygon(self, mask):
        polygons = {}
        for t in range(mask.shape[0]):
            channels = {}
            for c in range(mask.shape[3]):
                mt = mask[t, ..., c]
                if np.sum(mt.flatten()) > 0:
                    poly = mask_to_polygons(mt.astype('uint8'))
                    channels[c] = poly
            polygons[t] = channels

        return polygons
