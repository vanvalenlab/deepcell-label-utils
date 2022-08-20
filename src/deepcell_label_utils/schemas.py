# Copyright 2016-2022 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/tf-keras-retinanet/LICENSE
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
"""Label schemas"""

from marshmallow import Schema, fields, EXCLUDE
from marshmallow import pre_load, validates, validates_schema, ValidationError
from marshmallow.validate import Range, Length, OneOf

"""
Node schemas
"""

class SpatialLabelSchema(Schema):
    """Schema for spatial labels"""
    class Meta:
        ordered = True

    segments = fields.List(description='Segments outlining id in DeepCell Label outputs')
    mask = fields.List(description='Binary mask annotation in scipy sparse format')
    xy = fields.List(description='Coordinate annotation')
    bbox = fields.List(description='Bounding box annotation')
    polygon = fields.List(description='Polygon annotation')


class CompartmentSchema(Schema):
    """Schema for compartments that contain spatial labels"""
    class Meta:
        ordered = True

    spatial_label = fields.Nested(SpatialLabelSchema())  


class GeneSchema(Schema):
    """Schema for gene expression"""
    class Meta:
        ordered = True


class CellSchema(Schema):
    """ Fields specific to cell data entries"""
    class Meta:
        ordered = True

    ID = fields.Str(description='Cell ID')
    mapping = fields.List(description='Mapping to allow for vector embeddings for cells')
    spatial_label = fields.List(fields.Nested(CompartmentSchema()),
                                description='Spatial labels')


class DebrisSchema(Schema):
    """Fields specific to debris entry"""
    class Meta:
        ordered = True

    ID = fields.Str(description='Debris ID')
    spatial_label = fields.Nested(SpatialLabelSchema())


class FunctionalTissueUnitSchema(Schema):
    """Fields specific to functional tissue units"""
    class Meta:
        ordered = True

    ID = fields.Str(description='FTU ID')
    spatial_label = fields.Nested((SpatialLabelSchema()))


"""
Edge schemas
"""

class CellDivisionSchema(Schema):
    """Fields specific to cell divisions"""
    class Meta:
        ordered = True

    ID = fields.Str(description='Division ID')
    parent_id = fields.Str(description='Parent ID')
    child_id = fields.List(description='Child IDs')
    frame = fields.Str(description='Frame at which division occured')