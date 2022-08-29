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
"""Label schemas"""

from marshmallow import Schema, fields, EXCLUDE
from marshmallow import pre_load, validates, validates_schema, ValidationError
from marshmallow.validate import Range, Length, OneOf


class OrderedSchema(Schema):
    """Schema with ordering"""
    class Meta:
        ordered = True


"""
Spatial label schemas
"""


class SegmentSchema(OrderedSchema):
    """Schema for segments"""
    segments = fields.List(fields.Integer(), description='Segments')


class MaskSchema(OrderedSchema):
    """Schema for binary masks - expects COO formated sparse array"""
    coords = fields.List(fields.List(fields.Integer()),
                         description='Coordinate array')
    data = fields.List(fields.Integer(), description='Data array')


class CoordinateSchema(OrderedSchema):
    """Schema for 2D coordinate labels"""
    x = fields.Float(required=True)
    y = fields.Float(required=True)
    z = fields.Float()


class BboxSchema(OrderedSchema):
    """Schema for bounding box labels"""
    minr = fields.Float(required=True)
    minc = fields.Float(required=True)
    maxr = fields.Float(required=True)
    maxc = fields.Float(required=True)


class PolygonSchema(OrderedSchema):
    """Schema for polygon labels"""
    polygon = fields.List(fields.Tuple((fields.Float(), fields.Float())),
                          description='Polygon')


class SpatialSchema(OrderedSchema):
    """Schema for spatial labels"""

    segments = fields.List(fields.Int(), description='Segments')
    mask = fields.Dict(keys=fields.Int(), values=fields.Nested(MaskSchema))
    xy = fields.List(fields.Dict(keys=fields.Int(), values=fields.Nested(CoordinateSchema)))
    bbox = fields.Dict(keys=fields.Int(), values=fields.Nested(BboxSchema))
    polygon = fields.Dict(keys=fields.Int(), values=fields.Nested(PolygonSchema))


"""
Node schemas
"""


class CompartmentSchema(OrderedSchema):
    """Schema for compartments that contain spatial labels"""

    name = fields.Str()
    spatial_label = fields.Nested(SpatialLabelSchema())


class CellSchema(OrderedSchema):
    """ Fields specific to cell data entries"""

    ID = fields.Integer(description='Node ID')
    mapping = fields.List(description='Mapping to allow for vector embeddings for cells')
    spatial_label = fields.List(fields.Nested(CompartmentSchema()),
                                description='Spatial labels')


class DebrisSchema(OrderedSchema):
    """Fields specific to debris entry"""

    ID = fields.Integer(description='Node ID')
    spatial_label = fields.Nested(SpatialLabelSchema())


class FunctionalTissueUnitSchema(OrderedSchema):
    """Fields specific to functional tissue units"""

    ID = fields.Integer(description='Node ID')
    spatial_label = fields.Nested((SpatialLabelSchema()))


"""
Edge schemas
"""


class CellDivisionSchema(OrderedSchema):
    """Fields specific to cell divisions"""

    ID = fields.Integer(description='Edge ID')
    parent_id = fields.Integer(description='Parent ID')
    child_id = fields.List(description='Child IDs')
    frame = fields.Str(description='Frame at which division occured')


class LineageSchema(OrderedSchema):
    """Fields specific to object lineage"""

    ID = fields.Integer(description='Edge ID')