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


class PointSchema(OrderedSchema):
    type = fields.Str(
        required=True,
        validate=OneOf(
            ['Point'],
            error='Invalid point type'
        )
    )

    coordinates = fields.Tuple((fields.Float(), fields.Float()), required=True)


class PolygonSchema(OrderedSchema):
    type = fields.Str(
        required=True,
        validate=OneOf(
            ['Polygon'],
            error='Invalid polygon type'
        )
    )

    coordinates = fields.List(
        fields.List(
            fields.Tuple((fields.Float(), fields.Float()), required=True),
            required=True
        ),
        required=True
    )


class MultiPolygonSchema(OrderedSchema):
    type = fields.Str(
        required=True,
        validate=OneOf(
            ['MultiPolygon'],
            error='Invalid multi polygon type'
        )
    )

    coordinates = fields.List(
        fields.List(
            fields.List(
                fields.Tuple((fields.Float(),
                              fields.Float()),
                             required=True),
                required=True
            ),
            required=True,
        ),
        required=True
    )


class BboxSchema(OrderedSchema):
    bbox = fields.List(
        fields.Float(required=True),
        required=True
    )


class SpatialLabelSchema(OrderedSchema):
    """Schema for spatial labels"""
    segment = fields.List(
        fields.Dict(
            keys=fields.Str(required=True,
                            validate=OneOf(['value', 't', 'c'])),
            value=fields.Int()
        )
    )

    coordinate = fields.Dict(
        keys=fields.Int(),
        values=fields.Dict(
            keys=fields.Int(),
            values=fields.Nested(PointSchema)
        )
    )

    bbox = fields.Dict(
        keys=fields.Int(),
        values=fields.Dict(
            keys=fields.Int(),
            values=fields.List(
                fields.Float(
                    required=True),
                required=True)
        )
    )

    polygon = fields.Dict(
        keys=fields.Int(),
        values=fields.Nested(MultiPolygonSchema),
        description='Polygon'
    )


"""
Node schemas
"""


class NodeSchema(OrderedSchema):
    """Schema for nodes in scene graph"""

    ID = fields.Integer(description='Node ID')
    type = fields.Str(
        required=True,
        validate=OneOf(
            ['cell', 'debris', 'ftu']
        )
    )


class CompartmentSchema(OrderedSchema):
    """Schema for compartments that contain spatial labels"""

    ID = fields.Integer(description='Compartment ID')
    type = fields.Str(
        required=True,
        validate=OneOf(
            ['nuclei', 'whole-cell']
        )
    )
    spatial_label = fields.Nested(SpatialLabelSchema())


class CellSchema(NodeSchema):
    """ Fields specific to cell data entries"""
    frames = fields.List(fields.Int(), description='Frames this \
        cell appears in')
    spatial_label = fields.List(fields.Nested(CompartmentSchema()),
                                description='Spatial labels')


class DebrisSchema(NodeSchema):
    """Fields specific to debris entry"""

    spatial_label = fields.Nested(SpatialLabelSchema())


class FunctionalTissueUnitSchema(NodeSchema):
    """Fields specific to functional tissue units"""

    spatial_label = fields.Nested((SpatialLabelSchema()))


"""
Edge schemas
"""


class CellDivisionSchema(OrderedSchema):
    """Fields specific to cell divisions"""

    ID = fields.Integer(description='Edge ID')
    parent_id = fields.Integer(description='Parent ID')
    child_id = fields.List(fields.Integer(), description='Child IDs')
    frame = fields.Str(description='Frame at which division occured')


class LineageSchema(OrderedSchema):
    """Fields specific to object lineage"""

    ID = fields.Integer(description='Edge ID')
    parent_id = fields.Integer(description='Parent ID')
    child_id = fields.List(fields.Integer(), description='Child IDs')
