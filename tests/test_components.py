"""
Copyright 2023 Impulse Innovations Limited

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import unittest

from cai_causal_graph import EdgeType, NodeVariableType
from cai_causal_graph.graph_components import Edge, Node


class TestGraphComponents(unittest.TestCase):
    def test_serialization(self):
        source = Node(identifier='a', meta={'color': 'blue'}, variable_type=NodeVariableType.BINARY)
        destination = Node(identifier='b', meta={'color': 'green'}, variable_type=NodeVariableType.CONTINUOUS)
        edge = Edge(
            source=source, destination=destination, edge_type=EdgeType.UNKNOWN_DIRECTED_EDGE, meta={'color': 'red'}
        )
        edge_dict = edge.to_dict()
        self.assertDictEqual(edge_dict['source'], source.to_dict())
        self.assertDictEqual(edge_dict['destination'], destination.to_dict())
