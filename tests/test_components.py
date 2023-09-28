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
from cai_causal_graph.graph_components import Edge, Node, TimeSeriesNode


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

        # test serialization with dicts
        edge1 = Edge.from_dict(edge_dict)
        self.assertEqual(edge, edge1)

        # test edge with mixed nodes
        source = Node(identifier='a', meta={'color': 'blue'}, variable_type=NodeVariableType.BINARY)
        destination = TimeSeriesNode(identifier='b', meta={'color': 'green'}, variable_type=NodeVariableType.CONTINUOUS)
        edge = Edge(
            source=source, destination=destination, edge_type=EdgeType.UNKNOWN_DIRECTED_EDGE, meta={'color': 'red'}
        )

        # check node class is preserved
        edge2 = Edge.from_dict(edge.to_dict())
        self.assertEqual(edge, edge2)

        # test node serialization-deserialization
        node = Node(identifier='a', meta={'color': 'blue'}, variable_type=NodeVariableType.BINARY)
        node1 = Node.from_dict(node.to_dict())

        self.assertTrue(node.__eq__(node1, deep=True))

        # test time series node serialization-deserialization
        node = TimeSeriesNode(identifier='a lag(n=2)', meta={'color': 'blue'}, variable_type=NodeVariableType.BINARY)
        node1 = TimeSeriesNode.from_dict(node.to_dict())

        self.assertTrue(node.__eq__(node1, deep=True))

        # test when dict and meta disagree
        node = TimeSeriesNode(
            identifier='a lag(n=2)',
            meta={'variable_name': 'blue', 'time_lag': -1},
            variable_type=NodeVariableType.BINARY,
        )
        # check that the meta is updated
        self.assertEqual(node.meta['variable_name'], 'a')
        self.assertEqual(node.meta['time_lag'], -2)

        # now modify again the meta
        node.meta['variable_name'] = 'b'
        node.meta['time_lag'] = 3

        # check it raises if meta is not consistent
        with self.assertRaises(AssertionError):
            node1 = TimeSeriesNode.from_dict(node.to_dict())

        # test N -> TSN
        node = Node(identifier='a', meta={'color': 'blue'}, variable_type=NodeVariableType.BINARY)
        node1 = TimeSeriesNode.from_dict(node.to_dict())

        self.assertFalse(node.__eq__(node1, deep=True))
        self.assertEqual(node.identifier, node1.identifier)
        self.assertEqual(node.variable_type, node1.variable_type)

        # check meta is equal in the part in common
        meta1 = node1.meta.copy()
        meta1.pop('variable_name')
        meta1.pop('time_lag')
        self.assertEqual(node.meta, meta1)

        # check it is not equal if we add the time lag and variable name
        self.assertNotEqual(node.meta, node1.meta)

        # test TSN -> N
        node = TimeSeriesNode(identifier='a lag(n=2)', meta={'color': 'blue'}, variable_type=NodeVariableType.BINARY)
        node1 = Node.from_dict(node.to_dict())

        self.assertNotEqual(node, node1)
        self.assertEqual(node.identifier, node1.identifier)
        self.assertEqual(node.variable_type, node1.variable_type)
        self.assertEqual(node.meta, node1.meta)
