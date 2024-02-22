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

from cai_causal_graph import CausalGraph, EdgeType, NodeVariableType
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
        self.assertEqual(node.identifier, node.node_name)
        self.assertEqual(node1.identifier, node1.node_name)

        # test time series node serialization-deserialization
        node = TimeSeriesNode(identifier='a lag(n=2)', meta={'color': 'blue'}, variable_type=NodeVariableType.BINARY)
        node1 = TimeSeriesNode.from_dict(node.to_dict())

        self.assertTrue(node.__eq__(node1, deep=True))
        self.assertEqual(node.identifier, node.node_name)
        self.assertEqual(node1.identifier, node1.node_name)

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
        self.assertEqual(node.identifier, node.node_name)
        self.assertEqual(node1.identifier, node1.node_name)
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
        self.assertEqual(node.identifier, node.node_name)
        self.assertEqual(node1.identifier, node1.node_name)
        self.assertEqual(node.variable_type, node1.variable_type)
        self.assertEqual(node.meta, node1.meta)

    def test_equality(self):
        # Most of the equality is exercised above but want to check the edge logic here.
        source = Node(identifier='a', meta={'color': 'blue'}, variable_type=NodeVariableType.BINARY)
        destination = Node(identifier='b', meta={'color': 'green'}, variable_type=NodeVariableType.CONTINUOUS)
        edge = Edge(source=source, destination=destination, edge_type=EdgeType.UNDIRECTED_EDGE, meta={'color': 'red'})

        edge_dict = edge.to_dict()
        edge_recovered = Edge.from_dict(edge_dict)

        edge_reversed = Edge(
            source=destination, destination=source, edge_type=EdgeType.UNDIRECTED_EDGE, meta={'color': 'red'}
        )

        edge_different_direction = Edge(
            source=source, destination=destination, edge_type=EdgeType.BIDIRECTED_EDGE, meta={'color': 'red'}
        )

        self.assertEqual(edge, edge_recovered)
        self.assertEqual(edge, edge_reversed)
        self.assertEqual(edge_recovered, edge_reversed)
        self.assertNotEqual(edge, edge_different_direction)
        self.assertNotEqual(edge_recovered, edge_different_direction)
        self.assertNotEqual(edge_reversed, edge_different_direction)

        self.assertTrue(edge.__eq__(edge_recovered, deep=True))
        self.assertTrue(edge.__eq__(edge_reversed, deep=True))
        self.assertFalse(edge.__eq__(edge_different_direction, deep=True))
        self.assertTrue(edge_recovered.__eq__(edge, deep=True))
        self.assertTrue(edge_recovered.__eq__(edge_reversed, deep=True))
        self.assertFalse(edge_recovered.__eq__(edge_different_direction, deep=True))
        self.assertTrue(edge_reversed.__eq__(edge, deep=True))
        self.assertTrue(edge_reversed.__eq__(edge_recovered, deep=True))
        self.assertFalse(edge_reversed.__eq__(edge_different_direction, deep=True))
        self.assertFalse(edge_different_direction.__eq__(edge, deep=True))
        self.assertFalse(edge_different_direction.__eq__(edge_recovered, deep=True))
        self.assertFalse(edge_different_direction.__eq__(edge_reversed, deep=True))

        edge_reversed.metadata['color'] = 'blue'
        self.assertFalse(edge.__eq__(edge_reversed, deep=True))
        self.assertFalse(edge_recovered.__eq__(edge_reversed, deep=True))
        self.assertFalse(edge_different_direction.__eq__(edge_reversed, deep=True))
        self.assertFalse(edge_reversed.__eq__(edge, deep=True))
        self.assertFalse(edge_reversed.__eq__(edge_recovered, deep=True))
        self.assertFalse(edge_reversed.__eq__(edge_different_direction, deep=True))

    def test_is_source_node(self):

        cg = CausalGraph()
        cg.add_edge('a', 'b')
        cg.add_edge('b', 'c')

        self.assertTrue(cg.get_node('a').is_source_node())
        self.assertFalse(cg.get_node('b').is_source_node())
        self.assertFalse(cg.get_node('c').is_source_node())

    def test_is_sink_node(self):

        cg = CausalGraph()
        cg.add_edge('a', 'b')
        cg.add_edge('b', 'c')

        self.assertFalse(cg.get_node('a').is_sink_node())
        self.assertFalse(cg.get_node('b').is_sink_node())
        self.assertTrue(cg.get_node('c').is_sink_node())
