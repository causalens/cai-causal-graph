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

import numpy

from cai_causal_graph import CausalGraph, EdgeType
from cai_causal_graph.graph_components import Node


class TestNodeSupport(unittest.TestCase):
    def test_coerce_to_nodelike(self):
        nodelike = CausalGraph.coerce_to_nodelike(Node(identifier='x'))
        self.assertIsInstance(nodelike, Node)
        self.assertEqual(nodelike.identifier, 'x')
        nodelike = CausalGraph.coerce_to_nodelike('x')
        self.assertIsInstance(nodelike, str)
        self.assertEqual(nodelike, 'x')
        nodelike = CausalGraph.coerce_to_nodelike(0)
        self.assertIsInstance(nodelike, str)
        self.assertEqual(nodelike, '0')

    def test_causal_graph_accepts_nodes(self):
        causal_graph = CausalGraph()
        x = Node('x')
        y = Node('y')

        causal_graph.add_edge(x, y)
        edge = causal_graph[x, y]
        self.assertEqual(('x', 'y'), edge.get_edge_pair())

    def test_input_output_lists(self):
        input_list = ['a', Node('b')]
        output_list = ['x', Node('y')]

        # specify a graph with nodes and edges
        causal_graph = CausalGraph(
            input_list=input_list,
            output_list=output_list,
            fully_connected=True,
        )

        node_names = set(causal_graph.get_node_names())
        self.assertSetEqual({'a', 'b', 'x', 'y'}, node_names)
        all_edges = causal_graph.get_edges()
        directed_edges = causal_graph.get_edges(edge_type=EdgeType.DIRECTED_EDGE)
        self.assertListEqual(all_edges, directed_edges)
        self.assertListEqual(directed_edges, causal_graph.get_directed_edges())
        self.assertEqual(len(all_edges), 4)
        self.assertTrue(causal_graph.edge_exists('a', 'x', edge_type=EdgeType.DIRECTED_EDGE))
        self.assertTrue(causal_graph.edge_exists('b', 'x', edge_type=EdgeType.DIRECTED_EDGE))
        self.assertTrue(causal_graph.edge_exists('a', 'y', edge_type=EdgeType.DIRECTED_EDGE))
        self.assertTrue(causal_graph.edge_exists('b', 'y', edge_type=EdgeType.DIRECTED_EDGE))

        # specify a graph with nodes but no edges
        causal_graph = CausalGraph(
            input_list=input_list,
            output_list=output_list,
            fully_connected=False,
        )

        # Nodes should still be added, but there should be no edges.
        node_names = set(causal_graph.get_node_names())
        self.assertSetEqual({'a', 'b', 'x', 'y'}, node_names)
        self.assertEqual(len(causal_graph.get_edges()), 0)

        # specify a graph with only inputs so fully_connected param does nothing
        for fc in [True, False]:
            causal_graph = CausalGraph(
                input_list=input_list,
                output_list=None,
                fully_connected=fc,
            )

            # Nodes should still be added, but there should be no edges.
            node_names = set(causal_graph.get_node_names())
            self.assertSetEqual({'a', 'b'}, node_names)
            self.assertEqual(len(causal_graph.get_edges()), 0)

        # specify a graph with only inputs so fully_connected param does nothing
        for fc in [True, False]:
            causal_graph = CausalGraph(
                input_list=None,
                output_list=output_list,
                fully_connected=fc,
            )

            # Nodes should still be added, but there should be no edges.
            node_names = set(causal_graph.get_node_names())
            self.assertSetEqual({'x', 'y'}, node_names)
            self.assertEqual(len(causal_graph.get_edges()), 0)

        # specify a graph with nothing
        causal_graph = CausalGraph()
        self.assertEqual(len(causal_graph.get_nodes()), 0)
        self.assertEqual(len(causal_graph.get_edges()), 0)

    def test_adjacency_with_inputs_and_outputs(self):
        input_list = ['a', Node('b'), 'c']
        output_list = [Node('y')]

        adj = numpy.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])

        graph = CausalGraph(input_list=input_list, output_list=output_list, fully_connected=False)
        graph.add_edge('a', 'b')
        graph.add_edge('b', 'c')
        graph.add_edge('c', 'y')

        # construct the graph from an adjacency matrix and compare
        graph_2 = CausalGraph.from_adjacency_matrix(adj, [*input_list, *output_list])

        self.assertEqual(graph_2, graph)
