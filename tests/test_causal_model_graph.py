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

from cai_causal_graph import CausalGraph as CaiCausalGraph
from cai_causal_graph.causal_model_graph import (
    DEFAULT_EDGE_ACTIVATION,
    DEFAULT_NODE_ACTIVATION,
    DEFAULT_NODE_AGGREGATION,
    CausalModelGraph,
)
from cai_causal_graph.type_definitions import NodeVariableType


class TestCausalGraphSerialization(unittest.TestCase):
    def setUp(self):
        self.empty_graph = CausalModelGraph()

        self.nodes = ['x', 'y', 'z1', 'z2']
        self.latent_nodes = ['x_L']

        self.fully_connected_graph = CausalModelGraph()
        self.fully_connected_graph.add_node('x', aggregation='sum', variable_type=NodeVariableType.CONTINUOUS)
        self.fully_connected_graph.add_edge('x', 'z1', activations=['linear', 'piecewise_linear'])
        self.fully_connected_graph.add_edge('x', 'z2')
        self.fully_connected_graph.add_edge('y', 'z1')
        self.fully_connected_graph.add_edge('y', 'z2')

        self.fully_connected_graph_edges = [['x', 'z1'], ['x', 'z2'], ['y', 'z1'], ['y', 'z2']]

        self.fully_connected_graph_edges_sources = [('x', 2), ('y', 2)]
        self.fully_connected_graph_edges_destinations = [('z1', 2), ('z2', 2)]

        self.graph_with_latent_node = CausalModelGraph()
        self.graph_with_latent_node.add_nodes_from(['x', 'y', 'z1', 'z2'])
        self.graph_with_latent_node.add_edge('x', 'x_L')
        self.graph_with_latent_node.add_edge('x_L', 'y')
        self.graph_with_latent_node.add_edge('y', 'z1')
        self.graph_with_latent_node.add_edge('y', 'z2')
        self.graph_with_latent_node.add_edge('x', 'z2')

        self.graph_with_latent_nodes_edges_sources = [('x', 2), ('y', 2), ('x_L', 1)]
        self.graph_with_latent_nodes_edges_destinations = [('x_L', 1), ('y', 1), ('z1', 1), ('z2', 2)]
        self.graph_with_latent_nodes_edges = [['x', 'x_L'], ['x_L', 'y'], ['y', 'z1'], ['y', 'z2'], ['x', 'z2']]

    def test_cmg_to_cai_cg(self):
        # test it copies everything but the aggregation and activations
        converted_graph = CaiCausalGraph.from_dict(self.fully_connected_graph.to_dict())

        self.assertEqual(converted_graph.nodes, self.fully_connected_graph.nodes)
        self.assertEqual(converted_graph.edges, self.fully_connected_graph.edges)

    def test_cmg_from_cai_cg(self):
        # test it copies everything and set default aggregation and activations
        cai_cg = CaiCausalGraph.from_dict(self.fully_connected_graph.to_dict())
        converted_graph = CausalModelGraph.from_dict(cai_cg.to_dict())

        for node in converted_graph.nodes:
            self.assertEqual(node.aggregation, DEFAULT_NODE_AGGREGATION)
            self.assertEqual(node.activations, DEFAULT_NODE_ACTIVATION)

        # same for edges
        for edge in converted_graph.edges:
            self.assertEqual(edge.activations, DEFAULT_EDGE_ACTIVATION)
