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

from cai_causal_graph import CausalGraph, TimeSeriesCausalGraph
from cai_causal_graph.identify_utils import identify_confounders
from cai_causal_graph.type_definitions import EDGE_T


class TestIdentifyConfounders(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # graph: z -> u -> x -> y and z -> y
        graph_1 = CausalGraph()
        graph_1.add_edge('z', 'u')
        graph_1.add_edge('z', 'y')
        graph_1.add_edge('u', 'x')
        graph_1.add_edge('x', 'y')
        cls.graph_1 = graph_1

        # graph: z -> u -> x -> y and u -> y
        graph_2 = CausalGraph()
        graph_2.add_edge('z', 'u')
        graph_2.add_edge('u', 'x')
        graph_2.add_edge('u', 'y')
        graph_2.add_edge('x', 'y')
        cls.graph_2 = graph_2

        # graph: z -> u -> x -> y and z -> y and u -> y
        graph_3 = CausalGraph()
        graph_3.add_edge('z', 'u')
        graph_3.add_edge('u', 'x')
        graph_3.add_edge('z', 'y')
        graph_3.add_edge('u', 'y')
        graph_3.add_edge('x', 'y')
        cls.graph_3 = graph_3

        # graph: z -> u -> x -> y and z -> y and u -> y
        graph_4 = CausalGraph()
        graph_4.add_edge('z', 'x')
        graph_4.add_edge('u', 'x')
        graph_4.add_edge('z', 'y')
        graph_4.add_edge('u', 'y')
        graph_4.add_edge('x', 'y')
        cls.graph_4 = graph_4

    def test_graph_1(self):
        # compute confounders between source and destination
        confounders = identify_confounders(self.graph_1, node_1='x', node_2='y')
        self.assertSetEqual(set(confounders), {'z'})

    def test_graph_2(self):
        # compute confounders between source and destination
        confounders = identify_confounders(self.graph_2, node_1='x', node_2='y')
        self.assertSetEqual(set(confounders), {'u'})

    def test_graph_3(self):
        # compute confounders between source and destination
        confounders = identify_confounders(self.graph_3, node_1='x', node_2='y')
        self.assertSetEqual(set(confounders), {'u'})

    def test_graph_4(self):
        # compute confounders between source and destination
        confounders = identify_confounders(self.graph_4, node_1='x', node_2='y')
        self.assertSetEqual(set(confounders), {'u', 'z'})

    def test_non_dag(self):
        # create a mixed graph
        graph = CausalGraph()
        graph.add_edge('z', 'u', edge_type=EDGE_T.UNDIRECTED_EDGE)
        graph.add_edge('z', 'y')
        graph.add_edge('u', 'x')
        graph.add_edge('x', 'y')

        with self.assertRaises(TypeError):
            identify_confounders(graph, node_1='x', node_2='y')

    def test_reverse_edge(self):
        # identify confounders but swap source and destination
        confounders = identify_confounders(self.graph_1, node_1='x', node_2='y')
        confounders_rev = identify_confounders(self.graph_1, node_1='y', node_2='x')
        self.assertSetEqual(set(confounders), set(confounders_rev))

    def test_time_series_graph(self):
        # create a time-series causal graph
        ts_cg = TimeSeriesCausalGraph()
        ts_cg.add_edge('z', 'x')
        ts_cg.add_edge('z', 'y')
        ts_cg.add_edge('x', 'y')
        ts_cg.add_edge('z lag(n=1)', 'z')
        ts_cg.add_edge('x lag(n=1)', 'x')
        ts_cg.add_edge('y lag(n=1)', 'y')

        # compute confounders between source and destination
        confounders = identify_confounders(ts_cg, node_1='x', node_2='y')
        self.assertSetEqual(set(confounders), {'z'})
