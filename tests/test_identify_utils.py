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
from itertools import combinations

from cai_causal_graph import CausalGraph, TimeSeriesCausalGraph
from cai_causal_graph.identify_utils import identify_confounders, identify_instruments
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

    def test_symmetric(self):
        # Identify confounders but swap source and destination; test function is symmetric
        for cg in [self.graph_1, self.graph_2, self.graph_3, self.graph_4]:
            cg_rev = cg.copy()
            cg_rev.remove_edge('x', 'y')
            cg_rev.add_edge('y', 'x')

            # Let us check all pairs
            for u, v in combinations(['u', 'z', 'x', 'y'], 2):
                confounders_1a = identify_confounders(cg, node_1=u, node_2=v)
                confounders_1b = identify_confounders(cg, node_1=v, node_2=u)
                confounders_2a = identify_confounders(cg_rev, node_1=u, node_2=v)
                confounders_2b = identify_confounders(cg_rev, node_1=v, node_2=u)
                self.assertSetEqual(set(confounders_1a), set(confounders_1b))
                self.assertSetEqual(set(confounders_2a), set(confounders_2b))
                if (u == 'x' and v == 'y') or (u == 'y' and v == 'x'):
                    # If x, y then the answers should match for the graphs with swapped x - y edge
                    self.assertSetEqual(set(confounders_1a), set(confounders_2a))

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


class TestIdentifyInstruments(unittest.TestCase):
    def test_simple(self):
        # define a simple graph (standard instrumental variable example)
        cg = CausalGraph()
        cg.add_edge('z', 'x')
        cg.add_edge('u', 'x')
        cg.add_edge('u', 'y')
        cg.add_edge('x', 'y')

        # identify instrumental variables
        instruments = identify_instruments(cg, source='x', destination='y')
        self.assertSetEqual(set(instruments), {'z'})

    def test_multiple_instruments(self):
        # define a simple graph with multiple instruments (in a chain)
        cg = CausalGraph()
        cg.add_edge('z_3', 'z_2')
        cg.add_edge('z_2', 'z_1')
        cg.add_edge('z_1', 'x')
        cg.add_edge('u', 'x')
        cg.add_edge('u', 'y')
        cg.add_edge('x', 'y')

        # identify instrumental variables
        instruments = identify_instruments(cg, source='x', destination='y')
        self.assertSetEqual(set(instruments), {'z_3', 'z_2', 'z_1'})

    def test_diamond_and_multiple_confounders(self):
        # define a graph with potential instruments in a diamond shape and multiple confounders
        cg = CausalGraph()
        cg.add_edge('z_3', 'z_2')
        cg.add_edge('z_3', 'z_1')
        cg.add_edge('z_2', 'x')
        cg.add_edge('z_1', 'x')
        cg.add_edge('u_2', 'x')
        cg.add_edge('u_2', 'y')
        cg.add_edge('u_1', 'x')
        cg.add_edge('u_1', 'y')
        cg.add_edge('x', 'y')

        # identify instrumental variables
        instruments = identify_instruments(cg, source='x', destination='y')
        self.assertSetEqual(set(instruments), {'z_3'})

    def test_no_instruments(self):
        # graph 1: directed edge between instrument and destination
        cg = CausalGraph()
        cg.add_edge('z', 'x')
        cg.add_edge('u', 'x')
        cg.add_edge('u', 'y')
        cg.add_edge('x', 'y')
        cg.add_edge('z', 'y')

        # identify instrumental variables
        instruments = identify_instruments(cg, source='x', destination='y')
        self.assertEqual(len(instruments), 0)

        # graph 2: confounder between instrument and destination
        cg = CausalGraph()
        cg.add_edge('z', 'x')
        cg.add_edge('u', 'x')
        cg.add_edge('u', 'y')
        cg.add_edge('x', 'y')
        cg.add_edge('w', 'z')
        cg.add_edge('w', 'y')

        # identify instrumental variables
        instruments = identify_instruments(cg, source='x', destination='y')
        self.assertEqual(len(instruments), 0)

        # graph 3: directed edge between instrument and confounder
        cg = CausalGraph()
        cg.add_edge('z', 'x')
        cg.add_edge('u', 'x')
        cg.add_edge('u', 'y')
        cg.add_edge('x', 'y')
        cg.add_edge('z', 'u')

        # identify instrumental variables
        instruments = identify_instruments(cg, source='x', destination='y')
        self.assertEqual(len(instruments), 0)

        # graph 3: no instrument at all but confounder has a parent
        cg = CausalGraph()
        cg.add_edge('u', 'x')
        cg.add_edge('u', 'y')
        cg.add_edge('x', 'y')
        cg.add_edge('z', 'u')

        # identify instrumental variables
        instruments = identify_instruments(cg, source='x', destination='y')
        self.assertEqual(len(instruments), 0)

    def test_non_dag(self):
        # create a mixed graph
        cg = CausalGraph()
        cg.add_edge('z', 'x', edge_type=EDGE_T.UNDIRECTED_EDGE)
        cg.add_edge('u', 'x')
        cg.add_edge('u', 'y')
        cg.add_edge('x', 'y')

        with self.assertRaises(TypeError):
            identify_instruments(cg, source='x', destination='y')

    def test_time_series_graph(self):
        # create a time-series causal graph
        ts_cg = TimeSeriesCausalGraph()
        ts_cg.add_edge('z', 'x')
        ts_cg.add_edge('u', 'x')
        ts_cg.add_edge('u', 'y')
        ts_cg.add_edge('x', 'y')
        ts_cg.add_edge('z lag(n=1)', 'z')
        ts_cg.add_edge('u lag(n=1)', 'u')
        ts_cg.add_edge('x lag(n=1)', 'x')
        ts_cg.add_edge('y lag(n=1)', 'y')

        # identify instrumental variables
        instruments = identify_instruments(ts_cg, source='x', destination='y')
        self.assertSetEqual(set(instruments), {'z', 'z lag(n=1)', 'x lag(n=1)'})
