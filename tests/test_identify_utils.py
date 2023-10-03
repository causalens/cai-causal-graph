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
from cai_causal_graph.exceptions import CausalGraphErrors
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
        # Confirm you can also pass Nodes
        confounders = identify_confounders(
            self.graph_1, node_1=self.graph_1.get_node('x'), node_2=self.graph_1.get_node('y')
        )
        self.assertSetEqual(set(confounders), {'z'})

    def test_graph_2(self):
        # compute confounders between source and destination
        confounders = identify_confounders(self.graph_2, node_1='x', node_2='y')
        self.assertSetEqual(set(confounders), {'u'})
        # Confirm you can pass mix of Node and identifier
        confounders = identify_confounders(self.graph_2, node_1='x', node_2=self.graph_2.get_node('y'))
        self.assertSetEqual(set(confounders), {'u'})

    def test_graph_3(self):
        # compute confounders between source and destination
        confounders = identify_confounders(self.graph_3, node_1='x', node_2='y')
        self.assertSetEqual(set(confounders), {'u'})
        # Confirm you can pass mix of Node and identifier
        confounders = identify_confounders(self.graph_3, node_1=self.graph_3.get_node('x'), node_2='y')
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

    def test_error_cases(self):
        # node_1 not in graph
        with self.assertRaises(CausalGraphErrors.NodeDoesNotExistError):
            identify_confounders(self.graph_1, node_1='w', node_2='y')

        # node_2 not in graph
        with self.assertRaises(CausalGraphErrors.NodeDoesNotExistError):
            identify_confounders(self.graph_1, node_1='x', node_2='w')

        # node_1 == node_2
        with self.assertRaises(ValueError):
            identify_confounders(self.graph_1, node_1='x', node_2='x')
        with self.assertRaises(ValueError):
            identify_confounders(self.graph_1, node_1='x', node_2=self.graph_1.get_node('x'))
        with self.assertRaises(ValueError):
            identify_confounders(self.graph_1, node_1=self.graph_1.get_node('x'), node_2='x')
        with self.assertRaises(ValueError):
            identify_confounders(self.graph_1, node_1=self.graph_1.get_node('x'), node_2=self.graph_1.get_node('x'))

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

    def test_bigger_graph(self):
        cg = CausalGraph()
        cg.add_edge('z', 'u')
        cg.add_edge('z', 'y')
        cg.add_edge('u', 'x')
        cg.add_edge('u', 'y')
        # mediators between x and y
        cg.add_edge('x', 'm1')
        cg.add_edge('m1', 'm2')
        cg.add_edge('m2', 'm3')
        cg.add_edge('m3', 'y')

        # bunch of kids on u, z, x, y, and mediators that we don't care about
        cg.add_edge('u', 'uc0')
        cg.add_edge('z', 'zc0')
        cg.add_edge('x', 'xc0')
        cg.add_edge('m1', 'm1c0')
        cg.add_edge('m2', 'm2c0')
        cg.add_edge('m3', 'm3c0')
        cg.add_edge('y', 'yc0')
        for i in range(1, 10):
            cg.add_edge(f'uc{i - 1}', f'uc{i}')
            cg.add_edge(f'zc{i - 1}', f'zc{i}')
            cg.add_edge(f'xc{i-1}', f'xc{i}')
            cg.add_edge(f'm1c{i - 1}', f'm1c{i}')
            cg.add_edge(f'm2c{i - 1}', f'm2c{i}')
            cg.add_edge(f'm3c{i - 1}', f'm3c{i}')
            cg.add_edge(f'yc{i - 1}', f'yc{i}')

        confounders = identify_confounders(cg, 'x', 'y')
        self.assertSetEqual(set(confounders), {'u'})
        confounders = identify_confounders(cg, 'y', 'x')
        self.assertSetEqual(set(confounders), {'u'})

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
