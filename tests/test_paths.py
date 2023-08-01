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

from cai_causal_graph import CausalGraph
from cai_causal_graph.graph_components import Node
from cai_causal_graph.type_definitions import EDGE_T


class TestTopologicalOrder(unittest.TestCase):
    def test_confounder(self):
        g = CausalGraph()
        g.add_edge('x', 'y')
        g.add_edge('x', 't')
        g.add_edge('t', 'y')

        top_order = g.get_topological_order()
        self.assertListEqual(top_order, ['x', 't', 'y'])

        top_order = g.get_topological_order(return_all=True)
        self.assertListEqual(top_order, [['x', 't', 'y']])

    def test_collider(self):
        g = CausalGraph()
        g.add_edge('x', 'c')
        g.add_edge('y', 'c')

        top_order = g.get_topological_order()
        all_order = [['x', 'y', 'c'], ['y', 'x', 'c']]
        self.assertIn(top_order, all_order)
        all_order_gen = g.get_topological_order(return_all=True)
        self.assertEqual(len(all_order_gen), len(all_order))

        for order in all_order_gen:
            self.assertIn(order, all_order)

    def test_fork(self):
        g = CausalGraph()
        g.add_edge('c', 'x')
        g.add_edge('c', 'y')

        top_order = g.get_topological_order()
        all_order = [['c', 'x', 'y'], ['c', 'y', 'x']]
        self.assertIn(top_order, all_order)
        all_order_gen = g.get_topological_order(return_all=True)
        self.assertEqual(len(all_order_gen), len(all_order))

        for order in all_order_gen:
            self.assertIn(order, all_order)


class TestCommonAncestorsDescendants(unittest.TestCase):
    def test_fork(self):
        g = CausalGraph()
        g.add_edge('a', 'b')
        g.add_edge('a', 'c')
        g.add_edge('d', 'b')
        g.add_edge('e', 'c')

        self.assertSetEqual(g.get_common_ancestors('b', 'c'), {'a'})
        self.assertSetEqual(g.get_common_ancestors('d', 'e'), set())

    def test_collider(self):
        g = CausalGraph()
        g.add_edge('a', 'b')
        g.add_edge('c', 'b')
        g.add_edge('a', 'd')
        g.add_edge('c', 'e')

        self.assertSetEqual(g.get_common_descendants('a', 'c'), {'b'})
        self.assertSetEqual(g.get_common_ancestors('b', 'd'), {'a'})


class TestAncestorsDescendants(unittest.TestCase):
    def test_chain(self):
        g = CausalGraph()
        g.add_edge('a', 'b')
        g.add_edge('b', 'c')

        self.assertSetEqual(g.get_ancestors('c'), {'a', 'b'})
        self.assertTrue(g.is_descendant('c', {'a', 'b'}))
        self.assertTrue(g.is_descendant('c', ['a']))
        self.assertTrue(g.is_descendant('c', 'b'))
        self.assertFalse(g.is_descendant('a', {'b'}))

        self.assertSetEqual(g.get_descendants('a'), {'b', 'c'})
        self.assertTrue(g.is_ancestor('a', ['b', 'c']))
        self.assertTrue(g.is_ancestor('b', 'c'))
        self.assertTrue(g.is_ancestor('a', 'c'))

    def test_fork(self):
        g = CausalGraph()
        g.add_edge('a', 'b')
        g.add_edge('a', 'c')

        self.assertSetEqual(g.get_descendants('a'), {'b', 'c'})
        self.assertTrue(g.is_descendant('b', 'a'))
        self.assertTrue(g.is_descendant('c', 'a'))
        self.assertFalse(g.is_descendant('c', ['a', 'b']))

        self.assertSetEqual(g.get_ancestors('b'), {'a'})
        self.assertSetEqual(g.get_ancestors('c'), {'a'})
        self.assertTrue(g.is_ancestor('a', ['b', 'c']))

    def test_collider(self):
        g = CausalGraph()
        g.add_edge('a', 'b')
        g.add_edge('c', 'b')

        self.assertSetEqual(g.get_ancestors('b'), {'a', 'c'})
        self.assertTrue(g.is_descendant('b', {'a', 'c'}))
        self.assertTrue(g.is_ancestor('a', 'b'))
        self.assertFalse(g.is_ancestor('a', ['b', 'c']))
        self.assertSetEqual(g.get_descendants('a'), {'b'})


class TestCausalPaths(unittest.TestCase):
    def test_case_1(self):
        g = CausalGraph()
        g.add_edge('a', 'b')
        g.add_edge('b', 'c')
        g.add_edge('b', 'd')
        g.add_edge('c', 'e')
        g.add_edge('d', 'e')

        all_paths = g.get_all_causal_paths('a', 'e')

        self.assertListEqual(all_paths, [['a', 'b', 'c', 'e'], ['a', 'b', 'd', 'e']])

    def test_case_2(self):
        g = CausalGraph()
        g.add_edge('a', 'b')
        g.add_edge('b', 'c')
        g.add_edge('b', 'd')
        g.add_edge('c', 'e')
        g.add_edge('e', 'd')

        all_paths = g.get_all_causal_paths('a', 'e')

        self.assertListEqual(all_paths, [['a', 'b', 'c', 'e']])

    def test_no_path(self):
        g = CausalGraph()
        g.add_edge('a', 'b')
        g.add_edge('a', 'c')

        all_paths = g.get_all_causal_paths('b', 'c')

        self.assertListEqual(all_paths, [])

    def test_neighbours(self):
        g = CausalGraph()
        g.add_edge('a', 'b')

        all_paths = g.get_all_causal_paths('a', 'b')

        self.assertListEqual(all_paths, [['a', 'b']])

    def test_edge_does_not_exist(self):
        g = CausalGraph()

        with self.assertRaises(AssertionError):
            g.get_all_causal_paths('a', 'b')

    def test_with_nodes(self):
        g = CausalGraph()
        output_node = Node('t')
        input_node = Node('f')

        g.add_edge(input_node, output_node)
        g.add_edge(input_node, 'a')
        g.add_edge('a', output_node)

        all_paths = g.get_all_causal_paths(input_node, output_node)

        self.assertListEqual(
            all_paths,
            [[input_node.identifier, output_node.identifier], [input_node.identifier, 'a', output_node.identifier]],
        )


class TestDirectedPathExists(unittest.TestCase):
    """Class that tests the existence of a directed path between two nodes in a causal graph."""

    def test_dag(self):
        g = CausalGraph()
        g.add_edge('a', 'b')
        g.add_edge('b', 'c')
        g.add_edge('b', 'd')
        g.add_edge('c', 'e')
        g.add_edge('d', 'e')

        self.assertTrue(g.directed_path_exists('a', 'b'))
        self.assertTrue(g.directed_path_exists('a', 'c'))
        self.assertTrue(g.directed_path_exists('a', 'd'))
        self.assertTrue(g.directed_path_exists('a', 'e'))
        self.assertTrue(g.directed_path_exists('b', 'e'))

        self.assertFalse(g.directed_path_exists('c', 'd'))
        self.assertFalse(g.directed_path_exists('e', 'a'))
        self.assertFalse(g.directed_path_exists('c', 'a'))

    def test_cpdag(self):
        g = CausalGraph()
        g.add_edge('a', 'b')
        g.add_edge('b', 'c')
        g.add_edge('b', 'd', edge_type=EDGE_T.UNDIRECTED_EDGE)
        g.add_edge('c', 'e', edge_type=EDGE_T.UNDIRECTED_EDGE)
        g.add_edge('d', 'e')

        self.assertTrue(g.directed_path_exists('a', 'b'))
        self.assertTrue(g.directed_path_exists('a', 'c'))

        self.assertFalse(g.directed_path_exists('a', 'd'))
        self.assertFalse(g.directed_path_exists('a', 'e'))
        self.assertFalse(g.directed_path_exists('c', 'd'))
        self.assertFalse(g.directed_path_exists('b', 'e'))
        self.assertFalse(g.directed_path_exists('e', 'a'))
        self.assertFalse(g.directed_path_exists('c', 'a'))

    def test_no_path(self):
        g = CausalGraph()
        g.add_edge('a', 'b')
        g.add_edge('a', 'c')

        self.assertFalse(g.directed_path_exists('b', 'c'))

    def test_neighbours(self):
        g = CausalGraph()
        g.add_edge('a', 'b')

        self.assertTrue(g.directed_path_exists('a', 'b'))

    def test_edge_does_not_exist(self):
        g = CausalGraph()

        with self.assertRaises(AssertionError):
            g.directed_path_exists('a', 'b')

    def test_with_target_and_features(self):
        g = CausalGraph()
        output_node = Node('t')
        input_node = Node('f')

        g.add_edge(input_node, output_node)
        g.add_edge(input_node, 'a')
        g.add_edge('a', output_node)

        self.assertTrue(g.directed_path_exists(input_node, output_node))
