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


class TestDSeparation(unittest.TestCase):
    def test_unknown_node(self):
        g = CausalGraph()
        g.add_edge('A', 'B')
        g.add_edge('C', 'B')

        with self.assertRaises(AssertionError):
            g.get_d_separation_set('A', 'D')

        with self.assertRaises(AssertionError):
            g.is_d_separated({'A', 'B'}, {'D'})

        with self.assertRaises(AssertionError):
            g.is_d_separated('A', 'D')

        with self.assertRaises(AssertionError):
            g.is_minimally_d_separated('A', 'D')

    def test_case_1(self):
        g = CausalGraph()
        g.add_edge('A', 'B')
        g.add_edge('C', 'B')
        g.add_edge('B', 'D')
        g.add_edge('D', 'E')
        g.add_edge('B', 'F')
        g.add_edge('G', 'E')

        self.assertFalse(g.is_d_separated('B', 'E'))

        # minimal set of the corresponding graph
        # for B and E should be (D,)
        z_min = g.get_d_separation_set('B', 'E')

        # the minimal separating set should pass the test for minimality
        self.assertTrue(g.is_d_separated('B', 'E', z_min))
        self.assertTrue(g.is_minimally_d_separated('B', 'E', z_min))
        self.assertSetEqual({'D'}, z_min)

    def test_case_2(self):
        g = CausalGraph()
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        g.add_edge('B', 'D')
        g.add_edge('D', 'C')

        self.assertFalse(g.is_d_separated('A', 'C'))

        zmin = g.get_d_separation_set('A', 'C')

        self.assertTrue(g.is_d_separated('A', 'C', zmin))
        self.assertSetEqual(zmin, {'B'})

    def test_collider(self):
        g = CausalGraph()
        g.add_edge('A', 'B')
        g.add_edge('C', 'B')

        self.assertTrue(g.is_d_separated('A', 'C'))
        self.assertFalse(g.is_d_separated('A', 'C', {'B'}))

        # minimal set of the corresponding graph
        # for A and C should be empty
        z_min = g.get_d_separation_set('A', 'C')

        # the minimal separating set should pass the test for minimality
        self.assertTrue(g.is_minimally_d_separated('A', 'C'))
        self.assertTrue(g.is_minimally_d_separated('A', 'C', z_min))
        self.assertSetEqual(set(), z_min)

    def test_node_like(self):
        g = CausalGraph()
        output_node = Node('B')
        input_node = Node('A')
        g.add_edge(input_node, output_node)
        g.add_edge('C', output_node)

        self.assertTrue(g.is_d_separated('A', 'C'))
        self.assertTrue(g.is_d_separated(input_node, 'C'))
        self.assertFalse(g.is_d_separated(input_node, 'C', {output_node}))

        g = CausalGraph()
        g.add_edge(input_node, output_node)
        g.add_edge(output_node, 'C')

        self.assertSetEqual(g.get_d_separation_set(input_node, 'C'), {output_node.identifier})

    def test_parent(self):
        g = CausalGraph()
        g.add_edge('A', 'B')

        with self.assertRaises(AssertionError):
            g.get_d_separation_set('A', 'B')
