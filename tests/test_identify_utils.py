"""
Copyright (c) 2023 by Impulse Innovations Ltd. Private and confidential. Part of the causaLens product.
"""
import unittest

from cai_causal_graph import CausalGraph
from cai_causal_graph.identify_utils import identify_confounders


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
        # compute confounders between treatment and outcome
        confounders = identify_confounders(self.graph_1, source='x', destination='y')
        self.assertSetEqual(set(confounders), {'z'})

    def test_graph_2(self):
        # compute confounders between treatment and outcome
        confounders = identify_confounders(self.graph_2, source='x', destination='y')
        self.assertSetEqual(set(confounders), {'u'})

    def test_graph_3(self):
        # compute confounders between treatment and outcome
        confounders = identify_confounders(self.graph_3, source='x', destination='y')
        self.assertSetEqual(set(confounders), {'u'})

    def test_graph_4(self):
        # compute confounders between treatment and outcome
        confounders = identify_confounders(self.graph_4, source='x', destination='y')
        self.assertSetEqual(set(confounders), {'u', 'z'})
