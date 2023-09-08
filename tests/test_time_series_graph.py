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

import json
import unittest

import numpy

from cai_causal_graph import CausalGraph, EdgeType, NodeVariableType, TimeSeriesCausalGraph
from cai_causal_graph.exceptions import CausalGraphErrors
from cai_causal_graph.graph_components import TimeSeriesNode
from cai_causal_graph.utils import extract_names_and_lags, get_name_with_lag, get_variable_name_and_lag


class TestTimeSeriesCausalGraph(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up a time series causal graph tests.

        Create a causal graph `graph_1` with 4 nodes with lagged variables
        autoregressive edges:
            - X1(t-1) -> X1(t)
            - X2(t-1) -> X2(t)
            - X3(t-1) -> X3(t)
            - X1(t-1) -> X2(t)
        instantaneous edges:
            - X1(t) -> X3(t)
        """
        self.rng = numpy.random.default_rng(42)
        self.nodes = ['X1', 'X1 lag(n=1)', 'X2 lag(n=1)', 'X3 lag(n=1)', 'X2', 'X3']

        # define a DAG
        self.dag = CausalGraph()
        self.dag.add_nodes_from(self.nodes)
        # auto-regressive edges
        self.dag.add_edge('X1 lag(n=1)', 'X1', edge_type=EdgeType.DIRECTED_EDGE)
        self.dag.add_edge('X2 lag(n=1)', 'X2', edge_type=EdgeType.DIRECTED_EDGE)
        self.dag.add_edge('X3 lag(n=1)', 'X3', edge_type=EdgeType.DIRECTED_EDGE)
        self.dag.add_edge('X1 lag(n=1)', 'X2', edge_type=EdgeType.DIRECTED_EDGE)
        # instantaneous edges
        self.dag.add_edge('X1', 'X3', edge_type=EdgeType.DIRECTED_EDGE)
        self.tsdag = TimeSeriesCausalGraph.from_causal_graph(self.dag)
        self.ground_truth_summary_graph = CausalGraph()
        self.ground_truth_summary_graph.add_nodes_from(['X1', 'X2', 'X3'])
        self.ground_truth_summary_graph.add_edge('X1', 'X2', edge_type=EdgeType.DIRECTED_EDGE)
        self.ground_truth_summary_graph.add_edge('X1', 'X3', edge_type=EdgeType.DIRECTED_EDGE)
        # dag is already minimal
        self.ground_truth_minimal_graph = self.tsdag.copy()

        # create a more complex DAG as follows
        self.nodes_1 = [
            'X1',
            'X1 lag(n=1)',
            'X1 lag(n=2)',
            'X2',
            'X2 lag(n=1)',
            'X2 lag(n=2)',
            'X3',
            'X3 lag(n=1)',
            'X3 lag(n=2)',
        ]
        self.dag_1 = CausalGraph()
        self.dag_1.add_nodes_from(self.nodes_1)
        # auto-regressive edges
        # X1
        self.dag_1.add_edge('X1 lag(n=1)', 'X1', edge_type=EdgeType.DIRECTED_EDGE)
        self.dag_1.add_edge('X1 lag(n=2)', 'X1 lag(n=1)', edge_type=EdgeType.DIRECTED_EDGE)
        # X2
        self.dag_1.add_edge('X2 lag(n=1)', 'X2', edge_type=EdgeType.DIRECTED_EDGE)
        self.dag_1.add_edge('X2 lag(n=2)', 'X2 lag(n=1)', edge_type=EdgeType.DIRECTED_EDGE)
        # X3
        self.dag_1.add_edge('X3 lag(n=1)', 'X3', edge_type=EdgeType.DIRECTED_EDGE)
        self.dag_1.add_edge('X3 lag(n=2)', 'X3 lag(n=1)', edge_type=EdgeType.DIRECTED_EDGE)

        # X3 (t-1) -> X1 (t)
        self.dag_1.add_edge('X3 lag(n=1)', 'X1', edge_type=EdgeType.DIRECTED_EDGE)
        # X3 (t-2) -> X2 (t)
        self.dag_1.add_edge('X3 lag(n=2)', 'X2', edge_type=EdgeType.DIRECTED_EDGE)

        # instantaneous edges (X3 -> X1)
        self.dag_1.add_edge('X3', 'X1', edge_type=EdgeType.DIRECTED_EDGE)
        self.tsdag_1 = TimeSeriesCausalGraph.from_causal_graph(self.dag_1)

        self.ground_truth_summary_graph_1 = CausalGraph()
        self.ground_truth_summary_graph_1.add_nodes_from(['X1', 'X2', 'X3'])

        self.ground_truth_summary_graph_1.add_edge('X3', 'X1', edge_type=EdgeType.DIRECTED_EDGE)
        self.ground_truth_summary_graph_1.add_edge('X3', 'X2', edge_type=EdgeType.DIRECTED_EDGE)

        # create the minimal graph
        self.ground_truth_minimal_graph_1 = TimeSeriesCausalGraph()
        self.ground_truth_minimal_graph_1.add_nodes_from(['X1', 'X2', 'X3'])
        self.ground_truth_minimal_graph_1.add_edge('X3', 'X1', edge_type=EdgeType.DIRECTED_EDGE)
        self.ground_truth_minimal_graph_1.add_edge('X1 lag(n=1)', 'X1', edge_type=EdgeType.DIRECTED_EDGE)
        self.ground_truth_minimal_graph_1.add_edge('X2 lag(n=1)', 'X2', edge_type=EdgeType.DIRECTED_EDGE)
        self.ground_truth_minimal_graph_1.add_edge('X3 lag(n=1)', 'X3', edge_type=EdgeType.DIRECTED_EDGE)
        self.ground_truth_minimal_graph_1.add_edge('X3 lag(n=1)', 'X1', edge_type=EdgeType.DIRECTED_EDGE)
        self.ground_truth_minimal_graph_1.add_edge('X3 lag(n=2)', 'X2', edge_type=EdgeType.DIRECTED_EDGE)

        self.tsdag_2 = TimeSeriesCausalGraph()
        self.tsdag_2.add_nodes_from(['X1'])
        self.tsdag_2.add_edge('X1 lag(n=2)', 'X1', edge_type=EdgeType.DIRECTED_EDGE)

        # third graph
        causal_graph_3 = CausalGraph()
        nodes = [
            'A',
            'B',
            'C',
            'D',
            'E',
            'F',
            'A lag(n=1)',
            'B lag(n=1)',
            'F lag(n=1)',
        ]
        causal_graph_3.add_nodes_from(nodes)
        causal_graph_3.add_edge('A', 'C')
        causal_graph_3.add_edge('B', 'C')
        causal_graph_3.add_edge('C', 'D')
        causal_graph_3.add_edge('D', 'E')
        causal_graph_3.add_edge('C', 'E')
        causal_graph_3.add_edge('E', 'F')

        causal_graph_3.add_edge('A lag(n=1)', 'A')
        causal_graph_3.add_edge('B lag(n=1)', 'B')
        causal_graph_3.add_edge('F lag(n=1)', 'F')

        self.ground_truth_minimal_graph_3 = TimeSeriesCausalGraph.from_causal_graph(causal_graph_3.copy())

        causal_graph_3.add_node('C lag(n=1)')
        causal_graph_3.add_node('D lag(n=1)')
        causal_graph_3.add_node('E lag(n=1)')

        # We would not need to do this, but we want to test that the underlying time series graph is being correctly
        # constructed.
        for edge in causal_graph_3.edges:
            source_name = edge.source.identifier
            dest_name = edge.destination.identifier
            if 'lag(n=1)' not in source_name and 'lag(n=1)' not in dest_name:
                causal_graph_3.add_edge(f'{source_name} lag(n=1)', f'{dest_name} lag(n=1)')

        self.tsdag_3 = TimeSeriesCausalGraph.from_causal_graph(causal_graph_3)

    def test_constructor(self):
        input_nodes = ['A lag(n=2)', 'A lag(n=1)', 'B']
        output_nodes = ['A', 'B future(n=1)']

        # Just confirm this does not raise.
        ts_graph = TimeSeriesCausalGraph(input_list=input_nodes, output_list=output_nodes, fully_connected=False)

        # Again confirm this does not raise.
        ts_graph = TimeSeriesCausalGraph(input_list=input_nodes, output_list=output_nodes, fully_connected=True)

        # Now add a bad node and it should fail.
        input_nodes.append('A future(n=1)')
        with self.assertRaises(ValueError):
            ts_graph = TimeSeriesCausalGraph(input_list=input_nodes, output_list=output_nodes, fully_connected=True)

    def test_time_series_nodes(self):
        tsnode = TimeSeriesNode(variable_name='X1', time_lag=-1)
        tsnode_1 = TimeSeriesNode('X2 lag(n=1)')

        self.assertEqual(tsnode.time_lag, -1)
        self.assertEqual(tsnode_1.time_lag, -1)

        # test with future lag
        tsnode = TimeSeriesNode(variable_name='X1', time_lag=3)
        tsnode_1 = TimeSeriesNode('X2 future(n=3)')
        self.assertEqual(tsnode.time_lag, 3)
        self.assertEqual(tsnode_1.time_lag, 3)

    def test_add_node(self):
        ts_cg = TimeSeriesCausalGraph()
        ts_cg.add_node('X1 lag(n=1)')
        self.assertEqual(ts_cg.get_node('X1 lag(n=1)').time_lag, -1)
        self.assertEqual(ts_cg.get_node('X1 lag(n=1)').variable_name, 'X1')
        self.assertEqual(ts_cg.get_node('X1 lag(n=1)').identifier, 'X1 lag(n=1)')

        ts_cg.add_node(variable_name='X1', time_lag=-2)
        self.assertEqual(ts_cg.get_node('X1 lag(n=2)').time_lag, -2)
        self.assertEqual(ts_cg.get_node('X1 lag(n=2)').variable_name, 'X1')
        self.assertEqual(ts_cg.get_node('X1 lag(n=2)').identifier, 'X1 lag(n=2)')

        node = ts_cg.add_node('X2')
        self.assertIsInstance(node, TimeSeriesNode)
        self.assertEqual(node.time_lag, 0)
        self.assertEqual(node.variable_name, 'X2')
        self.assertEqual(node.identifier, 'X2')

        node = ts_cg.add_node('X3 future(n=2)')
        self.assertIsInstance(node, TimeSeriesNode)
        self.assertEqual(node.time_lag, 2)
        self.assertEqual(node.variable_name, 'X3')
        self.assertEqual(node.identifier, 'X3 future(n=2)')

        node = ts_cg.add_node(variable_name='X3', time_lag=3)
        self.assertIsInstance(node, TimeSeriesNode)
        self.assertEqual(node.time_lag, 3)
        self.assertEqual(node.variable_name, 'X3')
        self.assertEqual(node.identifier, 'X3 future(n=3)')

        node = ts_cg.add_node(variable_name='X4', time_lag=0)
        self.assertIsInstance(node, TimeSeriesNode)
        self.assertEqual(node.time_lag, 0)
        self.assertEqual(node.variable_name, 'X4')
        self.assertEqual(node.identifier, 'X4')

        # check equality
        ts_cg2 = TimeSeriesCausalGraph()
        node2 = ts_cg2.add_node(None, 'X1', -1)
        self.assertEqual(ts_cg.get_node('X1 lag(n=1)'), node2)
        self.assertNotEqual(node, node2)

        cg = CausalGraph()
        node = cg.add_node('X1 lag(n=1)')
        self.assertNotEqual(node, node2)  # different types so should be unequal

    def test_add_edge(self):
        ts_cg = TimeSeriesCausalGraph()
        ts_cg.add_edge('X1 lag(n=1)', 'X2 lag(n=1)', edge_type=EdgeType.DIRECTED_EDGE)

        # test that nodes are added correctly
        self.assertEqual(ts_cg.get_node('X1 lag(n=1)').time_lag, -1)
        self.assertEqual(ts_cg.get_node('X2 lag(n=1)').time_lag, -1)

        # confirm we cannot add an edge going into the past.
        with self.assertRaises(ValueError):
            ts_cg.add_edge('X1 lag(n=1)', 'X1 lag(n=2)')
        with self.assertRaises(ValueError):
            ts_cg.add_edge('X1 lag(n=1)', 'X2 lag(n=2)', edge_type=EdgeType.DIRECTED_EDGE)
        with self.assertRaises(ValueError):
            ts_cg.add_edge('X1 future(n=1)', 'X3')

    def test_replace_node(self):
        ts_cg = TimeSeriesCausalGraph()
        ts_cg.add_edge('X1 lag(n=1)', 'X2 lag(n=1)', edge_type=EdgeType.DIRECTED_EDGE)

        ts_cg.replace_node('X1 lag(n=1)', 'X1 lag(n=2)')
        self.assertEqual(ts_cg.get_node('X1 lag(n=2)').time_lag, -2)
        # check 'X1 lag(n=1)' does not exist
        self.assertFalse(ts_cg.node_exists('X1 lag(n=1)'))

    def test_get_variable_name_and_lag(self):
        # create a bad node name
        bad_node_name = 'X1 lag(n=1) lag(n=1)'
        with self.assertRaises(ValueError):
            get_variable_name_and_lag(bad_node_name)

        # create a bad node name
        bad_node_name = 'X1 lag(n=1) future(n=1)'
        with self.assertRaises(ValueError):
            get_variable_name_and_lag(bad_node_name)

        # create a correct node name
        node_name = 'X1 lag(n=1)'
        name, lag = get_variable_name_and_lag(node_name)
        self.assertEqual(name, 'X1')
        self.assertEqual(lag, -1)

    def test_extract_names_and_lags(self):
        nodes, maxlag = extract_names_and_lags(self.nodes)
        # sort the nodes by value
        nodes = sorted(nodes, key=lambda x: list(x.values())[0])
        self.assertEqual(nodes, [{'X1': -1}, {'X2': -1}, {'X3': -1}, {'X1': 0}, {'X2': 0}, {'X3': 0}])
        self.assertEqual(maxlag, -1)

    def test_serialization(self):
        for tscg in [
            self.tsdag,
            self.tsdag_1,
            self.tsdag_2,
            self.tsdag_3,
            self.ground_truth_minimal_graph_1,
            self.ground_truth_minimal_graph_3,
        ]:
            graph_as_dict = tscg.to_dict()
            graph_as_dict_2 = dict(tscg)
            self.assertDictEqual(graph_as_dict, graph_as_dict_2)
            self.assertIsInstance(json.dumps(graph_as_dict), str)

            reconstruction = TimeSeriesCausalGraph.from_dict(tscg.to_dict())

            # Check that their dict representations are the same.
            self.assertDictEqual(tscg.to_dict(), reconstruction.to_dict())

            # Also confirm that equality method works.
            self.assertEqual(tscg, reconstruction)

        # test with include_metadata=False
        graph_as_dict_nometa = self.tsdag.to_dict(include_meta=False)
        self.assertNotIn('meta', graph_as_dict_nometa['nodes']['X3 lag(n=1)'].keys())
        # confirm node info is still correct on reconstruction
        self.assertEqual(TimeSeriesCausalGraph.from_dict(graph_as_dict_nometa).get_node('X3 lag(n=1)').time_lag, -1)
        self.assertEqual(
            TimeSeriesCausalGraph.from_dict(graph_as_dict_nometa).get_node('X3 lag(n=1)').variable_name, 'X3'
        )

        graph_as_dict_withmeta = self.tsdag.to_dict(include_meta=True)
        self.assertIn('meta', graph_as_dict_withmeta['nodes']['X3 lag(n=1)'].keys())
        # confirm node info is still correct on reconstruction
        self.assertEqual(TimeSeriesCausalGraph.from_dict(graph_as_dict_withmeta).get_node('X3 lag(n=1)').time_lag, -1)
        self.assertEqual(
            TimeSeriesCausalGraph.from_dict(graph_as_dict_withmeta).get_node('X3 lag(n=1)').variable_name, 'X3'
        )

        # test with a custom metadata
        newg = self.tsdag.copy()
        newg.add_node('xm future(n=2)', variable_type=NodeVariableType.CONTINUOUS, meta={'test': 'test'})
        graph_as_dict_withmeta = newg.to_dict(include_meta=True)
        # test that the metadata is in the dict
        self.assertIn('test', graph_as_dict_withmeta['nodes']['xm future(n=2)']['meta'].keys())
        # confirm node info is still correct on reconstruction
        self.assertEqual(TimeSeriesCausalGraph.from_dict(graph_as_dict_withmeta).get_node('xm future(n=2)').time_lag, 2)
        self.assertEqual(
            TimeSeriesCausalGraph.from_dict(graph_as_dict_withmeta).get_node('xm future(n=2)').variable_name, 'xm'
        )

        graph_as_dict_nometa = newg.to_dict(include_meta=False)
        # test that the metadata is not in the dict
        self.assertNotIn('meta', graph_as_dict_nometa['nodes']['xm future(n=2)'].keys())
        # confirm node info is still correct on reconstruction
        self.assertEqual(TimeSeriesCausalGraph.from_dict(graph_as_dict_nometa).get_node('xm future(n=2)').time_lag, 2)
        self.assertEqual(
            TimeSeriesCausalGraph.from_dict(graph_as_dict_nometa).get_node('xm future(n=2)').variable_name, 'xm'
        )

    def test_from_causal_graph(self):
        # dag
        self.assertEqual(
            self.tsdag.identifier,
            '<X1 lag(n=1)>_<X2 lag(n=1)>_<X3 lag(n=1)>_<X1>_<X2>_<X3>',
        )
        self.assertNotEqual(self.tsdag, self.dag)

        # dag_1
        self.assertEqual(
            self.tsdag_1.identifier,
            '<X1 lag(n=2)>_<X2 lag(n=2)>_<X3 lag(n=2)>_<X1 lag(n=1)>_<X2 lag(n=1)>_<X3 lag(n=1)>_<X2>_<X3>_<X1>',
        )
        self.assertNotEqual(self.tsdag_1, self.dag_1)

        # Bad causal graph that has arrows going back in time. Confirm it raises when we try to build TS graph.
        cg = CausalGraph()
        nodes = ['A', 'B', 'A lag(n=1)', 'B future(n=1)']
        cg.add_nodes_from(nodes)
        cg.add_edge('A lag(n=1)', 'A')
        cg.add_edge('B future(n=1)', 'B')
        with self.assertRaises(ValueError):
            TimeSeriesCausalGraph.from_causal_graph(cg)

        # test with mixed graphs with random edges
        cg = CausalGraph()
        cg.add_nodes_from(self.nodes)
        added_edges = 0
        while added_edges < 5:
            # add edge with random type
            source = self.rng.choice(self.nodes)
            destination = self.rng.choice(self.nodes)
            vns, lags = get_variable_name_and_lag(source)
            vnd, lagd = get_variable_name_and_lag(destination)
            edge_type = self.rng.choice(list(EdgeType.__members__.values()))
            if edge_type == EdgeType.DIRECTED_EDGE and lags > lagd:
                # swap source and destination
                source, destination = destination, source

            # check for edge existence and self loops
            if not cg.edge_exists(source, destination) and not source == destination:
                cg.add_edge(source, destination, edge_type=edge_type)
                added_edges += 1

        # it should not raise
        tscg = TimeSeriesCausalGraph.from_causal_graph(cg)

        # check it preserves the edge types
        for edge in cg.edges:
            self.assertEqual(tscg.get_edge(edge.source, edge.destination).get_edge_type(), edge.get_edge_type())

    def test_from_adjacency_matrix(self):
        # test with the adjacency matrix corresponding to th minimal tsdag
        mg = self.tsdag.get_minimal_graph()
        tsdag = TimeSeriesCausalGraph.from_adjacency_matrix(mg.adjacency_matrix, mg.get_node_names())
        self.assertEqual(tsdag, mg)

    def test_from_adjacency_matrices(self):
        # The minimal tsdag has maxlag = 1, so we need two adjacency matrices one for lag 0 and one for lag 1.
        mg = self.tsdag.get_minimal_graph()
        # extract the adjacency matrices from the adjacency matrix of the minimal graph
        intra_indices = [0, 2, 4]  # 'X1', 'X2', 'X3'
        lagged_indices = [1, 3, 5]  # 'X1 lag(n=1)', 'X2 lag(n=1)', 'X3 lag(n=1)'
        full_adj_mat = mg.adjacency_matrix
        adj_mat_lag_0 = full_adj_mat[intra_indices, :][:, intra_indices]
        adj_mat_lag_1 = full_adj_mat[lagged_indices, :][:, intra_indices]

        matrices = {0: adj_mat_lag_0, -1: adj_mat_lag_1}

        variables = ['X1', 'X2', 'X3']
        tsdag = TimeSeriesCausalGraph.from_adjacency_matrices(matrices, variables)
        self.assertEqual(tsdag, mg)

        # test the attribute adjacency_matrices

        # get the adjacency matrices from the tsdag
        adj_matrices = self.tsdag.adjacency_matrices
        for key, value in adj_matrices.items():
            numpy.testing.assert_equal(value, matrices[key])

        # test without time delta 0
        matrices = {-1: adj_mat_lag_1}
        variables = ['X1', 'X2', 'X3']
        tsdag = TimeSeriesCausalGraph.from_adjacency_matrices(matrices, variables)
        self.assertEqual(len(tsdag.edges), adj_mat_lag_1.sum())

        # test with mismatch sizes
        matrices = {0: numpy.eye(3), -1: numpy.eye(4)}
        variables = ['X1', 'X2', 'X3']
        with self.assertRaises(AssertionError):
            _ = TimeSeriesCausalGraph.from_adjacency_matrices(matrices, variables)

    def test_summary_graph(self):
        summary_graph = self.tsdag.get_summary_graph()
        # the graph should be  X3 <- X1 -> X2
        self.assertEqual(summary_graph, self.ground_truth_summary_graph)

        # dag_1
        summary_graph_1 = self.tsdag_1.get_summary_graph()
        # the graph should be X2 <- X3 -> X1
        self.assertEqual(summary_graph_1.identifier, '<X3>_<X1>_<X2>')
        self.assertEqual(summary_graph_1, self.ground_truth_summary_graph_1)

        # test it fails when it is a DAG
        for edge_type in EdgeType.__members__:
            tscg = self.tsdag_1.copy()
            tscg.change_edge_type('X1 lag(n=1)', 'X1', new_edge_type=edge_type)

            with self.assertRaises(AssertionError):
                _ = tscg.get_summary_graph()

    def test_variable_names(self):
        variables = self.tsdag.variables
        self.assertEqual(variables, ['X1', 'X2', 'X3'])

    def test_is_minimal(self):
        # dag
        self.assertTrue(self.tsdag.is_minimal_graph())
        # dag_1
        self.assertFalse(self.tsdag_1.is_minimal_graph())

    def test_get_minimal_graph(self):
        # dag
        minimal_graph = self.tsdag.get_minimal_graph()
        self.assertEqual(minimal_graph, self.ground_truth_minimal_graph)

        # dag_1
        minimal_graph_1 = self.tsdag_1.get_minimal_graph()
        self.assertEqual(minimal_graph_1, self.ground_truth_minimal_graph_1)

        # dag 2
        mg = self.tsdag_2.get_minimal_graph()

        # they should be equal as it is already a minimal graph
        self.assertEqual(mg, self.tsdag_2)

        # dag 3
        mg = self.tsdag_3.get_minimal_graph()
        self.assertEqual(mg, self.ground_truth_minimal_graph_3)

        # dag 4
        # test with a graph that is not aligned at lag 0
        tsdag_4 = TimeSeriesCausalGraph()
        tsdag_4.add_edge('X1 lag(n=2)', 'X1 lag(n=1)')

        mg = tsdag_4.get_minimal_graph()
        ground_truth_mg = TimeSeriesCausalGraph()
        ground_truth_mg.add_edge('X1 lag(n=1)', 'X1')
        self.assertEqual(mg, ground_truth_mg)

        # TODO: CAUSALAI-3397 - Need to test with CPGAG, MAG, and PAGs
        for edge_type in EdgeType.__members__:
            tscg = TimeSeriesCausalGraph()
            tscg.add_edge('x', 'x future(n=2)', edge_type=edge_type)
            self.assertFalse(tscg.is_minimal_graph())

            true_minimal = TimeSeriesCausalGraph()
            true_minimal.add_edge('x lag(n=2)', 'x', edge_type=edge_type)
            self.assertTrue(true_minimal.is_minimal_graph())

            self.assertEqual(tscg.get_minimal_graph(), true_minimal)
            self.assertTrue(tscg.get_minimal_graph().is_minimal_graph())

    def test_extend_backward(self):
        # with 1 steps
        # dag
        extended_dag = self.tsdag.extend_graph(backward_steps=1)

        # create the extended graph ground truth
        extended_graph = TimeSeriesCausalGraph()
        extended_graph.add_nodes_from(['X1', 'X2', 'X3'])
        extended_graph.add_edge('X1', 'X3', edge_type=EdgeType.DIRECTED_EDGE)
        extended_graph.add_edge('X1 lag(n=1)', 'X1', edge_type=EdgeType.DIRECTED_EDGE)
        extended_graph.add_edge('X2 lag(n=1)', 'X2', edge_type=EdgeType.DIRECTED_EDGE)
        extended_graph.add_edge('X3 lag(n=1)', 'X3', edge_type=EdgeType.DIRECTED_EDGE)
        extended_graph.add_edge('X1 lag(n=1)', 'X2', edge_type=EdgeType.DIRECTED_EDGE)
        extended_graph.add_edge('X1 lag(n=1)', 'X3 lag(n=1)', edge_type=EdgeType.DIRECTED_EDGE)

        self.assertEqual(extended_dag, extended_graph)

        # dag_1
        extended_dag_1 = self.tsdag_1.extend_graph(backward_steps=1)

        # create the extended graph
        extended_graph_1 = TimeSeriesCausalGraph()
        extended_graph_1.add_nodes_from(['X1', 'X2', 'X3'])
        extended_graph_1.add_edge('X3', 'X1', edge_type=EdgeType.DIRECTED_EDGE)
        extended_graph_1.add_edge('X1 lag(n=1)', 'X1', edge_type=EdgeType.DIRECTED_EDGE)
        extended_graph_1.add_edge('X2 lag(n=1)', 'X2', edge_type=EdgeType.DIRECTED_EDGE)
        extended_graph_1.add_edge('X3 lag(n=1)', 'X3', edge_type=EdgeType.DIRECTED_EDGE)
        extended_graph_1.add_edge('X3 lag(n=1)', 'X1', edge_type=EdgeType.DIRECTED_EDGE)
        extended_graph_1.add_edge('X3 lag(n=2)', 'X2', edge_type=EdgeType.DIRECTED_EDGE)
        extended_graph_1.add_edge('X3 lag(n=1)', 'X1 lag(n=1)', edge_type=EdgeType.DIRECTED_EDGE)

        self.assertEqual(extended_dag_1, extended_graph_1)

        # with 2 steps
        # dag
        extended_dag = self.tsdag.extend_graph(backward_steps=2)

        # create the extended graph from the previous extended graph
        extended_graph.add_edge('X1 lag(n=2)', 'X1 lag(n=1)', edge_type=EdgeType.DIRECTED_EDGE)
        extended_graph.add_edge('X2 lag(n=2)', 'X2 lag(n=1)', edge_type=EdgeType.DIRECTED_EDGE)
        extended_graph.add_edge('X3 lag(n=2)', 'X3 lag(n=1)', edge_type=EdgeType.DIRECTED_EDGE)
        extended_graph.add_edge('X1 lag(n=2)', 'X2 lag(n=1)', edge_type=EdgeType.DIRECTED_EDGE)
        extended_graph.add_edge('X1 lag(n=2)', 'X3 lag(n=2)', edge_type=EdgeType.DIRECTED_EDGE)

        self.assertEqual(extended_dag, extended_graph)

        # dag_1
        extended_dag_1 = self.tsdag_1.extend_graph(backward_steps=2)

        # create the extended graph from the previous extended graph
        extended_graph_1.add_edge('X3 lag(n=2)', 'X1 lag(n=1)', edge_type=EdgeType.DIRECTED_EDGE)
        extended_graph_1.add_edge('X3 lag(n=2)', 'X3 lag(n=1)', edge_type=EdgeType.DIRECTED_EDGE)
        extended_graph_1.add_edge('X3 lag(n=2)', 'X1 lag(n=2)', edge_type=EdgeType.DIRECTED_EDGE)
        extended_graph_1.add_edge('X1 lag(n=2)', 'X1 lag(n=1)', edge_type=EdgeType.DIRECTED_EDGE)
        extended_graph_1.add_edge('X2 lag(n=2)', 'X2 lag(n=1)', edge_type=EdgeType.DIRECTED_EDGE)

        self.assertEqual(extended_dag_1, extended_graph_1)

        # dag 2
        extended_dag_2 = self.tsdag_2.extend_graph(backward_steps=1)

        gr_ext_dag_2 = self.tsdag_2.copy()
        # add floating node
        gr_ext_dag_2.add_node('X1 lag(n=1)')

        self.assertEqual(extended_dag_2, gr_ext_dag_2)

        # test with other types of edges
        tscg1 = self.ground_truth_minimal_graph_1.copy()

        # change randomly the edge types of the graph
        for edge in tscg1.get_edges():
            edge._edge_type = self.rng.choice(list(EdgeType.__members__.values()))

        extended_tscg1 = tscg1.extend_graph(backward_steps=2)

        # create the extended graph from the previous extended graph
        for edge in extended_tscg1.edges:
            vars = edge.source.variable_name
            vard = edge.destination.variable_name
            time_delta = edge.source.time_lag - edge.destination.time_lag

            # check if the corresponding edge in the minimal graph has same edge type
            new_source = get_name_with_lag(vars, time_delta)
            new_destination = get_name_with_lag(vard, 0)

            self.assertEqual(tscg1.get_edge(new_source, new_destination).get_edge_type(), edge.get_edge_type())

    def test_extend_graph_forward(self):
        # with 1 steps
        # dag
        extended_dag = self.tsdag.extend_graph(forward_steps=1)

        # create the extended graph by extending the minimal graph
        extended_graph = self.ground_truth_minimal_graph.copy()

        extended_graph.add_edge('X1', 'X1 future(n=1)', edge_type=EdgeType.DIRECTED_EDGE)
        extended_graph.add_edge('X2', 'X2 future(n=1)', edge_type=EdgeType.DIRECTED_EDGE)
        extended_graph.add_edge('X3', 'X3 future(n=1)', edge_type=EdgeType.DIRECTED_EDGE)
        extended_graph.add_edge('X1', 'X2 future(n=1)', edge_type=EdgeType.DIRECTED_EDGE)
        extended_graph.add_edge('X1 future(n=1)', 'X3 future(n=1)', edge_type=EdgeType.DIRECTED_EDGE)

        self.assertEqual(extended_dag, extended_graph)

        # dag_1
        extended_dag_1 = self.tsdag_1.extend_graph(forward_steps=1)

        # create the extended graph by extending the minimal graph
        extended_graph_1 = self.ground_truth_minimal_graph_1.copy()

        extended_graph_1.add_edge('X3', 'X1 future(n=1)', edge_type=EdgeType.DIRECTED_EDGE)
        extended_graph_1.add_edge('X3', 'X3 future(n=1)', edge_type=EdgeType.DIRECTED_EDGE)
        extended_graph_1.add_edge('X1', 'X1 future(n=1)', edge_type=EdgeType.DIRECTED_EDGE)
        extended_graph_1.add_edge('X2', 'X2 future(n=1)', edge_type=EdgeType.DIRECTED_EDGE)
        extended_graph_1.add_edge('X3 future(n=1)', 'X1 future(n=1)', edge_type=EdgeType.DIRECTED_EDGE)
        extended_graph_1.add_edge('X3 lag(n=1)', 'X2 future(n=1)', edge_type=EdgeType.DIRECTED_EDGE)

        self.assertEqual(extended_dag_1, extended_graph_1)

        # dag 3
        extended_dag_3 = self.tsdag_3.extend_graph(forward_steps=1)

        # create the extended graph by extending the minimal graph
        extended_graph_3 = self.ground_truth_minimal_graph_3.copy()

        extended_graph_3.add_edge('A', 'A future(n=1)', edge_type=EdgeType.DIRECTED_EDGE)
        extended_graph_3.add_edge('B', 'B future(n=1)', edge_type=EdgeType.DIRECTED_EDGE)
        extended_graph_3.add_edge('A future(n=1)', 'C future(n=1)', edge_type=EdgeType.DIRECTED_EDGE)
        extended_graph_3.add_edge('B future(n=1)', 'C future(n=1)', edge_type=EdgeType.DIRECTED_EDGE)
        extended_graph_3.add_edge('C future(n=1)', 'D future(n=1)', edge_type=EdgeType.DIRECTED_EDGE)
        extended_graph_3.add_edge('C future(n=1)', 'E future(n=1)', edge_type=EdgeType.DIRECTED_EDGE)
        extended_graph_3.add_edge('D future(n=1)', 'E future(n=1)', edge_type=EdgeType.DIRECTED_EDGE)
        extended_graph_3.add_edge('E future(n=1)', 'F future(n=1)', edge_type=EdgeType.DIRECTED_EDGE)
        extended_graph_3.add_edge('F', 'F future(n=1)', edge_type=EdgeType.DIRECTED_EDGE)

        self.assertEqual(extended_dag_3, extended_graph_3)

        # test with 2 steps
        extended_graph_3 = extended_graph_3.copy()
        extended_dag_3 = self.tsdag_3.extend_graph(forward_steps=2)

        extended_graph_3.add_edge('A future(n=1)', 'A future(n=2)', edge_type=EdgeType.DIRECTED_EDGE)
        extended_graph_3.add_edge('B future(n=1)', 'B future(n=2)', edge_type=EdgeType.DIRECTED_EDGE)
        extended_graph_3.add_edge('A future(n=2)', 'C future(n=2)', edge_type=EdgeType.DIRECTED_EDGE)
        extended_graph_3.add_edge('B future(n=2)', 'C future(n=2)', edge_type=EdgeType.DIRECTED_EDGE)
        extended_graph_3.add_edge('C future(n=2)', 'D future(n=2)', edge_type=EdgeType.DIRECTED_EDGE)
        extended_graph_3.add_edge('C future(n=2)', 'E future(n=2)', edge_type=EdgeType.DIRECTED_EDGE)
        extended_graph_3.add_edge('D future(n=2)', 'E future(n=2)', edge_type=EdgeType.DIRECTED_EDGE)
        extended_graph_3.add_edge('E future(n=2)', 'F future(n=2)', edge_type=EdgeType.DIRECTED_EDGE)
        extended_graph_3.add_edge('F future(n=1)', 'F future(n=2)', edge_type=EdgeType.DIRECTED_EDGE)

        self.assertEqual(extended_dag_3, extended_graph_3)

        # test with other types of edges
        tscg3 = self.ground_truth_minimal_graph_3.copy()

        # change randomly the edge types of the graph
        for edge in tscg3.get_edges():
            edge._edge_type = self.rng.choice(list(EdgeType.__members__.values()))

        extended_tsdag_3 = tscg3.extend_graph(forward_steps=1)

        tscg3.add_edge('A', 'A future(n=1)', edge_type=tscg3.get_edge('A lag(n=1)', 'A').get_edge_type())
        tscg3.add_edge('B', 'B future(n=1)', edge_type=tscg3.get_edge('B lag(n=1)', 'B').get_edge_type())
        tscg3.add_edge('A future(n=1)', 'C future(n=1)', edge_type=tscg3.get_edge('A', 'C').get_edge_type())
        tscg3.add_edge('B future(n=1)', 'C future(n=1)', edge_type=tscg3.get_edge('B', 'C').get_edge_type())
        tscg3.add_edge('C future(n=1)', 'D future(n=1)', edge_type=tscg3.get_edge('C', 'D').get_edge_type())
        tscg3.add_edge('C future(n=1)', 'E future(n=1)', edge_type=tscg3.get_edge('C', 'E').get_edge_type())
        tscg3.add_edge('D future(n=1)', 'E future(n=1)', edge_type=tscg3.get_edge('D', 'E').get_edge_type())
        tscg3.add_edge('E future(n=1)', 'F future(n=1)', edge_type=tscg3.get_edge('E', 'F').get_edge_type())
        tscg3.add_edge('F', 'F future(n=1)', edge_type=tscg3.get_edge('F lag(n=1)', 'F').get_edge_type())

        self.assertEqual(extended_tsdag_3, tscg3)

    def test_add_time_edge(self):
        # test adding a time edge to a graph
        tsgraph = TimeSeriesCausalGraph()
        tsgraph.add_time_edge('A', -1, 'B', 2)

        # create the ground truth graph
        ground_truth_graph = TimeSeriesCausalGraph()
        ground_truth_graph.add_edge('A lag(n=1)', 'B future(n=2)', edge_type=EdgeType.DIRECTED_EDGE)
        self.assertEqual(tsgraph, ground_truth_graph)

        tsgraph = TimeSeriesCausalGraph()
        tsgraph.add_time_edge('A', 0, 'B', 0)
        ground_truth_graph = TimeSeriesCausalGraph()
        ground_truth_graph.add_edge('A', 'B', edge_type=EdgeType.DIRECTED_EDGE)
        self.assertEqual(tsgraph, ground_truth_graph)

    def test_is_stationary(self):

        tscg = TimeSeriesCausalGraph()
        tscg.add_edge('X1 lag(n=2)', 'X1 lag(n=1)')
        tscg.add_edge('X2 lag(n=2)', 'X2 lag(n=1)')
        tscg.add_edge('X1 lag(n=1)', 'X2 lag(n=1)')
        tscg.add_edge('X1', 'X2')
        tscg.add_edge('X2 lag(n=1)', 'X2')

        self.assertFalse(tscg.is_stationary_graph())

        # now make it stationary manually
        tscg.add_edge('X1 lag(n=2)', 'X2 lag(n=2)')
        tscg.add_edge('X1 lag(n=1)', 'X1')

        self.assertTrue(tscg.is_stationary_graph())

    def test_make_stationary(self):

        tscg = TimeSeriesCausalGraph()
        tscg.add_edge('X1 lag(n=2)', 'X1 lag(n=1)')
        tscg.add_edge('X2 lag(n=2)', 'X2 lag(n=1)')
        tscg.add_edge('X1 lag(n=1)', 'X2 lag(n=1)')
        tscg.add_edge('X1', 'X2')
        tscg.add_edge('X2 lag(n=1)', 'X2')

        self.assertFalse(tscg.is_stationary_graph())

        stat_tscg = tscg.get_stationary_graph()

        self.assertTrue(stat_tscg.is_stationary_graph())

    def test_to_numpy_by_lag(self):
        # test with empty graph
        tsdag = TimeSeriesCausalGraph()
        adj_matrices, variables = tsdag.to_numpy_by_lag()
        self.assertEqual(len(adj_matrices), 0)
        self.assertEqual(len(variables), 0)

        # test with a simple graph
        adj_matrices, variables = self.tsdag_1.to_numpy_by_lag()
        self.assertEqual(len(adj_matrices), 3)
        self.assertEqual(len(variables), 3)
        self.assertEqual(adj_matrices[0].shape, (3, 3))
        self.assertEqual(adj_matrices[-1].shape, (3, 3))
        self.assertEqual(adj_matrices[-2].shape, (3, 3))


class TestTimeSeriesCausalGraphPrinting(unittest.TestCase):
    def test_default_nodes_and_edges(self):
        cg = TimeSeriesCausalGraph()

        n = cg.add_node('a')
        e = cg.add_edge('a', 'b')

        self.assertIsInstance(n.__hash__(), int)
        self.assertIsInstance(e.__hash__(), int)
        self.assertIsInstance(cg.__hash__(), int)

        self.assertIsInstance(n.__repr__(), str)
        self.assertIsInstance(e.__repr__(), str)
        self.assertIsInstance(cg.__repr__(), str)
        self.assertTrue(n.__repr__().startswith('TimeSeriesNode'))
        self.assertTrue(e.__repr__().startswith('Edge'))
        self.assertTrue(cg.__repr__().startswith('TimeSeriesCausalGraph'))

        self.assertIsInstance(n.details(), str)
        self.assertIsInstance(e.details(), str)
        self.assertIsInstance(cg.details(), str)
        self.assertTrue(n.details().startswith('TimeSeriesNode'))
        self.assertTrue(e.details().startswith('Edge'))
        self.assertTrue(cg.details().startswith('TimeSeriesCausalGraph'))

    def test_complex_nodes_and_edges(self):
        cg = TimeSeriesCausalGraph()

        n = cg.add_node('a lag(n=1)')
        e = cg.add_edge('a lag(n=1)', 'b', edge_type=EdgeType.BIDIRECTED_EDGE)

        self.assertIsInstance(n.__hash__(), int)
        self.assertIsInstance(e.__hash__(), int)
        self.assertIsInstance(cg.__hash__(), int)

        self.assertIsInstance(n.__repr__(), str)
        self.assertIsInstance(e.__repr__(), str)
        self.assertIsInstance(cg.__repr__(), str)
        self.assertTrue(n.__repr__().startswith('TimeSeriesNode'))
        self.assertTrue(e.__repr__().startswith('Edge'))
        self.assertTrue(cg.__repr__().startswith('TimeSeriesCausalGraph'))

        self.assertIsInstance(n.details(), str)
        self.assertIsInstance(e.details(), str)
        self.assertIsInstance(cg.details(), str)
        self.assertTrue(n.details().startswith('TimeSeriesNode'))
        self.assertTrue(e.details().startswith('Edge'))
        self.assertTrue(cg.details().startswith('TimeSeriesCausalGraph'))

    def test_add_node_from_node(self):
        identifier = 'apple lag(n=1)'
        cg = TimeSeriesCausalGraph()
        cg.add_node(identifier=identifier, meta={'color': 'blue'}, variable_type=NodeVariableType.BINARY)

        node = cg.get_node(identifier=identifier)
        self.assertIsInstance(node, TimeSeriesNode)

        cg2 = TimeSeriesCausalGraph()
        cg2.add_node(node=node)

        node2 = cg2.get_node(identifier=identifier)
        self.assertIsInstance(node2, TimeSeriesNode)

        self.assertEqual(node.variable_type, node2.variable_type)
        self.assertEqual(-1, node.time_lag)
        self.assertEqual(node.time_lag, node2.time_lag)
        self.assertEqual('apple', node.variable_name)
        self.assertEqual(node.variable_name, node2.variable_name)
        self.assertDictEqual(node.metadata, node2.metadata)

        node3 = cg.add_node(None, 'apple', 1)
        self.assertIsInstance(node3, TimeSeriesNode)

        cg2.add_node(node=node3)

        node4 = cg2.get_node(identifier='apple future(n=1)')
        self.assertIsInstance(node2, TimeSeriesNode)

        self.assertEqual(node3.variable_type, node4.variable_type)
        self.assertEqual(1, node3.time_lag)
        self.assertEqual(node3.time_lag, node4.time_lag)
        self.assertEqual('apple', node3.variable_name)
        self.assertEqual(node3.variable_name, node4.variable_name)
        self.assertDictEqual(node3.metadata, node4.metadata)

    def test_add_node_from_node_deepcopies(self):
        class DummyObj:
            pass

        identifier = 'apple lag(n=1)'
        cg = TimeSeriesCausalGraph()
        node = cg.add_node(identifier=identifier, meta={'dummy': DummyObj()})
        self.assertIsInstance(node, TimeSeriesNode)

        cg2 = TimeSeriesCausalGraph()
        node2 = cg2.add_node(node=node)
        self.assertIsInstance(node2, TimeSeriesNode)

        self.assertIsInstance(node.metadata['dummy'], DummyObj)
        self.assertIsInstance(node2.metadata['dummy'], DummyObj)
        self.assertNotEqual(id(node.metadata['dummy']), id(node2.metadata['dummy']))

    def test_add_node_from_node_raises(self):
        identifier = 'apple lag(n=1)'
        cg = TimeSeriesCausalGraph()
        node = cg.add_node(identifier=identifier)

        with self.assertRaises(AssertionError):
            cg.add_node()

        with self.assertRaises(AssertionError):
            cg.add_node(None, None, None)

        with self.assertRaises(AssertionError):
            cg.add_node(None, 'apple', None)

        with self.assertRaises(AssertionError):
            cg.add_node(None, None, -2)

        with self.assertRaises(AssertionError):
            cg.add_node(identifier, 'apple', None)

        with self.assertRaises(AssertionError):
            cg.add_node('apple lag(n=2)', None, -2)

        with self.assertRaises(AssertionError):
            cg.add_node('apple lag(n=2)', 'apple', -2)

        with self.assertRaises(CausalGraphErrors.NodeDuplicatedError):
            cg.add_node(node=node)

        with self.assertRaises(CausalGraphErrors.NodeDuplicatedError):
            cg.add_node(identifier)

        with self.assertRaises(CausalGraphErrors.NodeDuplicatedError):
            cg.add_node(None, 'apple', -1)

        with self.assertRaises(AssertionError):
            cg.add_node(node=node, identifier='b')

        with self.assertRaises(AssertionError):
            cg.add_node(node=node, meta={'foo': 'bar'})

        with self.assertRaises(AssertionError):
            cg.add_node(node=node, variable_type=NodeVariableType.BINARY)

        with self.assertRaises(AssertionError):
            cg.add_node(node=node, my_kwargs='foo')

    def test_node_variable_type(self):
        cg = TimeSeriesCausalGraph()

        cg.add_node('a', variable_type=NodeVariableType.CONTINUOUS)
        self.assertEqual(cg.get_node('a').variable_type, NodeVariableType.CONTINUOUS)

        cg.get_node('a').variable_type = NodeVariableType.MULTICLASS
        self.assertEqual(cg.get_node('a').variable_type, NodeVariableType.MULTICLASS)

        cg.get_node('a').variable_type = 'binary'
        self.assertEqual(cg.get_node('a').variable_type, NodeVariableType.BINARY)

        with self.assertRaises(ValueError):
            cg.get_node('a').variable_type = 'not_a_variable_type'

        # Test the above but directly through the constructor
        identifier = 'b future(n=1)'
        cg.add_node('b future(n=1)', variable_type='binary')
        self.assertEqual(cg.get_node(identifier).variable_type, NodeVariableType.BINARY)
        self.assertEqual(cg.get_node(identifier).variable_name, 'b')
        self.assertEqual(cg.get_node(identifier).time_lag, 1)

        with self.assertRaises(ValueError):
            cg.add_node('c', variable_type='not_a_variable_type')

    def test_node_repr(self):
        cg = TimeSeriesCausalGraph()
        cg.add_node('apple')
        cg.add_node('banana', variable_type=NodeVariableType.CONTINUOUS)
        cg.add_node('carrot lag(n=3)')
        cg.add_node('donut future(n=2)')

        self.assertEqual(cg['apple'], cg.get_node('apple'))
        self.assertEqual(cg.get_node('apple').variable_name, 'apple')
        self.assertEqual(cg.get_node('apple').time_lag, 0)
        self.assertEqual(cg['banana'], cg.get_node('banana'))
        self.assertEqual(cg.get_node('banana').variable_name, 'banana')
        self.assertEqual(cg.get_node('banana').time_lag, 0)
        self.assertEqual(cg['carrot lag(n=3)'], cg.get_node('carrot lag(n=3)'))
        self.assertEqual(cg.get_node('carrot lag(n=3)').variable_name, 'carrot')
        self.assertEqual(cg.get_node('carrot lag(n=3)').time_lag, -3)
        self.assertEqual(cg['donut future(n=2)'], cg.get_node('donut future(n=2)'))
        self.assertEqual(cg.get_node('donut future(n=2)').variable_name, 'donut')
        self.assertEqual(cg.get_node('donut future(n=2)').time_lag, 2)
        self.assertEqual(repr(cg['apple']), 'TimeSeriesNode("apple")')
        self.assertEqual(repr(cg['banana']), 'TimeSeriesNode("banana", type="continuous")')
        self.assertEqual(repr(cg['carrot lag(n=3)']), 'TimeSeriesNode("carrot lag(n=3)")')
        self.assertEqual(repr(cg['donut future(n=2)']), 'TimeSeriesNode("donut future(n=2)")')
