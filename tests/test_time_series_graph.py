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

from cai_causal_graph import CausalGraph, EdgeType, TimeSeriesCausalGraph
from cai_causal_graph.graph_components import TimeSeriesNode
from cai_causal_graph.utils import extract_names_and_lags, get_variable_name_and_lag


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

        ts_cg.add_node(variable_name='X1', time_lag=-2)
        self.assertEqual(ts_cg.get_node('X1 lag(n=2)').time_lag, -2)

    def test_add_edge(self):
        ts_cg = TimeSeriesCausalGraph()
        ts_cg.add_edge('X1 lag(n=1)', 'X2 lag(n=1)', edge_type=EdgeType.DIRECTED_EDGE)

        # test that nodes are added correctly
        self.assertEqual(ts_cg.get_node('X1 lag(n=1)').time_lag, -1)
        self.assertEqual(ts_cg.get_node('X2 lag(n=1)').time_lag, -1)

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
