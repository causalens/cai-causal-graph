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

import networkx
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

    def test_add_edges_from_paths(self):
        # try adding a single path
        ts_cg = TimeSeriesCausalGraph()
        ts_cg.add_edges_from_paths(['X1 lag(n=1)', 'X2 lag(n=1)', 'X2'])
        self.assertSetEqual(set([('X1 lag(n=1)', 'X2 lag(n=1)'), ('X2 lag(n=1)', 'X2')]), set(ts_cg.get_edge_pairs()))

        # try adding multiple paths
        ts_cg = TimeSeriesCausalGraph()
        ts_cg.add_edges_from_paths([['X1 lag(n=1)', 'X2 lag(n=1)', 'X2'], ['X2 lag(n=2)', 'X2', 'X3']])
        self.assertSetEqual(
            set([('X1 lag(n=1)', 'X2 lag(n=1)'), ('X2', 'X3'), ('X2 lag(n=1)', 'X2'), ('X2 lag(n=2)', 'X2')]),
            set(ts_cg.get_edge_pairs()),
        )

        # try adding an edge which does not respect time
        ts_cg = TimeSeriesCausalGraph()
        with self.assertRaises(ValueError):
            ts_cg.add_edges_from_paths(['X1 lag(n=1)', 'X2 lag(n=1)', 'X2 lag(n=2)'])

    def test_replace_node(self):
        ts_cg = TimeSeriesCausalGraph()
        ts_cg.add_edge('X1 lag(n=1)', 'X2 lag(n=1)', edge_type=EdgeType.DIRECTED_EDGE)

        ts_cg.replace_node('X1 lag(n=1)', 'X1 lag(n=2)')
        self.assertEqual(ts_cg.get_node('X1 lag(n=2)').time_lag, -2)
        # check 'X1 lag(n=1)' does not exist
        self.assertFalse(ts_cg.node_exists('X1 lag(n=1)'))

    def test_replace_edge(self):
        ts_cg = TimeSeriesCausalGraph()
        ts_cg.add_nodes_from(['X1', 'X2', 'X1 lag(n=1)', 'X2 lag(n=1)'])
        ts_cg.add_edge('X1', 'X2', edge_type=EdgeType.DIRECTED_EDGE)
        ts_cg.add_edge('X1 lag(n=1)', 'X2', edge_type=EdgeType.DIRECTED_EDGE)

        # check swapping of direction
        cg = ts_cg.copy()
        cg.replace_edge(source='X1', destination='X2', new_source='X2', new_destination='X1')
        self.assertTrue(cg.edge_exists('X2', 'X1'))
        self.assertFalse(cg.edge_exists('X1', 'X2'))

        # check adding another edge
        cg = ts_cg.copy()
        cg.replace_edge(source='X1 lag(n=1)', destination='X2', new_source='X2 lag(n=1)', new_destination='X2')
        self.assertTrue(cg.edge_exists('X2 lag(n=1)', 'X2'))
        self.assertFalse(cg.edge_exists('X1 lag(n=1)', 'X2'))

        # check it errors properly when the new edge flows backwards in time
        with self.assertRaises(ValueError):
            cg = ts_cg.copy()
            cg.replace_edge(source='X1 lag(n=1)', destination='X2', new_source='X2', new_destination='X1 lag(n=1)')

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

    def test_cg_to_tscg_serialization(self):

        # create a non-DAG graph
        cg = CausalGraph()
        cg.add_edge('X1', 'X2', edge_type=EdgeType.UNDIRECTED_EDGE)
        cg.add_edge('X2 lag(n=1)', 'X3', edge_type=EdgeType.BIDIRECTED_EDGE)

        # test it does not fail
        tscg1 = TimeSeriesCausalGraph.from_dict(cg.to_dict())

        for tscg in [
            self.tsdag,
            self.tsdag_1,
            self.tsdag_2,
            self.tsdag_3,
            self.ground_truth_minimal_graph_1,
            self.ground_truth_minimal_graph_3,
            tscg1,
        ]:
            cg = CausalGraph.from_dict(tscg.to_dict())
            ts_cg_back = TimeSeriesCausalGraph.from_dict(cg.to_dict())

            self.assertEqual(tscg, ts_cg_back)

    def test_tscg_to_cg_serialization(self):
        # create a non-DAG graph
        cg = CausalGraph()
        cg.add_edge('X1 lag(n=2)', 'X2', edge_type=EdgeType.UNDIRECTED_EDGE)
        cg.add_edge('X2', 'X3', edge_type=EdgeType.BIDIRECTED_EDGE)
        tscg1 = TimeSeriesCausalGraph.from_dict(cg.to_dict())

        for tscg in [
            self.tsdag,
            self.tsdag_1,
            self.tsdag_2,
            self.tsdag_3,
            self.ground_truth_minimal_graph_1,
            self.ground_truth_minimal_graph_3,
            tscg1,
        ]:
            cg = CausalGraph.from_dict(tscg.to_dict())

            # check same node identifiers
            self.assertListEqual([node.identifier for node in tscg.nodes], [node.identifier for node in cg.nodes])

            # check same edges
            for edge in tscg.edges:
                self.assertTrue(cg.edge_exists(edge.source, edge.destination))
                edge_cg = cg.get_edge(edge.source, edge.destination)
                self.assertEqual(edge.get_edge_type(), edge_cg.get_edge_type())

    def test_copy(self):
        # test copy preserves the correct class
        ts_cg = TimeSeriesCausalGraph()
        ts_cg.add_node('a', meta={'some': 'thing'})
        ts_cg.add_edge('a', 'b', meta={'foo': 'bar'})
        ts_cg.add_edge('a lag(n=1)', 'b', meta={'test': 'test'})
        ts_cg.add_edge('b lag(n=1)', 'b')
        ts_cg.add_edge('a', 'c', edge_type=EdgeType.BIDIRECTED_EDGE)

        ts_cg_copy = ts_cg.copy()
        self.assertIsInstance(ts_cg_copy, TimeSeriesCausalGraph)
        self.assertEqual(ts_cg_copy, ts_cg)
        self.assertTrue(ts_cg_copy.__eq__(ts_cg, deep=True))

        # No metadata
        ts_cg_copy_no_meta = ts_cg.copy(include_meta=False)
        self.assertIsInstance(ts_cg_copy_no_meta, TimeSeriesCausalGraph)

        self.assertEqual(ts_cg_copy_no_meta, ts_cg_copy)
        self.assertEqual(ts_cg_copy_no_meta, ts_cg)
        self.assertFalse(ts_cg_copy_no_meta.__eq__(ts_cg, deep=True))

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
            source_time_lag = get_variable_name_and_lag(edge.source)[1]
            destination_time_lag = get_variable_name_and_lag(edge.destination)[1]
            if source_time_lag > destination_time_lag:
                # swap source and destination
                source, destination = edge.destination, edge.source
            else:
                source, destination = edge.source, edge.destination
            self.assertEqual(tscg.get_edge(source, destination).get_edge_type(), edge.get_edge_type())

        # test if node meta are preserved
        cg = CausalGraph()
        cg.add_node('a', meta={'some': 'test'})
        cg.add_node('b', meta={'some': 'test2'})

        tscg = TimeSeriesCausalGraph.from_causal_graph(cg)
        self.assertDictEqual(tscg.get_node('a').meta, {'variable_name': 'a', 'time_lag': 0, 'some': 'test'})
        self.assertDictEqual(tscg.get_node('b').meta, {'variable_name': 'b', 'time_lag': 0, 'some': 'test2'})

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

        # test for floating nodes
        # create a matrix with just one edge
        matrices = {0: numpy.zeros((3, 3)), -1: numpy.zeros((3, 3))}
        matrices[-1][0, 1] = 1
        # there should be 3+1 nodes
        variables = ['X1', 'X2', 'X3']

        tsdag = TimeSeriesCausalGraph.from_adjacency_matrices(matrices, variables)  # will be minimal graph by default
        nodes = ['X2', 'X3', 'X1 lag(n=1)']  # Only ones that should be in minimal graph.
        self.assertSetEqual(set(n.identifier for n in tsdag.nodes), set(nodes))
        self.assertEqual(len(tsdag.edges), 1)

        full_insta = numpy.ones((3, 3))
        numpy.fill_diagonal(full_insta, 0)
        matrices = {0: full_insta, -1: numpy.zeros((3, 3))}
        matrices[-1][0, 1] = 1

        tsdag = TimeSeriesCausalGraph.from_adjacency_matrices(matrices, variables)

        # This should be the only non-contemporaneous edge. See check below that that is true.
        self.assertTrue(('X1 lag(n=1)', 'X2') in tsdag.get_edge_pairs())

        # check that contemporaneous are undirected edges and others are directed
        for edge in tsdag.edges:
            source, destination = edge.source, edge.destination
            self.assertIsInstance(source, TimeSeriesNode)
            self.assertIsInstance(destination, TimeSeriesNode)
            if source.time_lag == destination.time_lag == 0:
                self.assertEqual(edge.get_edge_type(), EdgeType.UNDIRECTED_EDGE)
            else:
                self.assertEqual(source.variable_name, 'X1')
                self.assertEqual(source.time_lag, -1)
                self.assertEqual(destination.variable_name, 'X2')
                self.assertEqual(destination.time_lag, 0)
                self.assertLess(source.time_lag, destination.time_lag)
                self.assertEqual(edge.get_edge_type(), EdgeType.DIRECTED_EDGE)

    def test_from_adjacency_matrices_floating_nodes(self):
        tscg = TimeSeriesCausalGraph.from_adjacency_matrices({0: numpy.zeros((1, 1))})  # will have 1 floating node
        self.assertFalse(tscg.is_empty())
        self.assertEqual(1, len(tscg.nodes))
        self.assertEqual(0, len(tscg.edges))
        tscg = TimeSeriesCausalGraph.from_adjacency_matrices({-1: numpy.zeros((1, 1))})  # will have 1 floating node
        self.assertFalse(tscg.is_empty())
        self.assertEqual(1, len(tscg.nodes))
        self.assertEqual(0, len(tscg.edges))
        tscg = TimeSeriesCausalGraph.from_adjacency_matrices(
            {0: numpy.zeros((1, 1))}, construct_minimal=False
        )  # will have 1 floating node
        self.assertFalse(tscg.is_empty())
        self.assertEqual(1, len(tscg.nodes))
        self.assertEqual(0, len(tscg.edges))
        tscg = TimeSeriesCausalGraph.from_adjacency_matrices(
            {-1: numpy.zeros((1, 1))}, construct_minimal=False
        )  # will have 2 floating nodes
        self.assertFalse(tscg.is_empty())
        self.assertEqual(2, len(tscg.nodes))
        self.assertEqual(0, len(tscg.edges))

        # test for the bugfix where there are floating lagged nodes
        edge_pairs = [
            ('node_0', 'node_2'),
            ('node_0 lag(n=1)', 'node_2'),
            ('node_0 lag(n=2)', 'node_0'),
            ('node_0 lag(n=2)', 'node_4'),
            ('node_1', 'node_3'),
            ('node_1 lag(n=1)', 'node_3'),
            ('node_1 lag(n=2)', 'node_0'),
            ('node_1 lag(n=2)', 'node_2'),
            ('node_1 lag(n=2)', 'node_3'),
            ('node_2 lag(n=2)', 'node_0'),
            ('node_3 lag(n=2)', 'node_0'),
            ('node_3 lag(n=2)', 'node_3'),
            ('node_3 lag(n=2)', 'node_4'),
            ('node_4', 'node_1'),
            ('node_4', 'node_3'),
            ('node_4 lag(n=1)', 'node_1'),
            ('node_4 lag(n=1)', 'node_3'),
            ('node_4 lag(n=2)', 'node_1'),
            ('node_4 lag(n=2)', 'node_2'),
            ('node_4 lag(n=2)', 'node_3'),
        ]

        # create the corresponding graph
        tsdag = TimeSeriesCausalGraph()
        for u, v in edge_pairs:
            tsdag.add_edge(u, v)

        # get the adjacency matrices from the tsdag
        adj_matrices = tsdag.adjacency_matrices

        # create the graph from the adjacency matrices
        tsdag_2 = TimeSeriesCausalGraph.from_adjacency_matrices(adj_matrices, tsdag.variables)

        # check that the graphs are equal
        self.assertEqual(tsdag, tsdag_2)
        self.assertTrue(tsdag.__eq__(tsdag_2, deep=True))

        # check that the adjacency matrices are the same
        self.assertEqual(adj_matrices.keys(), tsdag_2.adjacency_matrices.keys())
        for k in adj_matrices:
            numpy.testing.assert_array_equal(adj_matrices[k], tsdag_2.adjacency_matrices[k])

        tsdag_2 = TimeSeriesCausalGraph.from_adjacency_matrices(adj_matrices, tsdag.variables, construct_minimal=False)
        self.assertNotEqual(tsdag, tsdag_2)

        # create a graph with a floating node
        tsdag = TimeSeriesCausalGraph()
        tsdag.add_node('X1')

        self.assertDictEqual(tsdag.adjacency_matrices, {})
        numpy.testing.assert_equal(tsdag.adjacency_matrix, numpy.zeros((1, 1)))

        tsdag = TimeSeriesCausalGraph()
        tsdag.add_edge('a', 'b')
        tsdag.add_edge('a lag(n=1)', 'b lag(n=1)')
        tsdag.add_edge('a lag(n=1)', 'a')
        tsdag.add_edge('b lag(n=1)', 'b')

        # get the adjacency matrices from the tsdag
        adj_matrices = tsdag.adjacency_matrices

        # check they correspond to the minimal graph
        self.assertEqual(
            TimeSeriesCausalGraph.from_adjacency_matrices(adj_matrices, tsdag.variables), tsdag.get_minimal_graph()
        )
        # test they are different if not minimal
        self.assertNotEqual(
            TimeSeriesCausalGraph.from_adjacency_matrices(
                tsdag.adjacency_matrices, tsdag.variables, construct_minimal=False
            ),
            tsdag,
        )

        # test using the full adj matrix
        self.assertEqual(TimeSeriesCausalGraph.from_adjacency_matrix(tsdag.adjacency_matrix, tsdag.nodes), tsdag)
        self.assertTrue(
            tsdag.__eq__(TimeSeriesCausalGraph.from_adjacency_matrix(tsdag.adjacency_matrix, tsdag.nodes), deep=True)
        )

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

        # test get summary graph with a already summary graph
        tsgraph = TimeSeriesCausalGraph()
        tsgraph.add_node('c', variable_type=NodeVariableType.BINARY)
        tsgraph.add_edge('a', 'b')
        tsgraph.add_edge('b', 'c')
        graph = CausalGraph.from_dict(tsgraph.to_dict())

        summary_graph = tsgraph.get_summary_graph()

        self.assertEqual(summary_graph, graph)
        # deep equality
        self.assertTrue(graph.__eq__(summary_graph, True))

    def test_variable_names(self):
        variables = self.tsdag.variables
        self.assertEqual(variables, ['X1', 'X2', 'X3'])

    def test_is_minimal(self):
        # dag
        self.assertTrue(self.tsdag.is_minimal_graph())
        # dag_1
        self.assertFalse(self.tsdag_1.is_minimal_graph())

    def test_max_backwards_lag(self):
        # dag
        self.assertEqual(self.tsdag.max_backward_lag, 1)
        # dag_1
        self.assertEqual(self.tsdag_1.max_backward_lag, 2)

        # test with a new graph
        tsdag = TimeSeriesCausalGraph()
        self.assertIsNone(tsdag.max_backward_lag)

        tsdag.add_edge('X1 lag(n=2)', 'X1', edge_type=EdgeType.DIRECTED_EDGE)

        self.assertEqual(tsdag.max_backward_lag, 2)

        # remove the lagged node
        tsdag.remove_node('X1 lag(n=2)')
        self.assertEqual(tsdag.max_backward_lag, 0)

        # remove the contemporaneous node
        tsdag.remove_node('X1')
        self.assertIsNone(tsdag.max_backward_lag)

        # test with only positive lags
        tsdag = TimeSeriesCausalGraph()
        tsdag.add_edge('X1 future(n=1)', 'X1 future(n=2)', edge_type=EdgeType.DIRECTED_EDGE)
        self.assertIsNone(tsdag.max_backward_lag)

    def test_max_forwards_lag(self):
        # dag
        self.assertEqual(self.tsdag.max_forward_lag, 0)
        # dag_1
        self.assertEqual(self.tsdag_1.max_forward_lag, 0)

        # extend the graph in the future
        new_tscg = self.tsdag.extend_graph(forward_steps=2)
        self.assertEqual(new_tscg.max_forward_lag, 2)

        # test with a new graph
        tsdag = TimeSeriesCausalGraph()
        self.assertIsNone(tsdag.max_forward_lag)

        tsdag.add_edge('X1', 'X1 future(n=2)', edge_type=EdgeType.DIRECTED_EDGE)

        self.assertEqual(tsdag.max_forward_lag, 2)

        # remove the lagged node
        tsdag.remove_node('X1 future(n=2)')
        self.assertEqual(tsdag.max_forward_lag, 0)

        # remove the contemporaneous node
        tsdag.remove_node('X1')
        self.assertIsNone(tsdag.max_forward_lag)

        # test with only negative lags
        tsdag = TimeSeriesCausalGraph()
        tsdag.add_edge('X1 lag(n=2)', 'X1 lag(n=1)', edge_type=EdgeType.DIRECTED_EDGE)
        self.assertIsNone(tsdag.max_forward_lag)

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

        graph = TimeSeriesCausalGraph()
        graph.add_node('c', variable_type=NodeVariableType.BINARY)
        graph.add_edge('a', 'b')
        graph.add_edge('b', 'c')
        graph.add_time_edge('a', -1, 'a', 0)
        graph.add_time_edge('b', -1, 'b', 0)

        self.assertTrue(graph.is_minimal_graph())

        minimal_graph = graph.get_minimal_graph()
        self.assertEqual(minimal_graph, graph)
        self.assertTrue(graph.__eq__(minimal_graph, True))

        # test with floating nodes
        tsdag = TimeSeriesCausalGraph()
        tsdag.add_node('a lag(n=2)')
        min_tsgraph = tsdag.get_minimal_graph()
        self.assertEqual(len(min_tsgraph.nodes), 1)
        self.assertTrue(min_tsgraph.node_exists('a'))

        # test with b-1 -> b, c-2, c-1
        tsdag = TimeSeriesCausalGraph()
        tsdag.add_edge('b lag(n=1)', 'b')
        tsdag.add_node('c lag(n=2)')
        tsdag.add_node('c lag(n=1)')
        min_tsgraph = tsdag.get_minimal_graph()

        self.assertTrue(min_tsgraph.edge_exists('b lag(n=1)', 'b'))
        self.assertEqual(len(min_tsgraph.nodes), 3)
        self.assertTrue(min_tsgraph.node_exists('c'))

        # b - 1->c
        tsdag = TimeSeriesCausalGraph()
        tsdag.add_edge('b lag(n=1)', 'c')
        min_tsgraph = tsdag.get_minimal_graph()
        self.assertTrue(min_tsgraph.edge_exists('b lag(n=1)', 'c'))
        self.assertEqual(len(min_tsgraph.nodes), 2)

        # test it does not fail with a very big graph due to recursion
        self.tsdag_1.extend_graph(forward_steps=200, backward_steps=200).get_minimal_graph()

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

        # check that node variable types are the same
        graph = TimeSeriesCausalGraph()
        graph.add_node('c', variable_type=NodeVariableType.BINARY)
        graph.add_edge('a', 'b')
        graph.add_edge('b', 'c')

        extended_graph = graph.extend_graph(backward_steps=1)

        self.assertEqual(extended_graph.get_node('c lag(n=1)').variable_type, NodeVariableType.BINARY)

        # test it adds the contemporaneous node
        graph = TimeSeriesCausalGraph()
        graph.add_node('a lag(n=1)')
        ext_graph = graph.extend_graph(backward_steps=1, forward_steps=0)

        # check it adds the contemporaneous node
        self.assertTrue(ext_graph.node_exists('a'))

        ext_graph = graph.extend_graph(backward_steps=3, forward_steps=3)
        # check it adds all the nodes
        for i in range(-3, 4):
            name = get_name_with_lag('a', i)
            self.assertTrue(ext_graph.node_exists(name))

        # test with a graph that is not aligned at lag 0
        graph = TimeSeriesCausalGraph()
        graph.add_nodes_from(['tier_0_0 lag(n=1)', 'tier_0_1', 'tier_1', 'tier_2'])
        graph.add_edge('tier_0_0 lag(n=1)', 'tier_0_1')
        graph.add_edge('tier_0_1', 'tier_2')
        graph.add_edge('tier_1', 'tier_2')

        extended_graph = graph.extend_graph(backward_steps=1, forward_steps=0)
        self.assertTrue(extended_graph.node_exists('tier_0_0'))

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

        cg = TimeSeriesCausalGraph()
        cg.add_edge('x lag(n=1)', 'y')

        # test it does not raise an error and matches expected graph
        extended_cg = cg.extend_graph(forward_steps=1)
        gt_extended = TimeSeriesCausalGraph()
        gt_extended.add_edge('x lag(n=1)', 'y')
        gt_extended.add_edge('x', 'y future(n=1)')
        gt_extended.add_node('x future(n=1)')
        self.assertEqual(extended_cg, gt_extended)

        # test it adds the contemporaneous node
        graph = TimeSeriesCausalGraph()
        graph.add_node('a future(n=1)')
        ext_graph = graph.extend_graph(backward_steps=0, forward_steps=1)

        # check it adds the contemporaneous node
        self.assertTrue(ext_graph.node_exists('a'))

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

    def test_get_nodes_at_lag(self):
        self.assertEqual(['X1', 'X2', 'X3'], [node.identifier for node in self.tsdag.get_nodes_at_lag(0)])
        self.assertEqual(
            ['X1 lag(n=1)', 'X2 lag(n=1)', 'X3 lag(n=1)'], [node.identifier for node in self.tsdag.get_nodes_at_lag(-1)]
        )

        # same with tsdag_1
        self.assertEqual(['X1', 'X2', 'X3'], [node.identifier for node in self.tsdag_1.get_nodes_at_lag(0)])
        self.assertEqual(
            ['X1 lag(n=1)', 'X2 lag(n=1)', 'X3 lag(n=1)'],
            [node.identifier for node in self.tsdag_1.get_nodes_at_lag(-1)],
        )
        self.assertEqual(
            ['X1 lag(n=2)', 'X2 lag(n=2)', 'X3 lag(n=2)'],
            [node.identifier for node in self.tsdag_1.get_nodes_at_lag(-2)],
        )

    def test_contemporaneous_nodes(self):
        self.assertEqual([n.identifier for n in self.tsdag.get_contemporaneous_nodes('X1')], ['X2', 'X3'])
        self.assertEqual([n.identifier for n in self.tsdag.get_contemporaneous_nodes('X2')], ['X1', 'X3'])
        self.assertEqual([n.identifier for n in self.tsdag.get_contemporaneous_nodes('X3')], ['X1', 'X2'])

    def test_from_gml(self):
        reconstruction = TimeSeriesCausalGraph.from_gml_string(self.tsdag.to_gml_string())
        self.assertEqual(self.tsdag, reconstruction)

        # same with deep equality
        self.assertTrue(self.tsdag.__eq__(reconstruction, True))

    def test_from_networkx(self):
        reconstruction = TimeSeriesCausalGraph.from_networkx(self.tsdag.to_networkx())
        self.assertEqual(self.tsdag, reconstruction)

        # same with deep equality
        self.assertTrue(self.tsdag.__eq__(reconstruction, True))

    def test_from_skeleton(self):
        reconstruction = TimeSeriesCausalGraph.from_skeleton(self.tsdag.skeleton)

        # The reconstruction won't have directions so should just match skeleton.

        # Check that their networkx representations are the same.
        self.assertTrue(networkx.utils.graphs_equal(self.tsdag.skeleton.to_networkx(), reconstruction.to_networkx()))
        # Can check networkx as the reconstruction has no directions so networkx.Graph of CausalGraph and
        # Skeleton will be the same.
        self.assertTrue(
            networkx.utils.graphs_equal(reconstruction.to_networkx(), reconstruction.skeleton.to_networkx())
        )

        # Also confirm that equality method works. Again reconstruction won't have directions so just check skeletons.
        self.assertEqual(self.tsdag.skeleton, reconstruction.skeleton)

        # deep equality passes as we maintain time awareness and metadata for nodes.
        self.assertTrue(self.tsdag.skeleton.__eq__(reconstruction.skeleton, True))

    def test_get_topological_order(self):
        g = TimeSeriesCausalGraph()
        g.add_edge('x', 'y')
        g.add_edge('x', 't')
        g.add_edge('t', 'y')
        g.add_time_edge('x', -1, 't', 0)
        g.add_time_edge('t', -1, 't', 0)

        top_order = g.get_topological_order()
        self.assertListEqual(top_order, ['t lag(n=1)', 'x lag(n=1)', 'x', 't', 'y'])

        # test with respect time ordering to false
        top_order = g.get_topological_order(respect_time_ordering=False)
        self.assertListEqual(top_order, ['t lag(n=1)', 'x', 'x lag(n=1)', 't', 'y'])

        top_order = g.get_topological_order(return_all=True)
        self.assertListEqual(
            top_order, [['x lag(n=1)', 't lag(n=1)', 'x', 't', 'y'], ['t lag(n=1)', 'x lag(n=1)', 'x', 't', 'y']]
        )

        # Confirm it raises when not a dag.
        g.add_edge('z', 'x', edge_type=EdgeType.UNDIRECTED_EDGE)
        with self.assertRaises(AssertionError):
            g.get_topological_order()

        # Test with extended graph.
        cg = CausalGraph()
        cg.add_edge('x', 'y')
        ts_cg = TimeSeriesCausalGraph.from_causal_graph(cg)

        top_order = ts_cg.get_topological_order()
        self.assertListEqual(top_order, ['x', 'y'])

        top_order = ts_cg.get_topological_order(return_all=True)
        self.assertListEqual(top_order, [['x', 'y']])

        extended_graph = ts_cg.extend_graph(backward_steps=2, forward_steps=0)

        top_order = extended_graph.get_topological_order()
        self.assertListEqual(top_order, ['x lag(n=2)', 'y lag(n=2)', 'x lag(n=1)', 'y lag(n=1)', 'x', 'y'])

        top_order = extended_graph.get_topological_order(return_all=True)
        self.assertListEqual(top_order, [['x lag(n=2)', 'y lag(n=2)', 'x lag(n=1)', 'y lag(n=1)', 'x', 'y']])

        extended_graph = ts_cg.extend_graph(backward_steps=1, forward_steps=1)

        top_order = extended_graph.get_topological_order()
        self.assertListEqual(top_order, ['x lag(n=1)', 'y lag(n=1)', 'x', 'y', 'x future(n=1)', 'y future(n=1)'])

        top_order = extended_graph.get_topological_order(return_all=True)
        self.assertListEqual(top_order, [['x lag(n=1)', 'y lag(n=1)', 'x', 'y', 'x future(n=1)', 'y future(n=1)']])

        # Test with floating nodes.
        cg = CausalGraph()
        cg.add_node('x')
        cg.add_node('y')
        ts_cg = TimeSeriesCausalGraph.from_causal_graph(cg)

        self.assertListEqual(ts_cg.get_topological_order(), ['x', 'y'])
        self.assertListEqual(ts_cg.get_topological_order(return_all=True), [['y', 'x'], ['x', 'y']])

        extended_graph = ts_cg.extend_graph(1, 1)
        self.assertListEqual(
            extended_graph.get_topological_order(),
            ['x lag(n=1)', 'y lag(n=1)', 'x', 'y', 'x future(n=1)', 'y future(n=1)'],
        )

        # Test with no contemporaneous nodes.
        ts_cg = TimeSeriesCausalGraph()
        ts_cg.add_node('x lag(n=1)')

        self.assertListEqual(ts_cg.get_topological_order(), ['x lag(n=1)'])
        self.assertListEqual(ts_cg.get_topological_order(return_all=True), [['x lag(n=1)']])

        # Add contemporaneous in via extend.
        self.assertListEqual(ts_cg.extend_graph(1, 0).get_topological_order(), ['x lag(n=1)', 'x'])
        self.assertListEqual(ts_cg.extend_graph(1, 0).get_topological_order(return_all=True), [['x lag(n=1)', 'x']])

        self.assertListEqual(
            ts_cg.extend_graph(2, 2).get_topological_order(),
            ['x lag(n=2)', 'x lag(n=1)', 'x', 'x future(n=1)', 'x future(n=2)'],
        )

        # test with additional graphs
        tsdag = TimeSeriesCausalGraph()
        tsdag.add_edge('x', 'y')
        tsdag.add_time_edge('y', -1, 'y', 0)
        tsdag = tsdag.extend_graph(backward_steps=2)

        order = tsdag.get_topological_order()
        self.assertListEqual(order, ['x lag(n=2)', 'y lag(n=2)', 'x lag(n=1)', 'y lag(n=1)', 'x', 'y'])

        # with return all and respect time ordering to false
        order = tsdag.get_topological_order(return_all=True, respect_time_ordering=False)
        self.assertIn(['x lag(n=2)', 'x lag(n=1)', 'y lag(n=2)', 'y lag(n=1)', 'x', 'y'], order)

        tsdag = TimeSeriesCausalGraph()
        tsdag.add_edge('z', 'y')
        tsdag.add_edge('x', 'y')
        tsdag.add_time_edge('y', -1, 'y', 0)
        tsdag = tsdag.extend_graph(backward_steps=2)

        order = tsdag.get_topological_order()

        self.assertListEqual(
            order, ['x lag(n=2)', 'z lag(n=2)', 'y lag(n=2)', 'x lag(n=1)', 'z lag(n=1)', 'y lag(n=1)', 'x', 'z', 'y']
        )
        order = tsdag.get_topological_order(return_all=True, respect_time_ordering=False)
        self.assertIn(
            ['z lag(n=2)', 'x lag(n=2)', 'y lag(n=2)', 'z lag(n=1)', 'x lag(n=1)', 'y lag(n=1)', 'z', 'x', 'y'], order
        )

        tscg = TimeSeriesCausalGraph()
        # y-1 -> y <- x
        tscg.add_edge('x', 'y')
        tscg.add_time_edge('y', -1, 'y', 0)

        order = tscg.get_topological_order(return_all=True)
        # check there's no duplicates
        self.assertEqual(len(order), 1)
        self.assertListEqual(order[0], ['y lag(n=1)', 'x', 'y'])

        order = tscg.get_topological_order(return_all=True, respect_time_ordering=False)
        self.assertEqual(len(order), 2)
        self.assertListEqual(order[0], ['y lag(n=1)', 'x', 'y'])
        self.assertListEqual(order[1], ['x', 'y lag(n=1)', 'y'])

    def test_order_swapped(self):
        tscg = TimeSeriesCausalGraph()
        tscg.add_edge('x', 'y lag(n=1)', edge_type=EdgeType.UNDIRECTED_EDGE)

        # check y lag(n=1) -- x exists
        self.assertEqual(tscg.get_edge('y lag(n=1)', 'x').edge_type, EdgeType.UNDIRECTED_EDGE)


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
            cg.add_node('apple lag(n=3)', 'apple', -2)

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

    def test_get_nodes_for_variable_name(self):

        cg = TimeSeriesCausalGraph()
        cg.add_time_edge('a', -1, 'a', 0)
        cg.add_time_edge('a', -2, 'a', 0)
        cg.add_time_edge('a', -3, 'a', -1)
        cg.add_time_edge('a', -2, 'a', -1)
        cg.add_time_edge('a', -1, 'b', -1)
        cg.add_time_edge('a', -1, 'b', 0)
        cg.add_time_edge('a', 0, 'b', 0)
        cg.add_time_edge('a', 0, 'b', 1)
        cg.add_time_edge('a', 0, 'c', 1)

        self.assertSetEqual(
            set(cg.get_nodes_for_variable_name('a')), set(cg.get_nodes(['a lag(n=3)', 'a lag(n=2)', 'a lag(n=1)', 'a']))
        )
        self.assertSetEqual(
            set(cg.get_nodes_for_variable_name('b')), set(cg.get_nodes(['b lag(n=1)', 'b', 'b future(n=1)']))
        )
        self.assertSetEqual(set(cg.get_nodes_for_variable_name('c')), set(cg.get_nodes(['c future(n=1)'])))
