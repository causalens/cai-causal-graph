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
from copy import deepcopy

import networkx
import numpy
import pandas

from cai_causal_graph import __version__ as VERSION
from cai_causal_graph.causal_graph import CausalGraph
from cai_causal_graph.exceptions import CausalGraphErrors
from cai_causal_graph.type_definitions import EDGE_T, NODE_T


class TestCausalGraphSerialization(unittest.TestCase):
    def setUp(self):
        self.empty_graph = CausalGraph()

        self.nodes = ['x', 'y', 'z1', 'z2']
        self.latent_nodes = ['x_L']

        self.fully_connected_graph = CausalGraph()
        self.fully_connected_graph.add_edge('x', 'z1')
        self.fully_connected_graph.add_edge('x', 'z2')
        self.fully_connected_graph.add_edge('y', 'z1')
        self.fully_connected_graph.add_edge('y', 'z2')

        self.fully_connected_graph_edges = [['x', 'z1'], ['x', 'z2'], ['y', 'z1'], ['y', 'z2']]

        self.fully_connected_graph_edges_sources = [('x', 2), ('y', 2)]
        self.fully_connected_graph_edges_destinations = [('z1', 2), ('z2', 2)]

        self.graph_with_latent_node = CausalGraph()
        self.graph_with_latent_node.add_nodes_from(['x', 'y', 'z1', 'z2'])
        self.graph_with_latent_node.add_edge('x', 'x_L')
        self.graph_with_latent_node.add_edge('x_L', 'y')
        self.graph_with_latent_node.add_edge('y', 'z1')
        self.graph_with_latent_node.add_edge('y', 'z2')
        self.graph_with_latent_node.add_edge('x', 'z2')

        self.graph_with_latent_nodes_edges_sources = [('x', 2), ('y', 2), ('x_L', 1)]
        self.graph_with_latent_nodes_edges_destinations = [('x_L', 1), ('y', 1), ('z1', 1), ('z2', 2)]
        self.graph_with_latent_nodes_edges = [['x', 'x_L'], ['x_L', 'y'], ['y', 'z1'], ['y', 'z2'], ['x', 'z2']]

    def assert_graph_serialization_is_correct(self, graph: CausalGraph):
        reconstruction = CausalGraph.from_dict(graph.to_dict())

        # Check that their dict representations are the same.
        self.assertDictEqual(graph.to_dict(), reconstruction.to_dict())

        # Also confirm that equality method works.
        self.assertEqual(graph, reconstruction)

    def assert_graph_networkx_conversion_is_correct(self, graph: CausalGraph, include_dict: bool = True):
        reconstruction = CausalGraph.from_networkx(graph.to_networkx())

        # Confirm we get correct type on to_networkx.
        if graph._is_fully_directed():
            self.assertIsInstance(graph.to_networkx(), networkx.Graph)
            self.assertIsInstance(graph.to_networkx(), networkx.DiGraph)
        elif graph._is_fully_undirected():
            # DiGraph extends Graph so if it was a DiGraph then it would also be a Graph.
            self.assertIsInstance(graph.to_networkx(), networkx.Graph)
            self.assertNotIsInstance(graph.to_networkx(), networkx.DiGraph)

        # Check that their dict representations are the same.
        if include_dict:
            # Avoid this sometimes as dicts won't be equal, e.g., when metadata and/or non-default constraints are in
            # the original graph as this info won't make it to networkx.
            self.assertDictEqual(graph.to_dict(), reconstruction.to_dict())

        # Also confirm that equality method works.
        self.assertEqual(graph, reconstruction)

    def assert_graph_skeleton_conversion_is_correct(self, graph: CausalGraph):
        reconstruction = CausalGraph.from_skeleton(graph.skeleton)

        # The reconstruction won't have directions so should just match skeleton.

        # Check that their networkx representations are the same.
        self.assertTrue(networkx.utils.graphs_equal(graph.skeleton.to_networkx(), reconstruction.to_networkx()))
        # Can check networkx as the reconstruction has no directions so networkx.Graph of CausalGraph and
        # Skeleton will be the same.
        self.assertTrue(
            networkx.utils.graphs_equal(reconstruction.to_networkx(), reconstruction.skeleton.to_networkx())
        )

        # Also confirm that equality method works. Again reconstruction won't have directions so just check skeletons.
        self.assertEqual(graph.skeleton, reconstruction.skeleton)

        # We cannot check that their dict representations are the same as undirected edges can flip order in dictionary
        # without any issue around the actual graph representation. Therefore, keep this commented out.
        # self.assertDictEqual(graph.skeleton.to_dict(), reconstruction.skeleton.to_dict())

    def assert_graph_gml_conversion_is_correct(self, graph: CausalGraph, include_dict: bool = True):
        reconstruction = CausalGraph.from_gml_string(graph.to_gml_string())

        self.assertIsInstance(graph.to_gml_string(), str)

        # Check that their dict representations are the same.
        if include_dict:
            # Avoid this sometimes as dicts won't be equal, e.g., when metadata and/or non-default constraints are in
            # the original graph as this info won't make it to GML representation.
            self.assertDictEqual(graph.to_dict(), reconstruction.to_dict())

        # Also confirm that equality method works.
        self.assertEqual(graph, reconstruction)

    def test_graph_serializes(self):
        """This test ensures different serialization types and JSON serialization as well."""
        for graph in [self.empty_graph, self.fully_connected_graph, self.graph_with_latent_node]:
            graph_as_dict = graph.to_dict()
            graph_as_dict_2 = dict(graph)
            self.assertDictEqual(graph_as_dict, graph_as_dict_2)
            self.assertIsInstance(json.dumps(graph_as_dict), str)

    def test_graph_is_json_serializable(self):
        cg = CausalGraph()
        cg.add_edge('a', 'b')
        cg.add_edge('b', 'c')
        cg.add_node('d', variable_type=NODE_T.CONTINUOUS)

        cg_dict = cg.to_dict()

        cg_json = json.dumps(cg_dict)
        cg2 = CausalGraph.from_dict(json.loads(cg_json))

        self.assertEqual(cg2.to_dict(), cg_dict)

    def test_graph_identifier(self):
        self.assertEqual(self.fully_connected_graph.identifier, '<x>_<y>_<z1>_<z2>')
        empty_graph = CausalGraph()
        self.assertEqual(empty_graph.identifier, '<>')

    def test_graph_version(self):
        # results to package version
        self.assertEqual(VERSION, self.fully_connected_graph.to_dict().get('version'))

    def test_fully_connected_graph(self):
        self.assert_graph_serialization_is_correct(self.fully_connected_graph)
        self.assert_graph_networkx_conversion_is_correct(self.fully_connected_graph)
        self.assert_graph_skeleton_conversion_is_correct(self.fully_connected_graph)
        self.assert_graph_gml_conversion_is_correct(self.fully_connected_graph)

    def test_empty_graph(self):
        self.assert_graph_serialization_is_correct(self.empty_graph)
        self.assert_graph_networkx_conversion_is_correct(self.empty_graph)
        self.assert_graph_skeleton_conversion_is_correct(self.empty_graph)
        self.assert_graph_gml_conversion_is_correct(self.empty_graph)

    def test_graph_with_latent_nodes(self):
        self.assert_graph_serialization_is_correct(self.graph_with_latent_node)
        self.assert_graph_networkx_conversion_is_correct(self.graph_with_latent_node)
        self.assert_graph_skeleton_conversion_is_correct(self.graph_with_latent_node)
        self.assert_graph_gml_conversion_is_correct(self.graph_with_latent_node)

    def test_get_nodes_empty(self):
        self.assertEqual(len(self.empty_graph.get_nodes()), 0)

        for node in self.nodes:
            self.assertEqual(len(self.empty_graph.get_nodes(node)), 0)

    def test_get_nodes(self):
        self.assertEqual(len(self.fully_connected_graph.get_nodes()), len(self.nodes))

        for node in self.nodes:
            nodes = self.fully_connected_graph.get_nodes(node)
            self.assertEqual(len(nodes), 1)
            self.assertEqual(nodes[0].identifier, node)

    def test_get_nodes_list(self):
        self.assertEqual(len(self.fully_connected_graph.get_nodes()), len(self.nodes))

        self.assertEqual(len(self.fully_connected_graph.get_nodes(self.nodes)), len(self.nodes))
        self.assertListEqual([n.identifier for n in self.fully_connected_graph.get_nodes(['x', 'y'])], ['x', 'y'])

        with self.assertRaises(ValueError):
            self.fully_connected_graph.get_nodes(['x', 'y', 'z'])

    def test_get_nodes_latent(self):
        self.assertEqual(len(self.graph_with_latent_node.get_nodes()), len(self.nodes) + len(self.latent_nodes))

        for node in self.nodes + self.latent_nodes:
            nodes = self.graph_with_latent_node.get_nodes(node)
            self.assertEqual(len(nodes), 1)
            self.assertEqual(nodes[0].identifier, node)

    def test_get_edges_empty(self):
        self.assertEqual(len(self.empty_graph.get_edges()), 0)

        for source, destination in self.fully_connected_graph_edges:
            self.assertEqual(len(self.empty_graph.get_edges(source=source)), 0)
            self.assertEqual(len(self.empty_graph.get_edges(destination=destination)), 0)
            self.assertEqual(len(self.empty_graph.get_edges(source=source, destination=destination)), 0)

    def test_get_edges(self):
        self.assertEqual(len(self.fully_connected_graph.get_edges()), 4)

        for source, destination in self.fully_connected_graph_edges:
            self.assertEqual(len(self.fully_connected_graph.get_edges(source=source, destination=destination)), 1)

        for source, expected in self.fully_connected_graph_edges_sources:
            self.assertEqual(len(self.fully_connected_graph.get_edges(source=source)), expected)

        for destination, expected in self.fully_connected_graph_edges_destinations:
            self.assertEqual(len(self.fully_connected_graph.get_edges(destination=destination)), expected)

    def test_get_edges_latent(self):
        self.assertEqual(len(self.graph_with_latent_node.get_edges()), 5)

        for source, destination in self.graph_with_latent_nodes_edges:
            self.assertEqual(len(self.graph_with_latent_node.get_edges(source=source, destination=destination)), 1)

        for source, expected in self.graph_with_latent_nodes_edges_sources:
            self.assertEqual(len(self.graph_with_latent_node.get_edges(source=source)), expected)

        for destination, expected in self.graph_with_latent_nodes_edges_destinations:
            self.assertEqual(len(self.graph_with_latent_node.get_edges(destination=destination)), expected)

    def test_add_nodes_implicit(self):
        self.fully_connected_graph.add_edge('test_1', 'test_2')
        node_identifiers = [node.identifier for node in self.fully_connected_graph.get_nodes()]
        self.assertIn('test_1', node_identifiers)
        self.assertIn('test_2', node_identifiers)

    def test_add_node_raises_existing_and_updates(self):
        with self.assertRaises(CausalGraphErrors.NodeDuplicatedError):
            self.fully_connected_graph.add_node('x')

    def test_add_edge_raises_existing_and_updates(self):
        with self.assertRaises(CausalGraphErrors.EdgeDuplicatedError):
            self.fully_connected_graph.add_edge('x', 'z1')

    def test_graph_inputs(self):
        inputs = self.fully_connected_graph.get_inputs()
        input_identifiers = set(node.identifier for node in inputs)

        self.assertEqual(len(inputs), 2)
        self.assertSetEqual(input_identifiers, {'x', 'y'})

    def test_graph_outputs(self):
        outputs = self.fully_connected_graph.get_outputs()
        outputs_identifiers = set(node.identifier for node in outputs)

        self.assertEqual(len(outputs), 2)
        self.assertSetEqual(outputs_identifiers, {'z1', 'z2'})

    def test_delete_node(self):
        # Test that deleting the node works well
        self.fully_connected_graph.delete_node('x')
        self.assertNotIn('x', {node.identifier for node in self.fully_connected_graph.get_nodes()})

        # check that the appropriate error is raised when deleting the same node again
        with self.assertRaises(KeyError):
            self.fully_connected_graph.delete_node('x')

    def test_delete_edge(self):
        # Test that deleting edges works fine
        self.fully_connected_graph.delete_edge('x', 'z1')
        self.assertEqual(3, len(self.fully_connected_graph.get_edges()))

        # check that the appropriate error is raised when deleting the same edge again
        with self.assertRaises(CausalGraphErrors.EdgeDoesNotExistError):
            self.fully_connected_graph.delete_edge('x', 'z1')

    def test_replace_node_raises_error(self):
        causal_graph = CausalGraph()
        causal_graph.add_edge('x1', 'x2')
        causal_graph.add_edge('x1', 'x3')
        causal_graph.add_edge('x2', 'x3')
        causal_graph.add_edge('x2', 'x4')
        causal_graph.add_edge('x3', 'x4')
        causal_graph.add_edge('x4', 'x5')

        causal_graph.get_node('x2').meta = {'color': 'blue'}

        # replacing a node that does not exist raises an error
        with self.assertRaises(AssertionError):
            causal_graph.replace_node('y1', 'y2')

        # replace a node with a node that exists raises an error
        with self.assertRaises(AssertionError):
            causal_graph.replace_node('x1', 'x2')

    def test_replace_node_with_new(self):
        causal_graph = CausalGraph()
        causal_graph.add_edge('x1', 'x2')
        causal_graph.add_edge('x1', 'x3')
        causal_graph.add_edge('x2', 'x3')
        causal_graph.add_edge('x2', 'x4')
        causal_graph.add_edge('x3', 'x4')
        causal_graph.add_edge('x4', 'x5')

        causal_graph.get_node('x2').meta = {'color': 'blue'}

        # replacing a node works
        # get the inbound and outbound edges from original node

        input_nodes = set(edge.source.identifier for edge in causal_graph.get_edges(destination='x2'))
        output_nodes = set(edge.destination.identifier for edge in causal_graph.get_edges(source='x2'))

        meta = causal_graph.get_node('x2').meta
        causal_graph.replace_node('x2', 'y2')
        node_identifiers = [node.identifier for node in causal_graph.get_nodes()]
        self.assertIn('y2', node_identifiers)
        self.assertNotIn('x2', node_identifiers)
        self.assertEqual(causal_graph.get_node('y2').meta, meta)

        new_input_nodes = set(edge.source.identifier for edge in causal_graph.get_edges(destination='y2'))
        new_output_nodes = set(edge.destination.identifier for edge in causal_graph.get_edges(source='y2'))

        self.assertSetEqual(input_nodes, new_input_nodes)
        self.assertSetEqual(output_nodes, new_output_nodes)

    def test_replace_node_without_new(self):
        causal_graph = CausalGraph()
        causal_graph.add_edge('x1', 'x2')
        causal_graph.add_edge('x1', 'x3')
        causal_graph.add_edge('x2', 'x3')
        causal_graph.add_edge('x2', 'x4')
        causal_graph.add_edge('x3', 'x4')
        causal_graph.add_edge('x4', 'x5')

        causal_graph.get_node('x2').meta = {'color': 'blue'}

        # replacing a node without specifying new node, simply edits node parameters
        causal_graph.replace_node('x2', meta={'color': 'red'})
        self.assertEqual(causal_graph.get_node('x2').meta, {'color': 'red'})

    def test_replace_node_with_extras(self):
        causal_graph = CausalGraph()
        causal_graph.add_edge('x1', 'x2')
        causal_graph.add_edge('x1', 'x3')
        causal_graph.add_edge('x2', 'x3')
        causal_graph.add_edge('x2', 'x4')
        causal_graph.add_edge('x3', 'x4')
        causal_graph.add_edge('x4', 'x5')

        causal_graph.get_node('x2').meta = {'color': 'blue'}
        causal_graph.get_node('x2').variable_type = NODE_T.CONTINUOUS

        # replacing a node while specifying extra parameters
        meta = {'color': 'blue'}
        variable_type = NODE_T.BINARY
        causal_graph.replace_node('x2', 'y2', meta=meta, variable_type=variable_type)
        node_identifiers = [node.identifier for node in causal_graph.get_nodes()]
        self.assertIn('y2', node_identifiers)
        self.assertNotIn('x2', node_identifiers)
        self.assertEqual(causal_graph.get_node('y2').meta, meta)
        self.assertEqual(causal_graph.get_node('y2').variable_type, variable_type)

    def test_graph_with_metadata(self):
        causal_graph = CausalGraph()
        causal_graph.add_node('x', meta={'color': 'blue'})
        causal_graph.add_edge('a', 'b', meta={'color': 'red'})

        self.assertEqual({'color': 'blue'}, causal_graph.get_node('x').meta)
        self.assertEqual({'color': 'blue'}, causal_graph.get_node('x').get_metadata())
        self.assertEqual({'color': 'red'}, causal_graph.get_edge('a', 'b').meta)
        self.assertEqual({'color': 'red'}, causal_graph.get_edge('a', 'b').get_metadata())

        reconstruction = CausalGraph.from_dict(causal_graph.to_dict())

        # Evaluate the metadata from the dictionary reconstruction.
        self.assertEqual({'color': 'blue'}, reconstruction.get_node('x').meta)
        self.assertEqual({'color': 'red'}, reconstruction.get_edge('a', 'b').meta)

    def test_causal_graph_meta_is_preserved_in_copying(self):
        causal_graph = CausalGraph()
        causal_graph.add_node('x', meta={'color': 'blue'})
        causal_graph.add_edge('a', 'b', meta={'color': 'red'})

        self.assertEqual({'color': 'blue'}, causal_graph.get_node('x').meta)
        self.assertEqual({'color': 'blue'}, causal_graph.get_node('x').get_metadata())
        self.assertEqual({'color': 'red'}, causal_graph.get_edge('a', 'b').meta)
        self.assertEqual({'color': 'red'}, causal_graph.get_edge('a', 'b').get_metadata())

        cg_copy = causal_graph.copy()

        # Evaluate the metadata from the copy reconstruction.
        self.assertEqual({'color': 'blue'}, cg_copy.get_node('x').meta)
        self.assertEqual({'color': 'red'}, cg_copy.get_edge('a', 'b').meta)

    def test_add_edge_from_edge(self):
        causal_graph = CausalGraph()
        causal_graph.add_edge(source='a', destination='b', meta={'color': 'blue'}, edge_type=EDGE_T.UNDIRECTED_EDGE)

        edge = causal_graph.get_edge(source='a', destination='b')

        causal_graph_2 = CausalGraph()
        causal_graph_2.add_edge(edge=edge)

        edge_2 = causal_graph_2.get_edge(source='a', destination='b')

        self.assertEqual(edge.get_edge_type(), edge_2.get_edge_type())
        self.assertDictEqual(edge.metadata, edge_2.metadata)
        self.assertNotEqual(id(edge), id(edge_2))

    def test_add_edge_from_edge_deepcopies(self):
        class DummyObj:
            pass

        causal_graph = CausalGraph()
        causal_graph_2 = CausalGraph()

        causal_graph.add_edge(source='b', destination='c', meta={'dummy': DummyObj()})

        edge = causal_graph.get_edge(source='b', destination='c')
        causal_graph_2.add_edge(edge=edge)
        edge_2 = causal_graph_2.get_edge(source='b', destination='c')

        self.assertIsInstance(edge_2.meta['dummy'], DummyObj)
        self.assertNotEqual(id(edge_2.meta['dummy']), id(edge.meta['dummy']))

    def test_add_edge_from_edge_raises(self):
        causal_graph = CausalGraph()
        causal_graph.add_edge(source='a', destination='b')
        edge = causal_graph.get_edge(source='a', destination='b')

        with self.assertRaises(AssertionError):
            causal_graph.add_edge()

        with self.assertRaises(CausalGraphErrors.EdgeDuplicatedError):
            causal_graph.add_edge(edge=edge)

        with self.assertRaises(AssertionError):
            causal_graph.add_edge(edge=edge, source='a')

        with self.assertRaises(AssertionError):
            causal_graph.add_edge(edge=edge, meta={'foo': 'bar'})

        with self.assertRaises(AssertionError):
            causal_graph.add_edge(edge=edge, my_kwargs='foo')

    def test_add_edges_from(self):
        causal_graph = CausalGraph()
        edge_tuples = [('a', 'b'), ('b', 'c')]

        causal_graph.add_edges_from(edge_tuples)

        for edge in edge_tuples:
            self.assertTrue(causal_graph.edge_exists(*edge))
            self.assertEqual(
                causal_graph.get_edge(source=edge[0], destination=edge[1]).get_edge_type(), EDGE_T.DIRECTED_EDGE
            )

        # check can add more nodes again
        more_edge_tuples = [('c', 'd')]
        causal_graph.add_edges_from(more_edge_tuples)

        for edge in [*edge_tuples, *more_edge_tuples]:
            self.assertTrue(causal_graph.edge_exists(*edge))
            self.assertEqual(
                causal_graph.get_edge(source=edge[0], destination=edge[1]).get_edge_type(), EDGE_T.DIRECTED_EDGE
            )

        # check bad edge raises
        bad_edge_tuple = [('d', 'a')]

        with self.assertRaises(CausalGraphErrors.CyclicConnectionError):
            causal_graph.add_edges_from(bad_edge_tuple)

    def test_consistency_of_node_names(self):
        cg = CausalGraph()
        cg.add_edge('a', 'b')
        cg.add_edge('b', 'c')
        cg.add_edge('c', 'd')
        cg.add_edge('d', 'g')
        cg.add_edge('a', 'e')
        cg.add_edge('e', 'f')
        cg.add_edge('f', 'g')

        self.assertListEqual(cg.get_node_names(), [node.identifier for node in cg.nodes])

        cg = CausalGraph()
        cg.add_edge('d', 'a')
        cg.add_edge('a', 'r')
        cg.add_edge('aa', 'b')
        cg.add_edge('a', 'aa')
        cg.add_edge('b', 'c')

        self.assertListEqual(cg.get_node_names(), [node.identifier for node in cg.nodes])

    def test_adjacency_consistency(self):
        cg = CausalGraph()
        cg.add_edge('a', 'b')
        cg.add_edge('b', 'c')
        cg.add_edge('c', 'd')
        cg.add_edge('d', 'g')
        cg.add_edge('a', 'e')
        cg.add_edge('e', 'f')
        cg.add_edge('f', 'g')

        self.assertEqual(CausalGraph.from_adjacency_matrix(*cg.to_numpy()), cg)

        cg = CausalGraph()
        cg.add_edge('d', 'a')
        cg.add_edge('a', 'r')
        cg.add_edge('aa', 'b')
        cg.add_edge('a', 'aa')
        cg.add_edge('b', 'c')

        self.assertEqual(CausalGraph.from_adjacency_matrix(*cg.to_numpy()), cg)

    def test_adjacency(self):
        adj_1 = numpy.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        names_1 = ['a', 'b', 'c']
        adj_2 = numpy.array([[0, 1, 0], [0, 0, 0], [1, 0, 0]])
        names_2 = ['b', 'c', 'a']
        graph = CausalGraph(input_list=names_1)
        graph.add_edge('a', 'b')
        graph.add_edge('b', 'c')

        numpy.testing.assert_array_equal(graph.adjacency_matrix, adj_1)

        graph_1 = CausalGraph.from_adjacency_matrix(adj_1, names_1)

        self.assertEqual(graph_1, graph)

        # testing isomorphic graphs
        graph_2 = CausalGraph.from_adjacency_matrix(adj_2, names_2)
        self.assertEqual(graph_1, graph_2)

        # test non-equal graphs
        graph_3 = CausalGraph.from_adjacency_matrix(adj_1, names_2)
        self.assertNotEqual(graph_2, graph_3)

        # test adjacency without node names generates correct default node names
        graph_4 = CausalGraph.from_adjacency_matrix(adj_1)
        expected_node_names = [f'node_{i}' for i in range(3)]
        self.assertSetEqual(set(expected_node_names), set(graph_4.get_node_names()))

    def test_adjacency_raises(self):
        adj_er_1 = numpy.array([[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 0]])
        adj_er_2 = numpy.array([[1.1, 1, 0], [0, 0, 0], [1, 0, 0]])
        adj_correct = numpy.array([[0, 1, 0], [0, 0, 0], [1, 0, 0]])
        names_er = ['a', 'b']

        with self.assertRaises(CausalGraphErrors.InvalidAdjacencyMatrixError):
            CausalGraph.from_adjacency_matrix(adj_er_1)
            CausalGraph.from_adjacency_matrix(adj_er_2)
            CausalGraph.from_adjacency_matrix(adj_correct, names_er)

    def test_coerce_to_nodelike(self):
        self.assertEqual(CausalGraph.coerce_to_nodelike('a'), 'a')
        self.assertEqual(CausalGraph.coerce_to_nodelike(1), '1')

    def test_networkx(self):
        adj_1 = numpy.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        names_1 = ['a', 'b', 'c']
        adj_2 = numpy.array([[0, 1, 0], [0, 0, 0], [1, 0, 0]])
        names_2 = ['b', 'c', 'a']
        graph = CausalGraph(input_list=names_1)
        graph.add_edge('a', 'b')
        graph.add_edge('b', 'c')

        nx_g1 = networkx.from_pandas_adjacency(
            pandas.DataFrame(adj_1, index=names_1, columns=names_1), networkx.DiGraph
        )
        nx_g2 = networkx.from_pandas_adjacency(
            pandas.DataFrame(adj_2, index=names_2, columns=names_2), networkx.DiGraph
        )

        nx_g3 = graph.to_networkx()
        nx_g4 = networkx.from_pandas_adjacency(
            pandas.DataFrame(adj_1, index=names_2, columns=names_2), networkx.DiGraph
        )

        # test that from_networkx works
        self.assertTrue(CausalGraph.from_networkx(nx_g1) == graph)
        self.assertTrue(CausalGraph.from_networkx(nx_g2) == graph)
        self.assertFalse(CausalGraph.from_networkx(nx_g4) == graph)

        # test that to networkx also works
        numpy.testing.assert_array_equal(networkx.to_numpy_array(nx_g3), adj_1)
        self.assertTrue(list(nx_g3.nodes) == names_1)

        # test networkx from graph without nodes
        nx_g5 = networkx.from_numpy_array(adj_1, create_using=networkx.DiGraph)
        graph_5 = CausalGraph.from_networkx(nx_g5)
        # names should not be the same since networkx names are integers
        self.assertNotEqual(set(nx_g5.nodes), set(graph_5.get_node_names()))
        # they should be the same once converted to strings
        nx_node_names = [str(node) for node in nx_g5.nodes]
        self.assertSetEqual(set(nx_node_names), set(graph_5.get_node_names()))

        # test networkx.Graph vs networkx.DiGraph with equivalent graphs but with/without directions
        u = networkx.empty_graph(n=['a', 'b', 'c'], create_using=networkx.Graph)
        u.add_edge('a', 'b')
        u.add_edge('a', 'c')
        u.add_edge('b', 'c')

        d = networkx.empty_graph(n=['a', 'b', 'c'], create_using=networkx.DiGraph)
        d.add_edge('a', 'b')
        d.add_edge('a', 'c')
        d.add_edge('b', 'c')

        ug = CausalGraph.from_networkx(u)
        dg = CausalGraph.from_networkx(d)

        self.assertTrue(ug._is_fully_undirected())
        self.assertFalse(ug._is_fully_directed())
        self.assertFalse(dg._is_fully_undirected())
        self.assertTrue(dg._is_fully_directed())

        self.assertIsInstance(ug.to_networkx(), networkx.Graph)
        self.assertNotIsInstance(ug.to_networkx(), networkx.DiGraph)
        self.assertIsInstance(dg.to_networkx(), networkx.Graph)
        self.assertIsInstance(dg.to_networkx(), networkx.DiGraph)

        self.assertTrue(networkx.utils.graphs_equal(ug.to_networkx(), ug.skeleton.to_networkx()))
        self.assertTrue(networkx.utils.graphs_equal(u, ug.to_networkx()))
        self.assertTrue(networkx.utils.graphs_equal(d, dg.to_networkx()))
        self.assertTrue(networkx.utils.graphs_equal(u, dg.skeleton.to_networkx()))

    def test_networkx_raises(self):
        adj_1 = numpy.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        names_1 = ['a', 'b', 'c']

        nx_g1 = networkx.from_pandas_adjacency(
            pandas.DataFrame(adj_1, index=names_1, columns=names_1), networkx.MultiGraph
        )
        nx_g2 = networkx.from_pandas_adjacency(
            pandas.DataFrame(adj_1, index=names_1, columns=names_1), networkx.MultiDiGraph
        )

        with self.assertRaises(CausalGraphErrors.InvalidNetworkXError):
            CausalGraph.from_networkx(nx_g1)

        with self.assertRaises(CausalGraphErrors.InvalidNetworkXError):
            CausalGraph.from_networkx(nx_g2)

    def test_deepcopy(self):
        causal_graph_deepcopy = deepcopy(self.fully_connected_graph)
        self.fully_connected_graph.get_node('x')._identifier = 'test'
        self.assertEqual(causal_graph_deepcopy.get_node('x').get_identifier(), 'x')
        self.assertEqual(self.fully_connected_graph.get_node('x').get_identifier(), 'test')
        self.assertNotEqual(
            causal_graph_deepcopy.get_node('x').get_identifier(),
            self.fully_connected_graph.get_node('x').get_identifier(),
        )

    def test_get_children_graph(self):
        cg_original = CausalGraph()
        cg_original.add_edge('a', 'b')
        cg_original.add_edge('a', 'c')
        cg_original.add_edge('a', 'd')
        cg_original.add_edge('b', 'd')
        cg_original.add_edge('b', 'e')

        cg = cg_original.copy()

        a_children_graph = cg.get_children_graph('a')
        b_children_graph = cg.get_children_graph('b')
        c_children_graph = cg.get_children_graph('c')

        self.assertEqual(cg, cg_original)

        self.assertEqual(len(a_children_graph.get_nodes()), 4)
        self.assertEqual(len(a_children_graph.get_edges()), 3)
        for edge in a_children_graph.get_edges():
            self.assertEqual(edge.source.identifier, 'a')

        self.assertEqual(len(b_children_graph.get_nodes()), 3)
        self.assertEqual(len(b_children_graph.get_edges()), 2)
        for edge in b_children_graph.get_edges():
            self.assertEqual(edge.source.identifier, 'b')

    def test_get_children(self):
        cg_original = CausalGraph()
        cg_original.add_edge('a', 'b')
        cg_original.add_edge('a', 'c')
        cg_original.add_edge('a', 'd')
        cg_original.add_edge('b', 'd')
        cg_original.add_edge('b', 'e')

        cg = cg_original.copy()

        a_children = cg.get_children('a')
        b_children = cg.get_children('b')
        c_children = cg.get_children('c')

        self.assertEqual(cg, cg_original)

        self.assertSetEqual(set(a_children), {'d', 'c', 'b'})
        self.assertSetEqual(set(b_children), {'d', 'e'})
        self.assertListEqual(c_children, [])

    def test_get_parents_graph(self):
        causal_graph = CausalGraph(
            input_list=[i for i in 'abcdefg'], output_list=[i + ',0' for i in 'hijkl'], fully_connected=True
        )
        causal_graph.add_edge('a', 'b')
        causal_graph_2 = causal_graph.copy()

        star_causal_graph_h = causal_graph.get_parents_graph('h,0')
        star_causal_graph_a = causal_graph.get_parents_graph('a')
        star_causal_graph_k = causal_graph.get_parents_graph('k,0')

        # check that original graph is unaffected
        self.assertEqual(causal_graph, causal_graph_2)

        self.assertEqual(len(star_causal_graph_h.get_edges()), 7)
        self.assertEqual(len(star_causal_graph_h.get_nodes()), 8)
        for edge in star_causal_graph_h.edges:
            self.assertEqual(edge.destination.identifier, 'h,0')

        self.assertEqual(len(causal_graph_2.get_edges('a', 'b')), 1)
        self.assertEqual(len(star_causal_graph_h.get_edges('a', 'b')), 0)

        self.assertEqual(len(star_causal_graph_k.get_edges()), 7)
        self.assertEqual(len(star_causal_graph_k.get_nodes()), 8)
        for edge in star_causal_graph_k.edges:
            self.assertEqual(edge.destination.identifier, 'k,0')

        self.assertEqual(len(star_causal_graph_a.get_edges()), 0)
        self.assertListEqual(star_causal_graph_a.get_node_names(), ['a'])

        with self.assertRaises(AssertionError):
            _ = causal_graph.get_parents_graph('z')

    def test_get_parents(self):
        causal_graph = CausalGraph(
            input_list=[i for i in 'abcdefg'], output_list=[i + ',0' for i in 'hijkl'], fully_connected=True
        )
        causal_graph.add_edge('a', 'b')
        causal_graph_2 = causal_graph.copy()

        star_causal_graph_h_parents = causal_graph.get_parents('h,0')
        star_causal_graph_a_parents = causal_graph.get_parents('a')
        star_causal_graph_k_parents = causal_graph.get_parents('k,0')

        # check that original graph is unaffected
        self.assertEqual(causal_graph, causal_graph_2)

        self.assertSetEqual(set(star_causal_graph_h_parents), {'a', 'b', 'c', 'd', 'e', 'f', 'g'})
        self.assertSetEqual(set(star_causal_graph_k_parents), {'a', 'b', 'c', 'd', 'e', 'f', 'g'})
        self.assertListEqual(star_causal_graph_a_parents, [])
        self.assertListEqual(causal_graph.get_parents('b'), ['a'])

        with self.assertRaises(AssertionError):
            _ = causal_graph.get_parents('z')

    def test_get_ancestral_graph(self):
        causal_graph = CausalGraph()

        for s, d in [('a', 'b'), ('b', 'c'), ('d', 'c'), ('e', 'c')]:
            causal_graph.add_edge(s, d)

        c_ancestors = causal_graph.get_ancestors('c')
        b_ancestors = causal_graph.get_ancestors('b')
        a_ancestors = causal_graph.get_ancestors('a')

        self.assertEqual(c_ancestors, {'a', 'b', 'd', 'e'})
        self.assertEqual(b_ancestors, {'a'})
        self.assertTrue(len(a_ancestors) == 0)

        c_graph = causal_graph.get_ancestral_graph('c')
        for node in c_ancestors:
            self.assertTrue(c_graph.node_exists(node))

        b_graph = causal_graph.get_ancestral_graph('b')
        for node in b_ancestors:
            self.assertTrue(b_graph.node_exists(node))

        c_graph = causal_graph.get_ancestral_graph('c')
        self.assertEqual(causal_graph.get_nodes(), c_graph.get_nodes())
        self.assertEqual(causal_graph.get_edges(), c_graph.get_edges())

        self.assertEqual(len(causal_graph.get_ancestral_graph('a').get_nodes()), 1)

        # test catches non existent node
        with self.assertRaises(AssertionError):
            _ = causal_graph.get_ancestral_graph('f')

    def test_get_descendants_graph(self):
        causal_graph = CausalGraph()

        for s, d in [('a', 'b'), ('b', 'c'), ('d', 'c'), ('e', 'c')]:
            causal_graph.add_edge(s, d)

        c_descendants = causal_graph.get_descendants('c')
        b_descendants = causal_graph.get_descendants('b')
        a_descendants = causal_graph.get_descendants('a')

        self.assertTrue(len(c_descendants) == 0)
        self.assertEqual(b_descendants, {'c'})
        self.assertEqual(a_descendants, {'b', 'c'})

        a_graph = causal_graph.get_descendant_graph('a')
        for node in a_descendants:
            self.assertTrue(a_graph.node_exists(node))

        b_graph = causal_graph.get_descendant_graph('b')
        for node in b_descendants:
            self.assertTrue(b_graph.node_exists(node))

        self.assertEqual(len(causal_graph.get_descendant_graph('c').get_nodes()), 1)

        # test catches non existent node
        with self.assertRaises(AssertionError):
            _ = causal_graph.get_descendant_graph('f')


class TestCausalGraphPrinting(unittest.TestCase):
    def test_default_nodes_and_edges(self):
        cg = CausalGraph()

        n = cg.add_node('a')
        e = cg.add_edge('a', 'b')

        self.assertIsInstance(n.__repr__(), str)
        self.assertIsInstance(e.__repr__(), str)
        self.assertIsInstance(cg.__repr__(), str)

        self.assertIsInstance(n.details(), str)
        self.assertIsInstance(e.details(), str)
        self.assertIsInstance(cg.details(), str)

    def test_complex_nodes_and_edges(self):
        cg = CausalGraph()

        n = cg.add_node('a')
        e = cg.add_edge('a', 'b', edge_type=EDGE_T.BIDIRECTED_EDGE)

        self.assertIsInstance(n.__repr__(), str)
        self.assertIsInstance(e.__repr__(), str)
        self.assertIsInstance(cg.__repr__(), str)

        self.assertIsInstance(n.details(), str)
        self.assertIsInstance(e.details(), str)
        self.assertIsInstance(cg.details(), str)

    def test_add_node_from_node(self):
        causal_graph = CausalGraph()
        causal_graph.add_node(identifier='a', meta={'color': 'blue'}, variable_type=NODE_T.BINARY)

        node = causal_graph.get_node(identifier='a')

        causal_graph_2 = CausalGraph()
        causal_graph_2.add_node(node=node)

        node_2 = causal_graph_2.get_node(identifier='a')

        self.assertEqual(node.variable_type, node_2.variable_type)
        self.assertDictEqual(node.metadata, node_2.metadata)

    def test_add_node_from_node_deepcopies(self):
        class DummyObj:
            pass

        causal_graph = CausalGraph()
        causal_graph.add_node(identifier='a', meta={'dummy': DummyObj()})
        node = causal_graph.get_node(identifier='a')

        causal_graph_2 = CausalGraph()
        causal_graph_2.add_node(node=node)

        node_2 = causal_graph_2.get_node(identifier='a')

        self.assertIsInstance(node_2.metadata['dummy'], DummyObj)
        self.assertNotEqual(id(node_2.metadata['dummy']), id(node.metadata['dummy']))

    def test_add_node_from_node_raises(self):
        causal_graph = CausalGraph()
        causal_graph.add_node(identifier='a')
        node = causal_graph.get_node(identifier='a')

        with self.assertRaises(AssertionError):
            causal_graph.add_node()

        with self.assertRaises(CausalGraphErrors.NodeDuplicatedError):
            causal_graph.add_node(node=node)

        with self.assertRaises(AssertionError):
            causal_graph.add_node(node=node, identifier='b')

        with self.assertRaises(AssertionError):
            causal_graph.add_node(node=node, meta={'foo': 'bar'})

        with self.assertRaises(AssertionError):
            causal_graph.add_node(node=node, variable_type=NODE_T.BINARY)

        with self.assertRaises(AssertionError):
            causal_graph.add_node(node=node, my_kwargs='foo')

    def test_node_variable_type(self):
        cg = CausalGraph()

        cg.add_node('a', variable_type=NODE_T.CONTINUOUS)
        self.assertEqual(cg.get_node('a').variable_type, NODE_T.CONTINUOUS)

        cg.get_node('a').variable_type = NODE_T.MULTICLASS
        self.assertEqual(cg.get_node('a').variable_type, NODE_T.MULTICLASS)

        cg.get_node('a').variable_type = 'binary'
        self.assertEqual(cg.get_node('a').variable_type, NODE_T.BINARY)

        with self.assertRaises(ValueError):
            cg.get_node('a').variable_type = 'not_a_variable_type'

        # Test the above but directly through the constructor
        cg.add_node('b', variable_type='binary')
        self.assertEqual(cg.get_node('b').variable_type, NODE_T.BINARY)

        with self.assertRaises(ValueError):
            cg.add_node('c', variable_type='not_a_variable_type')

    def test_node_repr(self):
        cg = CausalGraph()
        cg.add_node('a')
        cg.add_node('b', variable_type=NODE_T.CONTINUOUS)

        self.assertEqual(repr(cg['a']), 'Node("a")')
        self.assertEqual(repr(cg['b']), 'Node("b", type="continuous")')
