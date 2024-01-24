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

from cai_causal_graph import CausalGraph, EdgeType
from cai_causal_graph.exceptions import CausalGraphErrors


class TestCausalGraphEdgeTypes(unittest.TestCase):
    def setUp(self):
        # list of node identifiers for all causal graphs
        self.nodes = ['a', 'b', 'c', 'd', 'e', 'f']

        # define a DAG
        self.dag = CausalGraph()
        self.dag.add_nodes_from(self.nodes)
        self.dag.add_edge('b', 'a', edge_type=EdgeType.DIRECTED_EDGE)
        self.dag.add_edge('b', 'c', edge_type=EdgeType.DIRECTED_EDGE)
        self.dag.add_edge('a', 'c', edge_type=EdgeType.DIRECTED_EDGE)
        self.dag.add_edge('c', 'f', edge_type=EdgeType.DIRECTED_EDGE)
        self.dag.add_edge('f', 'd', edge_type=EdgeType.DIRECTED_EDGE)
        self.dag.add_edge('f', 'e', edge_type=EdgeType.DIRECTED_EDGE)
        self.dag.add_edge('e', 'd', edge_type=EdgeType.DIRECTED_EDGE)

        # define a CPDAG
        self.cpdag = CausalGraph()
        self.cpdag.add_nodes_from(self.nodes)
        self.cpdag.add_edge('b', 'a', edge_type=EdgeType.UNDIRECTED_EDGE)
        self.cpdag.add_edge('b', 'c', edge_type=EdgeType.DIRECTED_EDGE)
        self.cpdag.add_edge('a', 'c', edge_type=EdgeType.UNDIRECTED_EDGE)
        self.cpdag.add_edge('c', 'f', edge_type=EdgeType.DIRECTED_EDGE)
        self.cpdag.add_edge('f', 'd', edge_type=EdgeType.DIRECTED_EDGE)
        self.cpdag.add_edge('f', 'e', edge_type=EdgeType.DIRECTED_EDGE)
        self.cpdag.add_edge('e', 'd', edge_type=EdgeType.UNDIRECTED_EDGE)

        # define a MAG
        self.mag = CausalGraph()
        self.mag.add_nodes_from(self.nodes)
        self.mag.add_edge('b', 'a', edge_type=EdgeType.UNDIRECTED_EDGE)
        self.mag.add_edge('b', 'c', edge_type=EdgeType.DIRECTED_EDGE)
        self.mag.add_edge('a', 'c', edge_type=EdgeType.UNDIRECTED_EDGE)
        self.mag.add_edge('c', 'f', edge_type=EdgeType.DIRECTED_EDGE)
        self.mag.add_edge('f', 'd', edge_type=EdgeType.BIDIRECTED_EDGE)
        self.mag.add_edge('f', 'e', edge_type=EdgeType.BIDIRECTED_EDGE)
        self.mag.add_edge('e', 'd', edge_type=EdgeType.UNDIRECTED_EDGE)

        # define a PAG
        self.pag = CausalGraph()
        self.pag.add_nodes_from(self.nodes)
        self.pag.add_edge('b', 'a', edge_type=EdgeType.UNKNOWN_UNDIRECTED_EDGE)
        self.pag.add_edge('b', 'c', edge_type=EdgeType.UNKNOWN_DIRECTED_EDGE)
        self.pag.add_edge('a', 'c', edge_type=EdgeType.UNDIRECTED_EDGE)
        self.pag.add_edge('c', 'f', edge_type=EdgeType.DIRECTED_EDGE)
        self.pag.add_edge('f', 'd', edge_type=EdgeType.BIDIRECTED_EDGE)
        self.pag.add_edge('f', 'e', edge_type=EdgeType.BIDIRECTED_EDGE)
        self.pag.add_edge('e', 'd', edge_type=EdgeType.UNDIRECTED_EDGE)

    def test_causal_graph_identifiers(self):
        # check that the identifiers are what we would expect from a DAG and other graph types
        self.assertEqual(self.dag.identifier, '<b>_<a>_<c>_<f>_<e>_<d>')
        self.assertEqual(self.cpdag.identifier, '<a>_<b>_<c>_<d>_<e>_<f>')
        self.assertEqual(self.mag.identifier, '<a>_<b>_<c>_<d>_<e>_<f>')
        self.assertEqual(self.pag.identifier, '<a>_<b>_<c>_<d>_<e>_<f>')

    def test_is_dag(self):
        # check that the DAG is identified as such and that the other graph types are not
        self.assertTrue(self.dag.is_dag())
        self.assertFalse(self.cpdag.is_dag())
        self.assertFalse(self.mag.is_dag())
        self.assertFalse(self.pag.is_dag())

    def test_is_dag_cyclic_graph(self):
        # create a cyclic causal graph with directed edges and check that it raises the correct error
        with self.assertRaises(CausalGraphErrors.CyclicConnectionError):
            causal_graph = CausalGraph()
            causal_graph.add_edge('a', 'b')
            causal_graph.add_edge('b', 'c')
            causal_graph.add_edge('c', 'a')

    def test_get_nodes(self):
        # check that we can get a single node
        self.assertEqual(self.dag.get_node('a').identifier, 'a')
        self.assertEqual(self.cpdag.get_node('a').identifier, 'a')
        self.assertEqual(self.mag.get_node('a').identifier, 'a')
        self.assertEqual(self.pag.get_node('a').identifier, 'a')

        # check that we can get all nodes
        self.assertEqual(len(self.dag.get_nodes()), 6)
        self.assertEqual(len(self.cpdag.get_nodes()), 6)
        self.assertEqual(len(self.mag.get_nodes()), 6)
        self.assertEqual(len(self.pag.get_nodes()), 6)

        # check that the returned node names are correct
        self.assertEqual(self.dag.get_node_names(), self.nodes)
        self.assertEqual(self.cpdag.get_node_names(), self.nodes)
        self.assertEqual(self.mag.get_node_names(), self.nodes)
        self.assertEqual(self.pag.get_node_names(), self.nodes)

        # also check that we throw a KeyError if the node does not exist
        with self.assertRaises(KeyError):
            self.dag.get_node('x')

    def test_add_and_remove_nodes(self):
        # add node and check that it is correctly added
        self.dag.add_node('x')
        self.assertEqual(len(self.dag.get_nodes()), 7)

        # add an edge to the new node
        self.dag.add_edge('a', 'x', edge_type=EdgeType.DIRECTED_EDGE)
        self.assertEqual(len(self.dag.get_edges()), 8)

        # delete node
        self.dag.remove_node('x')
        self.assertEqual(len(self.dag.get_nodes()), 6)
        self.assertEqual(len(self.dag.get_edges()), 7)

    def test_get_edges(self):
        # get a directed edge within a dag and make sure that it works with specifying the edge_type as well
        edge = self.dag.get_edge('b', 'a')
        self.assertEqual(edge.identifier, "('b', 'a')")
        self.assertEqual(edge.descriptor, '(b -> a)')
        edge_with_type = self.dag.get_edge('b', 'a', edge_type=EdgeType.DIRECTED_EDGE)
        self.assertEqual(edge.identifier, edge_with_type.identifier)
        self.assertEqual(edge.descriptor, edge_with_type.descriptor)

        # trying to get edges that do not exist should not be possible
        with self.assertRaises(CausalGraphErrors.EdgeDoesNotExistError):
            self.dag.get_edge('a', 'b')
        with self.assertRaises(CausalGraphErrors.EdgeDoesNotExistError):
            self.dag.get_edge('b', 'a', edge_type=EdgeType.UNDIRECTED_EDGE)
        with self.assertRaises(CausalGraphErrors.EdgeDoesNotExistError):
            self.dag.get_edge('x', 'y')

        # get all edges at once
        self.assertEqual(len(self.dag.get_edges()), 7)

        # get an undirected edge within a cpdag to make sure it works for other edge types as well
        edge = self.cpdag.get_edge('b', 'a')
        self.assertEqual(edge.identifier, "('b', 'a')")
        self.assertEqual(edge.descriptor, '(b -- a)')
        edge_with_type = self.cpdag.get_edge('b', 'a', edge_type=EdgeType.UNDIRECTED_EDGE)
        self.assertEqual(edge.identifier, edge_with_type.identifier)
        self.assertEqual(edge.descriptor, edge_with_type.descriptor)

    def test_get_edges_by_type(self):
        # check that we get the correct edge numbers for the DAG
        self.assertEqual(len(self.dag.get_directed_edges()), 7)
        self.assertEqual(len(self.dag.get_undirected_edges()), 0)
        self.assertEqual(len(self.dag.get_bidirected_edges()), 0)
        self.assertEqual(len(self.dag.get_unknown_edges()), 0)
        self.assertEqual(len(self.dag.get_unknown_directed_edges()), 0)
        self.assertEqual(len(self.dag.get_unknown_undirected_edges()), 0)

        # check that we get the correct edge numbers for the CPDAG
        self.assertEqual(len(self.cpdag.get_directed_edges()), 4)
        self.assertEqual(len(self.cpdag.get_undirected_edges()), 3)
        self.assertEqual(len(self.cpdag.get_bidirected_edges()), 0)
        self.assertEqual(len(self.cpdag.get_unknown_edges()), 0)
        self.assertEqual(len(self.cpdag.get_unknown_directed_edges()), 0)
        self.assertEqual(len(self.cpdag.get_unknown_undirected_edges()), 0)

        # check that we get the correct edge numbers for the MAG
        self.assertEqual(len(self.mag.get_directed_edges()), 2)
        self.assertEqual(len(self.mag.get_undirected_edges()), 3)
        self.assertEqual(len(self.mag.get_bidirected_edges()), 2)
        self.assertEqual(len(self.mag.get_unknown_edges()), 0)
        self.assertEqual(len(self.mag.get_unknown_directed_edges()), 0)
        self.assertEqual(len(self.mag.get_unknown_undirected_edges()), 0)

        # check that we get the correct edge numbers for the PAG
        self.assertEqual(len(self.pag.get_directed_edges()), 1)
        self.assertEqual(len(self.pag.get_undirected_edges()), 2)
        self.assertEqual(len(self.pag.get_bidirected_edges()), 2)
        self.assertEqual(len(self.pag.get_unknown_edges()), 0)
        self.assertEqual(len(self.pag.get_unknown_directed_edges()), 1)
        self.assertEqual(len(self.pag.get_unknown_undirected_edges()), 1)

    def test_add_and_remove_edges(self):
        # add edge and check that it is correctly added
        self.dag.add_edge('x', 'a', edge_type=EdgeType.DIRECTED_EDGE)
        self.dag.add_edge('x', 'y', edge_type=EdgeType.DIRECTED_EDGE)
        self.assertEqual(len(self.dag.get_nodes()), 8)
        self.assertEqual(len(self.dag.get_edges()), 9)

        # remove edge
        self.dag.remove_edge('x', 'y', edge_type=EdgeType.DIRECTED_EDGE)
        self.assertEqual(len(self.dag.get_nodes()), 8)
        self.assertEqual(len(self.dag.get_edges()), 8)

    def test_add_and_remove_edges_with_nodes(self):
        # add edge and check that it is correctly added
        self.dag.add_nodes_from(['x', 'y'])

        self.dag.add_edge(self.dag['x'], self.dag['a'], edge_type=EdgeType.DIRECTED_EDGE)
        self.dag.add_edge(self.dag['x'], self.dag['y'], edge_type=EdgeType.DIRECTED_EDGE)
        self.assertEqual(len(self.dag.get_nodes()), 8)
        self.assertEqual(len(self.dag.get_edges()), 9)

        # remove edge
        self.dag.remove_edge(self.dag['x'], self.dag['y'], edge_type=EdgeType.DIRECTED_EDGE)
        self.assertEqual(len(self.dag.get_nodes()), 8)
        self.assertEqual(len(self.dag.get_edges()), 8)

    def test_get_parent_graph(self):
        # get a parent graph for the DAG
        pa_dag = self.dag.get_parents_graph('d')
        self.assertEqual(len(pa_dag.get_nodes()), 3)
        self.assertEqual(len(pa_dag.get_edges()), 2)

        # get a parent graph for the CPDAG
        pa_cpdag = self.cpdag.get_parents_graph('d')
        self.assertEqual(len(pa_cpdag.get_nodes()), 2)
        self.assertEqual(len(pa_cpdag.get_edges()), 1)

        # get a parent graph for the MAG
        pa_mag = self.mag.get_parents_graph('f')
        self.assertEqual(len(pa_mag.get_nodes()), 2)
        self.assertEqual(len(pa_mag.get_edges()), 1)

        # get a parent graph for the PAG
        pa_pag = self.pag.get_parents_graph('f')
        self.assertEqual(len(pa_pag.get_nodes()), 2)
        self.assertEqual(len(pa_pag.get_edges()), 1)

    def test_to_networkx(self):
        # check that we get the correct set of nodes in the networkx object
        dag_networkx = self.dag.to_networkx()
        self.assertEqual(list(dag_networkx.nodes), self.dag.get_node_names())

    def test_to_numpy(self):
        # check that a sub-set of adjacency matrix elements are correct
        dag_adj, dag_nodes = self.dag.to_numpy()
        self.assertEqual(dag_adj[dag_nodes.index('b'), dag_nodes.index('a')], 1)
        self.assertEqual(dag_adj[dag_nodes.index('a'), dag_nodes.index('b')], 0)
        self.assertEqual(dag_adj[dag_nodes.index('b'), dag_nodes.index('c')], 1)
        self.assertEqual(dag_adj[dag_nodes.index('e'), dag_nodes.index('d')], 1)
        self.assertEqual(dag_adj[dag_nodes.index('d'), dag_nodes.index('e')], 0)

    def test_causal_graph_dict_reconstruction(self):
        # check that we can reconstruct the same CausalGraph object
        self.assertEqual(CausalGraph.from_dict(self.dag.to_dict()), self.dag)
        self.assertEqual(CausalGraph.from_dict(self.cpdag.to_dict()), self.cpdag)
        self.assertEqual(CausalGraph.from_dict(self.mag.to_dict()), self.mag)
        self.assertEqual(CausalGraph.from_dict(self.pag.to_dict()), self.pag)

    def test_causal_graph_networkx_reconstruction(self):
        # check that we can reconstruct the same CausalGraph object
        self.assertEqual(CausalGraph.from_networkx(self.dag.to_networkx()), self.dag)
        for graph in [self.cpdag, self.mag, self.pag]:
            with self.assertRaises(CausalGraphErrors.GraphConversionError):
                graph.to_networkx()

    def test_causal_graph_numpy_reconstruction(self):
        # check that we can reconstruct the same CausalGraph object
        dag_adj, dag_nodes = self.dag.to_numpy()
        cpdag_adj, cpdag_nodes = self.cpdag.to_numpy()
        self.assertEqual(CausalGraph.from_adjacency_matrix(dag_adj, dag_nodes), self.dag)
        self.assertEqual(CausalGraph.from_adjacency_matrix(cpdag_adj, cpdag_nodes), self.cpdag)
        for graph in [self.mag, self.pag]:
            with self.assertRaises(TypeError):
                graph.to_numpy()

    def test_change_edge_type(self):
        # test using node identifiers
        self.assertTrue(self.dag.is_dag())
        edge = self.dag.get_edges()[0]

        source, destination = edge.source.identifier, edge.destination.identifier

        self.assertIn(source, self.dag.get_parents(destination))
        self.assertIn(destination, self.dag.get_children(source))

        self.dag.change_edge_type(source=source, destination=destination, new_edge_type=EdgeType.UNDIRECTED_EDGE)
        self.assertEqual(
            self.dag.get_edge(source=source, destination=destination).get_edge_type(), EdgeType.UNDIRECTED_EDGE
        )
        self.assertFalse(self.dag.is_dag())

        self.assertNotIn(source, self.dag.get_parents(destination))
        self.assertNotIn(destination, self.dag.get_children(source))

        # test using nodes
        edge = self.dag.get_edges()[0]

        source, destination = edge.source, edge.destination

        self.dag.change_edge_type(source=source, destination=destination, new_edge_type=EdgeType.DIRECTED_EDGE)
        self.assertEqual(
            self.dag.get_edge(source=source.identifier, destination=destination.identifier).get_edge_type(),
            EdgeType.DIRECTED_EDGE,
        )
        self.assertTrue(self.dag.is_dag())

        self.assertIn(source.identifier, self.dag.get_parents(destination.identifier))
        self.assertIn(destination.identifier, self.dag.get_children(source.identifier))

        # test by setting to the sme type as before and checking that edge is still valid (has not been deleted)
        edge = self.dag.get_edges()[0]
        self.dag.change_edge_type(source=edge.source, destination=edge.destination, new_edge_type=edge.get_edge_type())

        edge_back = self.dag.get_edges()[0]

        self.assertEqual(id(edge), id(edge_back))

    def test_change_edge_type_raises(self):
        # check that trying to change an edge which does not exist, raises
        edge = self.dag.get_edges()[0]

        source, destination = edge.source, edge.destination

        self.dag.remove_edge(source=source.identifier, destination=destination.identifier)

        with self.assertRaises(CausalGraphErrors.EdgeDoesNotExistError):
            self.dag.change_edge_type(
                source=source.identifier, destination=destination.identifier, new_edge_type=EdgeType.DIRECTED_EDGE
            )
