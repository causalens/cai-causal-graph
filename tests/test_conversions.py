"""
Copyright 2024 Impulse Innovations Limited

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
from typing import Optional, Union

import networkx
import numpy

from cai_causal_graph import CausalGraph, NodeVariableType, Skeleton, TimeSeriesCausalGraph
from cai_causal_graph.graph_components import Edge, Node, TimeSeriesNode
from cai_causal_graph.type_definitions import TIME_LAG, VARIABLE_NAME


def simulate_er_dag(
    d: int, degree: float, dtype: type = int, rng: Optional[Union[numpy.random.Generator, int]] = None
) -> numpy.ndarray:
    """
    Simulates a random DAG using the Erdos-Renyi algorithm via the networkx package.

    :param d: The number of nodes contained in the dataset (i.e. number of columns).
    :param degree: The expected degree of the random DAG (i.e. average number of connections).
    :param dtype: The desired dtype for the final numpy.ndarray representing the adjacency matrix of the random DAG.)
    :param rng: The random number generator used for reproducibility. If set to None, a random number is used as
        the seed. It is also possible to provide a Generator object, e.g. obtained via numpy.random.default_rng().
        Note that numpy.random.RandomState() is not supported.
    :return: A binary adjacency matrix of the random DAG.
    """

    def _get_acyclic_graph(amat: numpy.ndarray) -> numpy.ndarray:
        return numpy.tril(amat, k=-1)

    def _graph_to_adjmat(graph: networkx.DiGraph) -> numpy.ndarray:
        return networkx.to_numpy_array(graph)

    # set up the random state
    if rng is None:
        rng = numpy.random.default_rng()
    elif isinstance(rng, int) and rng >= 0:
        rng = numpy.random.default_rng(seed=rng)
    elif isinstance(rng, numpy.random.Generator):
        pass
    elif isinstance(rng, numpy.random.RandomState):
        raise TypeError('RandomState is legacy, please use default_rng() to generate a new Generator object.')
    else:
        raise NotImplementedError('Please provide a valid seed or rng object.')

    # generate binary adjacency matrix using Erdos-Renyi algorithm
    p = float(degree) / (d - 1)

    # note that we need to pass an int seed to erdos_renyi_graph(), which we extract from the current rng state
    rng_state = rng.__getstate__()['state']['state']

    # generate graph, using a faster algorithm if graph is sparse, or deliberately specified
    sparsity_threshold = 0.1
    if p <= sparsity_threshold:
        graph_tmp = networkx.generators.fast_gnp_random_graph(n=d, p=p, seed=rng_state)
    else:
        graph_tmp = networkx.generators.gnp_random_graph(n=d, p=p, seed=rng_state)
    amat_tmp = _graph_to_adjmat(graph_tmp)  # Undirected
    amat_bin = _get_acyclic_graph(amat_tmp)

    return amat_bin.astype(dtype=dtype)


class TestConversions(unittest.TestCase):
    """
    This unit test explicitly tries the various to/from methods of the major classes. These are tested in the
    respective class unit tests but good to ensure exhaustive testing of to/from and from/to in one place.
    """

    def setUp(self):
        seed = 24
        degree = 2
        num_nodes = 20
        self.node_names = [f'x{str(i).zfill(2)}' for i in range(num_nodes)]  # To ensure nodes are sorted.
        self.random_dag_adj = simulate_er_dag(d=num_nodes, degree=degree, rng=seed)

        # With networkx >= 3.2, we can use `edge_attr` parameter, then use networkx.utils.graphs_equal() but
        # we cannot currently use it as the graph is made with a weights metadata on each edge, which we don't
        # care about.
        networkx_dag = networkx.from_numpy_array(
            self.random_dag_adj, create_using=networkx.DiGraph
        )  # , edge_attr=None)
        networkx_dag = networkx.relabel_nodes(networkx_dag, dict(zip(range(num_nodes), self.node_names)))  # name nodes

        # Remove weights from edges by constructing a new DiGraph from the edges from networkx_dag
        self.random_dag = networkx.DiGraph()
        self.random_dag.add_nodes_from(networkx_dag.nodes)  # in case there are any floating nodes
        self.random_dag.add_edges_from(networkx_dag.edges)

        # Make networkx skeleton from this random DAG.
        self.random_skeleton = self.random_dag.to_undirected(reciprocal=False, as_view=False)

    def test_node(self):
        # To / from dict
        cg = CausalGraph.from_networkx(self.random_dag)
        node = cg.get_node('x01')
        self.assertDictEqual(node.to_dict(), Node.from_dict(node.to_dict()).to_dict())
        self.assertEqual(node, Node.from_dict(node.to_dict()))
        self.assertTrue(node.__eq__(Node.from_dict(node.to_dict()), deep=True))

        node.variable_type = NodeVariableType.CONTINUOUS
        self.assertDictEqual(node.to_dict(), Node.from_dict(node.to_dict()).to_dict())
        self.assertEqual(node, Node.from_dict(node.to_dict()))
        self.assertTrue(node.__eq__(Node.from_dict(node.to_dict()), deep=True))

        # Replace node and confirm expectation.
        orig_node_inbound_edges = [Edge.from_dict(e.to_dict()) for e in node.get_inbound_edges()]
        orig_node_outbound_edges = [Edge.from_dict(e.to_dict()) for e in node.get_outbound_edges()]
        cg.replace_node(node, new_node_id='x99', variable_type=NodeVariableType.BINARY)
        new_node = cg.get_node('x99')
        self.assertDictEqual(new_node.to_dict(), Node.from_dict(new_node.to_dict()).to_dict())
        self.assertEqual(new_node, Node.from_dict(new_node.to_dict()))
        self.assertTrue(new_node.__eq__(Node.from_dict(new_node.to_dict()), deep=True))
        # New node should be equivalent to original (including same edges and stuff) but type is different.
        self.assertNotEqual(node, new_node)  # due to different variable types
        self.assertListEqual(
            [e.source for e in orig_node_inbound_edges], [e.source for e in new_node.get_inbound_edges()]
        )
        self.assertListEqual(
            [e.edge_type for e in orig_node_inbound_edges], [e.edge_type for e in new_node.get_inbound_edges()]
        )
        self.assertListEqual(
            [e.destination for e in orig_node_outbound_edges], [e.destination for e in new_node.get_outbound_edges()]
        )
        self.assertListEqual(
            [e.edge_type for e in orig_node_outbound_edges], [e.edge_type for e in new_node.get_outbound_edges()]
        )

    def test_time_series_node(self):
        # To / from dict
        cg = CausalGraph.from_networkx(self.random_dag)
        tscg = TimeSeriesCausalGraph.from_causal_graph(cg)
        tscg.add_edge('x01 lag(n=1)', 'x01')
        node = cg.get_node('x01')
        tsnode = tscg.get_node('x01')

        node_dict = node.to_dict()
        tsnode_dict = tsnode.to_dict()

        self.assertDictEqual(node.meta, {})
        self.assertDictEqual(tsnode.meta, {TIME_LAG: 0, VARIABLE_NAME: 'x01'})

        self.assertNotEqual(node, tsnode)

        self.assertDictEqual(node_dict, Node.from_dict(node_dict).to_dict())
        self.assertNotEqual(
            node_dict, TimeSeriesNode.from_dict(node_dict).to_dict()
        )  # not equal due to meta + node class

        self.assertNotEqual(node_dict, Node.from_dict(tsnode_dict).to_dict())  # not equal due to meta
        self.assertNotEqual(
            node_dict, TimeSeriesNode.from_dict(tsnode_dict).to_dict()
        )  # not equal due to meta + node class

        self.assertNotEqual(tsnode_dict, Node.from_dict(node_dict).to_dict())  # not equal due to meta + node class
        self.assertDictEqual(tsnode_dict, TimeSeriesNode.from_dict(node_dict).to_dict())

        self.assertNotEqual(tsnode_dict, Node.from_dict(tsnode_dict).to_dict())  # not equal due to meta + node class
        self.assertDictEqual(tsnode_dict, TimeSeriesNode.from_dict(tsnode_dict).to_dict())

        self.assertEqual(node, Node.from_dict(node_dict))
        self.assertTrue(node.__eq__(Node.from_dict(node_dict), deep=True))
        self.assertEqual(node, Node.from_dict(tsnode_dict))
        self.assertFalse(node.__eq__(Node.from_dict(tsnode_dict), deep=True))  # not equal due to meta

        self.assertEqual(tsnode, TimeSeriesNode.from_dict(node_dict))
        self.assertTrue(tsnode.__eq__(TimeSeriesNode.from_dict(node_dict), deep=True))
        self.assertEqual(tsnode, TimeSeriesNode.from_dict(tsnode_dict))
        self.assertTrue(tsnode.__eq__(TimeSeriesNode.from_dict(tsnode_dict), deep=True))

        tsnode = tscg.get_node('x01 lag(n=1)')
        tsnode_dict = tsnode.to_dict()

        self.assertDictEqual(tsnode.meta, {TIME_LAG: -1, VARIABLE_NAME: 'x01'})

        self.assertNotEqual(tsnode_dict, Node.from_dict(tsnode_dict).to_dict())  # not equal due to meta + node class
        self.assertDictEqual(tsnode_dict, TimeSeriesNode.from_dict(tsnode_dict).to_dict())
        self.assertEqual(tsnode, TimeSeriesNode.from_dict(tsnode_dict))
        self.assertTrue(tsnode.__eq__(TimeSeriesNode.from_dict(tsnode_dict), deep=True))

    def test_edge(self):
        # To / from dict
        cg = CausalGraph.from_networkx(self.random_dag)
        edge = cg.edges[0]
        self.assertDictEqual(edge.to_dict(), Edge.from_dict(edge.to_dict()).to_dict())
        self.assertEqual(edge, Edge.from_dict(edge.to_dict()))
        self.assertTrue(edge.__eq__(Edge.from_dict(edge.to_dict()), deep=True))

    def test_skeleton(self):
        for graph_class in [None, CausalGraph, TimeSeriesCausalGraph]:
            # To / from networkx
            self.assertTrue(
                networkx.utils.graphs_equal(
                    Skeleton.from_networkx(self.random_skeleton, graph_class=graph_class).to_networkx(),
                    self.random_skeleton,
                )
            )
            skeleton = Skeleton.from_networkx(self.random_skeleton, graph_class=graph_class)
            self.assertEqual(skeleton, Skeleton.from_networkx(self.random_skeleton, graph_class=graph_class))
            self.assertTrue(
                skeleton.__eq__(Skeleton.from_networkx(self.random_skeleton, graph_class=graph_class), deep=True)
            )

            # To / from dict
            self.assertDictEqual(
                skeleton.to_dict(), Skeleton.from_dict(skeleton.to_dict(), graph_class=graph_class).to_dict()
            )
            self.assertEqual(skeleton, Skeleton.from_dict(skeleton.to_dict(), graph_class=graph_class))
            self.assertTrue(skeleton.__eq__(Skeleton.from_dict(skeleton.to_dict(), graph_class=graph_class), deep=True))

            # To / from adjacency matrix / to numpy
            # Note that the adjacency matrix and nodes from CG are sorted by node names.
            # This is why in the setUp we must ensure that the node names are already sorted. Otherwise, we would need
            # to reorder the adjacency matrix for comparison.
            # It is OK though as cg.adjacency_matrix and cg.get_node_names() respect the same order.
            gt_adj = networkx.to_numpy_array(self.random_skeleton, nodelist=self.node_names)
            self.assertEqual(
                skeleton, Skeleton.from_adjacency_matrix(gt_adj, node_names=self.node_names, graph_class=graph_class)
            )
            self.assertTrue(
                skeleton.__eq__(
                    Skeleton.from_adjacency_matrix(gt_adj, node_names=self.node_names, graph_class=graph_class),
                    deep=True,
                )
            )
            numpy.testing.assert_array_equal(gt_adj, skeleton.adjacency_matrix)
            numpy.testing.assert_array_equal(
                gt_adj,
                Skeleton.from_adjacency_matrix(
                    gt_adj, node_names=self.node_names, graph_class=graph_class
                ).adjacency_matrix,
            )

            nodes = skeleton.get_node_names()
            self.assertListEqual(self.node_names, nodes)
            adj, nodes = skeleton.to_numpy()
            numpy.testing.assert_array_equal(gt_adj, adj)
            self.assertListEqual(self.node_names, nodes)

            nodes = Skeleton.from_adjacency_matrix(
                gt_adj, node_names=self.node_names, graph_class=graph_class
            ).get_node_names()
            self.assertListEqual(self.node_names, nodes)
            adj, nodes = Skeleton.from_adjacency_matrix(
                gt_adj, node_names=self.node_names, graph_class=graph_class
            ).to_numpy()
            numpy.testing.assert_array_equal(gt_adj, adj)
            self.assertListEqual(self.node_names, nodes)

            # To / from GML
            self.assertEqual(
                skeleton.to_gml_string(),
                Skeleton.from_gml_string(skeleton.to_gml_string(), graph_class=graph_class).to_gml_string(),
            )
            self.assertEqual(skeleton, Skeleton.from_gml_string(skeleton.to_gml_string(), graph_class=graph_class))
            self.assertTrue(
                skeleton.__eq__(Skeleton.from_gml_string(skeleton.to_gml_string(), graph_class=graph_class), deep=True)
            )

    def test_causal_graph(self):
        # To / from networkx
        self.assertTrue(
            networkx.utils.graphs_equal(CausalGraph.from_networkx(self.random_dag).to_networkx(), self.random_dag)
        )
        cg = CausalGraph.from_networkx(self.random_dag)
        self.assertEqual(cg, CausalGraph.from_networkx(self.random_dag))
        self.assertTrue(cg.__eq__(CausalGraph.from_networkx(self.random_dag), deep=True))

        # To / from dict
        self.assertDictEqual(cg.to_dict(), CausalGraph.from_dict(cg.to_dict()).to_dict())
        self.assertEqual(cg, CausalGraph.from_dict(cg.to_dict()))
        self.assertTrue(cg.__eq__(CausalGraph.from_dict(cg.to_dict()), deep=True))

        # To / from adjacency matrix / to numpy
        # Note that the adjacency matrix and nodes from CG are sorted by node names.
        # This is why in the setUp we must ensure that the node names are already sorted. Otherwise, we would need
        # to reorder the adjacency matrix for comparison.
        # It is OK though as cg.adjacency_matrix and cg.get_node_names() respect the same order.
        self.assertEqual(cg, CausalGraph.from_adjacency_matrix(self.random_dag_adj, node_names=self.node_names))
        self.assertTrue(
            cg.__eq__(CausalGraph.from_adjacency_matrix(self.random_dag_adj, node_names=self.node_names), deep=True)
        )
        numpy.testing.assert_array_equal(self.random_dag_adj, cg.adjacency_matrix)
        numpy.testing.assert_array_equal(
            self.random_dag_adj,
            CausalGraph.from_adjacency_matrix(self.random_dag_adj, node_names=self.node_names).adjacency_matrix,
        )

        nodes = cg.get_node_names()
        self.assertListEqual(self.node_names, nodes)
        adj, nodes = cg.to_numpy()
        numpy.testing.assert_array_equal(self.random_dag_adj, adj)
        self.assertListEqual(self.node_names, nodes)

        nodes = CausalGraph.from_adjacency_matrix(self.random_dag_adj, node_names=self.node_names).get_node_names()
        self.assertListEqual(self.node_names, nodes)
        adj, nodes = CausalGraph.from_adjacency_matrix(self.random_dag_adj, node_names=self.node_names).to_numpy()
        numpy.testing.assert_array_equal(self.random_dag_adj, adj)
        self.assertListEqual(self.node_names, nodes)

        # To / from GML
        self.assertEqual(cg.to_gml_string(), CausalGraph.from_gml_string(cg.to_gml_string()).to_gml_string())
        self.assertEqual(cg, CausalGraph.from_gml_string(cg.to_gml_string()))
        self.assertTrue(cg.__eq__(CausalGraph.from_gml_string(cg.to_gml_string()), deep=True))

        # To / from Skeleton
        # Obviously this will lose directional information so just check skeletons are same.
        self.assertEqual(cg.skeleton, CausalGraph.from_skeleton(cg.skeleton).skeleton)
        self.assertTrue(cg.skeleton.__eq__(CausalGraph.from_skeleton(cg.skeleton).skeleton, deep=True))
        self.assertNotEqual(cg, CausalGraph.from_skeleton(cg.skeleton))

    def test_time_series_causal_graph(self):
        cg = CausalGraph.from_networkx(self.random_dag)
        tscg = TimeSeriesCausalGraph.from_causal_graph(cg)

        # To / from dict and from_causal_graph
        cg_dict = cg.to_dict()
        tscg_dict = tscg.to_dict()

        self.assertNotEqual(cg, tscg)
        self.assertNotEqual(cg_dict, tscg_dict)  # due to node class in dict

        self.assertEqual(cg, CausalGraph.from_dict(cg_dict))
        self.assertTrue(cg.__eq__(CausalGraph.from_dict(cg_dict), deep=True))

        self.assertEqual(cg, CausalGraph.from_dict(tscg_dict))
        # This is False as meta on the second graph's nodes will have the time info
        self.assertFalse(cg.__eq__(CausalGraph.from_dict(tscg_dict), deep=True))

        self.assertNotEqual(tscg, CausalGraph.from_dict(cg_dict))
        self.assertNotEqual(tscg, CausalGraph.from_dict(tscg_dict))

        self.assertEqual(tscg, TimeSeriesCausalGraph.from_dict(cg_dict))
        self.assertTrue(tscg.__eq__(TimeSeriesCausalGraph.from_dict(cg_dict), deep=True))

        self.assertEqual(tscg, TimeSeriesCausalGraph.from_dict(tscg_dict))
        self.assertTrue(tscg.__eq__(TimeSeriesCausalGraph.from_dict(tscg_dict), deep=True))

        # To / from networkx
        self.assertTrue(
            networkx.utils.graphs_equal(
                TimeSeriesCausalGraph.from_networkx(self.random_dag).to_networkx(), self.random_dag
            )
        )
        tscg = TimeSeriesCausalGraph.from_networkx(self.random_dag)
        self.assertEqual(tscg, TimeSeriesCausalGraph.from_networkx(self.random_dag))
        self.assertTrue(tscg.__eq__(TimeSeriesCausalGraph.from_networkx(self.random_dag), deep=True))

        # To / from dict
        self.assertDictEqual(tscg.to_dict(), TimeSeriesCausalGraph.from_dict(tscg.to_dict()).to_dict())
        self.assertEqual(tscg, TimeSeriesCausalGraph.from_dict(tscg.to_dict()))
        self.assertTrue(tscg.__eq__(TimeSeriesCausalGraph.from_dict(tscg.to_dict()), deep=True))

        # To / from adjacency matrix / to numpy
        # Note that the adjacency matrix and nodes from TimeSeriesCG are sorted by node names.
        # This is why in the setUp we must ensure that the node names are already sorted. Otherwise, we would need
        # to reorder the adjacency matrix for comparison.
        # It is OK though as tscg.adjacency_matrix and tscg.get_node_names() respect the same order.
        self.assertEqual(
            tscg, TimeSeriesCausalGraph.from_adjacency_matrix(self.random_dag_adj, node_names=self.node_names)
        )
        self.assertTrue(
            tscg.__eq__(
                TimeSeriesCausalGraph.from_adjacency_matrix(self.random_dag_adj, node_names=self.node_names), deep=True
            )
        )
        numpy.testing.assert_array_equal(self.random_dag_adj, tscg.adjacency_matrix)
        numpy.testing.assert_array_equal(
            self.random_dag_adj,
            TimeSeriesCausalGraph.from_adjacency_matrix(
                self.random_dag_adj, node_names=self.node_names
            ).adjacency_matrix,
        )

        nodes = cg.get_node_names()
        self.assertListEqual(self.node_names, nodes)
        adj, nodes = tscg.to_numpy()
        numpy.testing.assert_array_equal(self.random_dag_adj, adj)
        self.assertListEqual(self.node_names, nodes)

        nodes = TimeSeriesCausalGraph.from_adjacency_matrix(
            self.random_dag_adj, node_names=self.node_names
        ).get_node_names()
        self.assertListEqual(self.node_names, nodes)
        adj, nodes = TimeSeriesCausalGraph.from_adjacency_matrix(
            self.random_dag_adj, node_names=self.node_names
        ).to_numpy()
        numpy.testing.assert_array_equal(self.random_dag_adj, adj)
        self.assertListEqual(self.node_names, nodes)

        # To / from GML
        self.assertEqual(
            tscg.to_gml_string(), TimeSeriesCausalGraph.from_gml_string(tscg.to_gml_string()).to_gml_string()
        )
        self.assertEqual(tscg, TimeSeriesCausalGraph.from_gml_string(tscg.to_gml_string()))
        self.assertTrue(tscg.__eq__(TimeSeriesCausalGraph.from_gml_string(tscg.to_gml_string()), deep=True))

        # To / from Skeleton
        # Obviously this will lose directional information so just check skeletons are same.
        self.assertEqual(tscg.skeleton, TimeSeriesCausalGraph.from_skeleton(tscg.skeleton).skeleton)
        self.assertTrue(tscg.skeleton.__eq__(TimeSeriesCausalGraph.from_skeleton(tscg.skeleton).skeleton, deep=True))
        self.assertNotEqual(tscg, TimeSeriesCausalGraph.from_skeleton(tscg.skeleton))

        # To / from adjacency matrices / to numpy_by_lag
        tscg.add_edge('x01 lag(n=1)', 'x01')
        adj_dict = tscg.adjacency_matrices
        tscg = TimeSeriesCausalGraph.from_adjacency_matrices(adj_dict, variable_names=self.node_names)
        self.assertEqual(adj_dict.keys(), tscg.adjacency_matrices.keys())
        for k in adj_dict:
            numpy.testing.assert_array_equal(adj_dict[k], tscg.adjacency_matrices[k])
        numpy.testing.assert_array_equal(self.random_dag_adj, tscg.adjacency_matrices[0])

        nodes = tscg.get_node_names()
        self.assertNotEqual(self.node_names, nodes)  # as we have one extra node now
        adj, nodes = tscg.to_numpy_by_lag()
        self.assertEqual(adj_dict.keys(), adj.keys())
        for k in adj_dict:
            numpy.testing.assert_array_equal(adj_dict[k], adj[k])
        numpy.testing.assert_array_equal(self.random_dag_adj, adj[0])
        self.assertListEqual(self.node_names, nodes)  # as nodes are just variables so the lagged node name is removed
