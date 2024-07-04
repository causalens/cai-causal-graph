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
from __future__ import annotations

import itertools
from collections import defaultdict
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Type, Union, cast

import networkx
import numpy

from cai_causal_graph import __version__ as CAUSAL_GRAPH_VERSION
from cai_causal_graph.exceptions import CausalGraphErrors
from cai_causal_graph.graph_components import Edge, Node
from cai_causal_graph.interfaces import CanDictDeserialize, CanDictSerialize, HasIdentifier, HasMetadata
from cai_causal_graph.type_definitions import PAIR_T, EdgeType, NodeLike, NodeVariableType, validate_pair_type
from cai_causal_graph.utils import pairwise


def to_list(var: Any) -> List[Any]:
    """Helper to make sure a var is always a list."""
    return var if isinstance(var, list) else [var]


def reset_cached_attributes_decorator(func: Callable) -> Callable:
    """
    Decorator to reset cached attributes of `cai_causal_graph.causal_graph.CausalGraph`.

    Whenever a function is called that changes the graph, we need to reset these attributes.
    """

    @wraps(func)
    def wrapper(self: CausalGraph, *args, **kwargs) -> Any:
        function = func(self, *args, **kwargs)
        self._reset_cached_attributes()
        return function

    return wrapper


class Skeleton(CanDictSerialize, CanDictDeserialize):
    """A utility class to obtain the skeleton of the causal graph including nodes, edges and adjacency matrix."""

    def __init__(self, graph: CausalGraph):
        """
        :param graph: The `cai_causal_graph.causal_graph.CausalGraph` instance to which this
            `cai_causal_graph.causal_graph.Skeleton` instance corresponds.
        """
        self._graph = graph

    def __eq__(self, other: object, deep: bool = False) -> bool:
        """
        Check if equal to another `cai_causal_graph.causal_graph.Skeleton`.

        Checks if all nodes and edges are equal.

        :param other: The other `cai_causal_graph.causal_graph.Skeleton` to compare to.
        :param deep: If `True`, also does deep equality checks on all the nodes and edges. Default is `False`.
        :return: `True` if equal, `False` otherwise.
        """
        if not isinstance(other, Skeleton):
            return False

        # Check that the number of nodes and edges agrees
        if len(self.nodes) != len(other.nodes) or len(self.edges) != len(other.edges):
            return False

        # Check that the set of node names matches
        if set(self._graph.get_node_names()) != set(other._graph.get_node_names()):
            return False

        # Check that the set of edges matches
        edge_pairs = {frozenset(e.get_edge_pair()) for e in self.edges}
        other_edge_pairs = {frozenset(e.get_edge_pair()) for e in other.edges}
        if edge_pairs != other_edge_pairs:
            return False

        # Check that edges and nodes are equivalent (ignores meta data)
        for node in self.nodes:
            if not node.__eq__(other.get_node(node.identifier), deep):
                return False

        for edge in self.edges:
            # Symmetric edges, e.g. x -- y, should be the same as their reverse, e.g. y -- x
            try:
                other_edge = other.get_edge(edge.source.identifier, edge.destination.identifier)
            except AssertionError:  # get_edge raises AssertionError in this case
                other_edge = other.get_edge(edge.destination.identifier, edge.source.identifier)

            # Check if equal
            if not edge.__eq__(other_edge, deep):
                return False

        return True

    def __ne__(self, other: object) -> bool:
        """Check if the skeleton is not equal to another skeleton."""
        return not (self == other)

    def __iter__(self) -> Iterator:
        """Return item,value tuples of the content of `self.to_dict()`."""
        for k, v in self.to_dict().items():
            yield k, v

    @property
    def nodes(self) -> List[Node]:
        """Return a list of all nodes."""
        # Instantiate new nodes to remove any edge information.
        return [
            self._graph._NodeCls(n.identifier, meta=n.meta, variable_type=n.variable_type) for n in self._graph.nodes
        ]

    def get_node(self, identifier: NodeLike) -> Node:
        """Return node that matches the identifier."""
        node_id = Node.identifier_from(identifier)
        matching_nodes = [node for node in self.nodes if node.identifier == node_id]
        assert len(matching_nodes) > 0, f'No node found matching identifier {node_id}.'
        assert len(matching_nodes) < 2, f'Found more than one matching node for identifier {node_id}: {matching_nodes}!'
        return matching_nodes[0]

    def get_node_names(self) -> List[str]:
        """Return a list of each node's identifier."""
        return [node.identifier for node in self._graph.nodes]

    def node_exists(self, identifier: NodeLike) -> bool:
        """Check if node exists."""
        node_id = Node.identifier_from(identifier)
        return node_id in self.get_node_names()

    @property
    def edges(self) -> List[Edge]:
        """
        Return a list of all edges. All edges will be of the type
        `cai_causal_graph.type_definitions.EdgeType.UNDIRECTED_EDGE` because this class represents a skeleton, which
        only has undirected edges and no directed edges.
        """
        # instantiate new edges to enforce undirected edge types
        return [
            Edge(e.source, e.destination, edge_type=EdgeType.UNDIRECTED_EDGE, meta=e.meta) for e in self._graph.edges
        ]

    def get_edge(
        self,
        source: str,
        destination: str,
    ) -> Edge:
        """
        Return edge by source and destination.

        As all edges are undirected, the order of source and destination does not matter.
        """
        edge_pair = (source, destination)
        # Also check (destination, source) as order does not matter as all edges are undirected.
        matching_edges = [edge for edge in self.edges if edge.get_edge_pair() in [edge_pair, (destination, source)]]
        assert len(matching_edges) > 0, f'No edge found matching edge pair {edge_pair}.'
        assert (
            len(matching_edges) < 2
        ), f'Found more than one matching edge for edge pair {edge_pair}: {matching_edges}!'
        return matching_edges[0]

    def edge_exists(
        self,
        source: str,
        destination: str,
    ) -> bool:
        """Return true if edge exists."""
        edge_pairs = self.get_edge_pairs()
        return (source, destination) in edge_pairs or (destination, source) in edge_pairs

    def get_edge_pairs(self) -> List[PAIR_T]:
        """Return all edge pairs in the current graph."""
        return [edge.get_edge_pair() for edge in self.edges]

    def get_edge_by_pair(self, pair: PAIR_T) -> Edge:
        """Return edge by pair identifier."""
        validate_pair_type(pair)
        return self.get_edge(pair[0], pair[1])

    def is_edge_by_pair(self, pair: PAIR_T) -> bool:
        """Check if a given edge exists by pair identifier."""
        validate_pair_type(pair)
        return self.edge_exists(pair[0], pair[1])

    def get_neighbors(self, node: NodeLike) -> List[str]:
        """Get node identifiers for all neighbor nodes for a specific node."""
        node_id = Node.identifier_from(node)
        assert node_id in self.get_node_names(), f'Node with identifier {node_id} does not exist in this skeleton.'

        return self._graph.get_neighbors(node_id)

    def get_neighbor_nodes(self, node: NodeLike) -> List[Node]:
        """Get all neighbor nodes for a specific node."""
        neighbor_ids = self.get_neighbors(node)
        return [node for node in self.nodes if node.identifier in neighbor_ids]

    def is_empty(self) -> bool:
        """Return `True` if there are no nodes and edges. `False` otherwise."""
        return len(self.nodes) == 0 and len(self.edges) == 0

    @property
    def adjacency_matrix(self) -> numpy.ndarray:
        """Return the adjacency matrix of the `cai_causal_graph.causal_graph.Skeleton` instance."""
        node_names = self._graph.get_node_names()
        adjacency = numpy.zeros((len(node_names), len(node_names)), dtype=int)
        for edge in self.edges:
            # get the source and destination of the edge
            source, destination = edge.get_edge_pair()
            # we only have undirected edges in the Skeleton, so no need to check for other types
            adjacency[node_names.index(source), node_names.index(destination)] = 1
            adjacency[node_names.index(destination), node_names.index(source)] = 1

        return adjacency

    @classmethod
    def from_adjacency_matrix(
        cls,
        adjacency: numpy.ndarray,
        node_names: Optional[List[Union[NodeLike, int]]] = None,
        graph_class: Union[Type[CausalGraph], None] = None,
    ) -> Skeleton:
        """
        Instantiate a `cai_causal_graph.causal_graph.Skeleton` object from an adjacency matrix.

        :param adjacency: The adjacency matrix.
        :param node_names: A list of the node names. These should be in the order of the rows and columns of the
            adjacency matrix.
        :param graph_class: The appropriate subclass of `cai_causal_graph.causal_graph.CausalGraph` to use. If `None`,
            which is the default, `cai_causal_graph.causal_graph.CausalGraph` will be used. This is to ensure the new
            `cai_causal_graph.causal_graph.Skeleton` object has the correct node class type.
        :return: A new `cai_causal_graph.causal_graph.Skeleton` based on the provided adjacency matrix.
        """
        if graph_class is not None:
            assert issubclass(
                graph_class, CausalGraph
            ), f'The provided graph_class is not a subclass of CausalGraph. Got type {type(graph_class)}.'
        else:
            graph_class = CausalGraph
        graph: CausalGraph = graph_class.from_adjacency_matrix(adjacency=adjacency, node_names=node_names)
        return cls(graph=graph)

    def to_dict(self, include_meta: bool = True) -> dict:
        """
        Serialize the `cai_causal_graph.causal_graph.Skeleton` instance to a dictionary.

        :param include_meta: Whether to include meta information about the skeleton in the dictionary. Default is
            `True`.
        :return: The dictionary representation of the `cai_causal_graph.causal_graph.Skeleton` instance.
        """
        nodes = {node.identifier: node.to_dict(include_meta=include_meta) for node in self.nodes}

        edges: Dict[str, Dict[str, dict]] = dict()
        for edge in self.edges:
            source, destination = edge.get_edge_pair()
            if source not in edges:
                edges[source] = dict()
            edges[source][destination] = edge.to_dict(include_meta=include_meta)

        return {
            'nodes': nodes,
            'edges': edges,
            'version': CAUSAL_GRAPH_VERSION,
        }

    def to_networkx(self) -> networkx.Graph:
        """Return a `networkx.Graph` corresponding to the `cai_causal_graph.causal_graph.Skeleton` instance."""
        # Create empty networkx.Graph object with current node names.
        networkx_graph = networkx.empty_graph(n=self._graph.get_node_names(), create_using=networkx.Graph)

        edge: Edge
        for edge in self.edges:
            networkx_graph.add_edge(edge.source.identifier, edge.destination.identifier)

        return networkx_graph

    def to_numpy(self) -> Tuple[numpy.ndarray, List[str]]:
        """
        Return a numpy array that represents the adjacency matrix of the `cai_causal_graph.causal_graph.Skeleton`
        instance.

        To avoid confusion, this method also returns a list of variables that corresponds to the order of columns in
        the numpy array.
        """
        return self.adjacency_matrix, self._graph.get_node_names()

    def to_gml_string(self) -> str:
        """
        Return a Graph Modelling Language (GML) string representative of the `cai_causal_graph.causal_graph.Skeleton`
        instance.
        """
        return '\n'.join(networkx.generate_gml(self.to_networkx()))

    @classmethod
    def from_dict(cls, d: dict, graph_class: Union[Type[CausalGraph], None] = None) -> Skeleton:
        """
        Instantiate a `cai_causal_graph.causal_graph.Skeleton` object from a dictionary.

        :param d: The dictionary representation of the skeleton.
        :param graph_class: The appropriate subclass of `cai_causal_graph.causal_graph.CausalGraph` to use. If `None`,
            which is the default, `cai_causal_graph.causal_graph.CausalGraph` will be used. This is to ensure the new
            `cai_causal_graph.causal_graph.Skeleton` object has the correct node class type.
        :return: A new `cai_causal_graph.causal_graph.Skeleton` based on the provided dictionary.
        """
        if graph_class is not None:
            assert issubclass(
                graph_class, CausalGraph
            ), f'The provided graph_class is not a subclass of CausalGraph. Got type {type(graph_class)}.'
        else:
            graph_class = CausalGraph
        return cls(graph=graph_class.from_dict(d))

    @classmethod
    def from_networkx(cls, g: networkx.Graph, graph_class: Union[Type[CausalGraph], None] = None) -> Skeleton:
        """
        Instantiate a `cai_causal_graph.causal_graph.Skeleton` object from a `networkx.Graph`.

        :param g: The `networkx.Graph` representing the skeleton.
        :param graph_class: The appropriate subclass of `cai_causal_graph.causal_graph.CausalGraph` to use. If `None`,
            which is the default, `cai_causal_graph.causal_graph.CausalGraph` will be used. This is to ensure the new
            `cai_causal_graph.causal_graph.Skeleton` object has the correct node class type.
        :return: A new `cai_causal_graph.causal_graph.Skeleton` based on the `networkx.Graph`.
        """
        if graph_class is not None:
            assert issubclass(
                graph_class, CausalGraph
            ), f'The provided graph_class is not a subclass of CausalGraph. Got type {type(graph_class)}.'
        else:
            graph_class = CausalGraph
        # Convert node names to strings.
        node_names: List[str] = [
            graph_class._NodeCls.identifier_from(graph_class.coerce_to_nodelike(node)) for node in g.nodes()
        ]
        return cls.from_adjacency_matrix(networkx.to_numpy_array(g), node_names, graph_class)  # type: ignore

    @classmethod
    def from_gml_string(cls, gml: str, graph_class: Union[Type[CausalGraph], None] = None) -> Skeleton:
        """
        Instantiate a `cai_causal_graph.causal_graph.Skeleton` object from a Graph Modelling Language (GML) string.

        :param gml: The GML string representing the skeleton.
        :param graph_class: The appropriate subclass of `cai_causal_graph.causal_graph.CausalGraph` to use. If `None`,
            which is the default, `cai_causal_graph.causal_graph.CausalGraph` will be used. This is to ensure the new
            `cai_causal_graph.causal_graph.Skeleton` object has the correct node class type.
        :return: A new `cai_causal_graph.causal_graph.Skeleton` based on the provided GML string.
        """
        g = networkx.parse_gml(gml)
        return cls.from_networkx(g, graph_class)

    def copy(self) -> Skeleton:
        """Copy a `cai_causal_graph.causal_graph.Skeleton` instance."""
        new_skeleton = self.__class__.from_dict(self.to_dict())
        assert isinstance(new_skeleton, self.__class__)  # for linting and sanity check
        return new_skeleton

    def __repr__(self) -> str:
        """Return a string description of the `cai_causal_graph.causal_graph.Skeleton` instance."""
        return (
            f'Skeleton('
            f'num_nodes={len(self.nodes)}, num_edges={len(self.edges)}, id={self.__hash__()}'
            f')\n'
            f'Nodes: {self.nodes}\nEdges: {self.edges}'
        )

    def details(self) -> str:
        """Return a detailed string description of the `cai_causal_graph.causal_graph.Skeleton` instance."""
        node_details = '\t' + '\n\t'.join([n.details().replace('\n', '\n\t') for n in self.nodes])
        edge_details = '\t' + '\n\t'.join([e.details().replace('\n', '\n\t') for e in self.edges])
        return f'{self.__repr__()}\n' f'Node Details:\n{node_details}\nEdge Details:\n{edge_details}'

    def __hash__(self) -> int:
        """Return a hash representation of the `cai_causal_graph.causal_graph.Skeleton` instance."""
        return hash(repr(self.to_dict()))


class CausalGraph(HasIdentifier, HasMetadata, CanDictSerialize, CanDictDeserialize):
    """A low-level class that uniquely defines the state of a causal graph."""

    _NodeCls: Type[Node] = Node
    _EdgeCls: Type[Edge] = Edge

    def __init__(
        self,
        input_list: Optional[List[NodeLike]] = None,
        output_list: Optional[List[NodeLike]] = None,
        fully_connected: bool = True,
        meta: Optional[dict] = None,
    ):
        """
        The `cai_causal_graph.causal_graph.CausalGraph` class manages and defines the state of a causal graph.

        It encodes and allows for easy visualization of causal relationships, which are presented as edges between
        nodes where each node is a variable from a data set.

        It can deal with a variety of different edge types such as directed, undirected, bidirected and unknown edges.
        Therefore, this class can represent a large variety of causal graph classes, such as (among others):
            - Directed acyclic graph (DAG)
            - Completed partially directed acyclic graph (CPDAG)
            - Maximal ancenstral graph (MAG)
            - Partial ancestral graph (PAG)

        Example:

            >>> from cai_causal_graph import CausalGraph
            >>>
            >>> # create an empty graph by not providing a `input_list` or `output_list`
            >>> causal_graph = CausalGraph()
            >>>
            >>> # add nodes to the causal graph
            >>> node_names = ['input1', 'input2', 'input3', 'target1', 'target2']
            >>> causal_graph.add_nodes_from(node_names)
            >>>
            >>> # add a directed edge from 'input1' to 'output1'
            >>> causal_graph.add_edge('input1', 'output1')

            By default, any edges added will be directed edges, e.g. 'input1' -> 'target1' for the edge added above. It
            is possible to specify different edge types via the `edge_type` argument. For the full list of edge types,
            see `cai_causal_graph.type_definitions.EdgeType`. For instance, an undirected edge can be added:

            >>> # add an undirected edge between 'input1' and 'input2'
            >>> causal_graph.add_edge('input1', 'input2', edge_type=EdgeType.UNDIRECTED_EDGE)

            Setting `fully_connected=True` (default) and providing an `input_list` and `n `output_list` during
            construction, automatically creates a fully-connected bipartite directed causal graph. This means that all
            inputs are automatically connected to all outputs by means of directed edges. But there will be no edges
            between inputs and no edges between outputs; this will be a simple two layer graph. For example,

            >>> from cai_causal_graph import CausalGraph
            >>>
            >>> input_list = ['input1', 'input2']
            >>> output_list = ['output1', 'output2']
            >>>
            >>> # Create a fully connected graph from inputs to outputs
            >>> causal_graph = CausalGraph(
            >>>     input_list=input_list,
            >>>     output_list=output_list,
            >>> )

            Finally, it is straightforward to export an instantiated `cai_causal_graph.causal_graph.CausalGraph` to a
            serializable dictionary,

            >>> causal_graph_dict = causal_graph.to_dict()

        :param input_list: List of objects coercable to `cai_causal_graph.graph_components.Node`. Each element is
            treated as an input node, if `full_connected` parameter is `True`. Otherwise, the nodes will simply be
            added to the graph with no edges.
        :param output_list:  List of objects coercable to `cai_causal_graph.graph_components.Node`. Each element is
            treated as an output node, if `fully_connected` parameter is `True`. Otherwise, the nodes will simply be
            added to the graph with no edges.
        :param fully_connected: If set to `True` (default), create a fully-connected bipartite directed graph, with all
            inputs connected to all outputs. If no `input_list` and no `output_list` is provided, an empty graph will
            be created. If either or both are provided, but this is `False`, then the nodes will be added but not
            connected by edges.
        :param meta: Any metadata defined on the graph. The keys must be strings, but no requirement is placed on the
            values of the dictionary. Default is `None`. If passed, meta is shallow-copied.
        """
        self._nodes_by_identifier: Dict[str, Node] = dict()
        self._edges_by_source: Dict[str, Dict[str, Edge]] = defaultdict(dict)
        self._edges_by_destination: Dict[str, Dict[str, Edge]] = defaultdict(dict)

        # Add nodes.
        if input_list is not None:
            self.add_nodes_from(input_list)
        if output_list is not None:
            self.add_nodes_from(output_list)

        # create a fully-connected causal graph if specified, using only directed edges
        if fully_connected and input_list is not None and output_list is not None:
            self.add_fully_connected_nodes(input_list, output_list)
        # No else needed as no edges need to be added.

        # construct the skeleton and separation sets
        self._skeleton = Skeleton(graph=self)
        self._sepsets: dict = dict()

        super(HasIdentifier, self).__init__(meta=meta)

        # cached attributes
        self._is_dag: Optional[bool] = None
        self._networkx: Optional[networkx.Graph] = None
        self._is_fully_directed_cached: Optional[bool] = None
        self._is_fully_undirected_cached: Optional[bool] = None
        self._adjacency: Optional[numpy.ndarray] = None

    def __copy__(self) -> CausalGraph:
        """Copy a `cai_causal_graph.causal_graph.CausalGraph` instance."""
        return self.copy()

    def __deepcopy__(self, memo) -> CausalGraph:
        """Deep-copy a `cai_causal_graph.causal_graph.CausalGraph` instance."""
        return self.copy()

    def __getitem__(self, item: Union[NodeLike, Tuple[NodeLike, NodeLike]]) -> Union[Node, Edge]:
        """Get a node (single identifier) or edge (tuple of identifiers)."""
        if not isinstance(item, tuple):
            return self.get_node(item)
        else:
            if len(item) != 2:
                raise TypeError(
                    f'The provided item must either be `NodeLike` (i.e. to return a single node) or a tuple of two '
                    f'`NodeLike` objects (i.e. to return an edge), got {item}.'
                )
            return self.get_edge(item[0], item[1])

    def __iter__(self) -> Iterator:
        """Return item,value tuples of the content of `self.to_dict()`."""
        for k, v in self.to_dict().items():
            yield k, v

    def __eq__(self, other: object, deep: bool = False) -> bool:
        """
        Check if equal to another `cai_causal_graph.causal_graph.CausalGraph`.

        Checks if all nodes and edges are equal.

        :param other: The other `cai_causal_graph.causal_graph.CausalGraph` to compare to.
        :param deep: If `True`, also does deep equality checks on all the nodes and edges. Default is `False`.
        :return: `True` if equal, `False` otherwise.
        """
        if not isinstance(other, self.__class__):
            return False

        # Check that the number of nodes and edges agrees
        if len(self.nodes) != len(other.nodes) or len(self.edges) != len(other.edges):
            return False

        # Check that the set of node names matches
        if set(self.get_node_names()) != set(other.get_node_names()):
            return False

        # Check that the set of edges matches
        edge_pairs = {frozenset(e.get_edge_pair()) for e in self.edges}
        other_edge_pairs = {frozenset(e.get_edge_pair()) for e in other.edges}
        if edge_pairs != other_edge_pairs:
            return False

        # Check that edges and nodes are equivalent (ignores meta data)
        for node in self.nodes:
            if not node.__eq__(other.get_node(node.identifier), deep):
                return False

        for edge in self.edges:
            # Symmetric edges, e.g. x -- y, should be the same as their reverse, e.g. y -- x
            try:
                other_edge = other.get_edge(edge.source.identifier, edge.destination.identifier)
            # get_edge can raise KeyError when trying to index the dict of edges, but it can also raise
            # CausalGraphErrors.EdgeDoesNotExistError.
            except (KeyError, CausalGraphErrors.EdgeDoesNotExistError):
                other_edge = other.get_edge(edge.destination.identifier, edge.source.identifier)

            # Check if equal
            if not edge.__eq__(other_edge, deep):
                return False

        return True

    def __ne__(self, other: object) -> bool:
        """Check if the graph is not equal to another graph."""
        return not (self == other)

    @property
    def nodes(self) -> List[Node]:
        """Return a list of nodes."""
        return self.get_nodes()

    @property
    def edges(self) -> List[Edge]:
        """Return a list of edges."""
        return self.get_edges()

    @property
    def skeleton(self) -> Skeleton:
        """Return a `cai_causal_graph.causal_graph.Skeleton` instance of the causal graph."""
        return self._skeleton

    @property
    def adjacency_matrix(self) -> numpy.ndarray:
        """
        Return the adjacency matrix of the causal graph.

        The column order is dependent upon the ordering of node names. To obtain both, adjacency matrix and node names
        in corresponding order use the to_numpy method instead.
        """
        if self._adjacency is not None:
            return deepcopy(self._adjacency)

        # check that the graph only has directed and undirected edges
        nodes_indices_map = {node: i for i, node in enumerate(self.get_node_names())}
        adj = numpy.zeros((len(self.nodes), len(self.nodes)), dtype=int)
        for edge in self.edges:
            source, destination = edge.get_edge_pair()
            if edge.get_edge_type() == EdgeType.DIRECTED_EDGE:
                adj[nodes_indices_map[source], nodes_indices_map[destination]] = 1
            elif edge.get_edge_type() == EdgeType.UNDIRECTED_EDGE:
                adj[nodes_indices_map[source], nodes_indices_map[destination]] = 1
                adj[nodes_indices_map[destination], nodes_indices_map[source]] = 1
            else:
                raise TypeError(
                    f'Adjacency matrices can only be computed if the CausalGraph instance solely contains directed and '
                    f'undirected edges. Got {edge.get_edge_type()} for the edge {edge.descriptor}.'
                )
        self._adjacency = adj
        return self._adjacency

    @property
    def sepsets(self) -> dict:
        """
        Return a dictionary of separation sets between variables, potentially obtained via conditional independence
        testing or via domain knowledge.
        """
        return self._sepsets

    @property
    def identifier(self) -> str:
        """
        Return a unique identifier for a graph by concatenating all node identifiers in topological order. If the causal
        graph is not a DAG, in which case the topological order is undefined, a sorted list of node names is returned.
        """
        if self.is_dag():
            identifier = '<' + '>_<'.join(networkx.topological_sort(self.to_networkx())) + '>'
        else:
            identifier = '<' + '>_<'.join(sorted(self.get_node_names())) + '>'

        return identifier

    def get_identifier(self) -> str:
        """Return the graph identifier."""
        return self.identifier

    def is_dag(self) -> bool:
        """Check whether the `cai_causal_graph.causal_graph.CausalGraph` instance is a Directed Acyclic Graph (DAG)."""
        if self._is_dag is None:
            self._is_dag = (
                networkx.is_directed_acyclic_graph(self.to_networkx()) if self._is_fully_directed() else False
            )
        return self._is_dag

    def is_empty(self) -> bool:
        """Return True if there are no nodes and edges. False otherwise."""
        return len(self.nodes) == 0 and len(self.edges) == 0

    @property
    def has_non_serializable_metadata(self) -> bool:
        """
        Return `True` whether the graph or any node or edge contains non-JSON serializable metadata.

        By default, this returns `False`. Please note that by default `CausalGraph` does not perform any checks
        on the metadata. Instead, this method enables any derived classes to implement custom checks.
        """
        return False

    def _check_node_exists(self, identifier: NodeLike) -> str:
        """Validate and return the node identifier."""
        identifier = self._NodeCls.identifier_from(identifier)

        if identifier in self._nodes_by_identifier:
            raise CausalGraphErrors.NodeDuplicatedError(f'Node already exists: {identifier}')
        return identifier

    def _prepare_nodes(self, source: NodeLike, destination: NodeLike) -> Tuple[Node, Node]:
        """
        Prepare the source and destination nodes for adding an edge and return the source and destination nodes.

        It will add the nodes to the graph if they do not exist yet. Additionally, it will check that, if the edge
        already exists, it is of the same type as the edge to be added.

        :param source: The source node.
        :param destination: The destination node.
        :return: A tuple of source and destination nodes.
        """
        source_meta = source.get_metadata() if isinstance(source, HasMetadata) else None
        destination_meta = destination.get_metadata() if isinstance(destination, HasMetadata) else None

        if not isinstance(source, Node):
            source = self._NodeCls.identifier_from(source)
        if not isinstance(destination, Node):
            destination = self._NodeCls.identifier_from(destination)

        # check that the source is not equal to destination
        if source == destination:
            raise CausalGraphErrors.CyclicConnectionError(
                f'Adding an edge from {source} to {destination} would create a self-loop, no matter the edge type. '
                f'This is currently not supported.'
            )

        source_nodes = self.get_nodes(source)
        destination_nodes = self.get_nodes(destination)
        edges = self.get_edges(source, destination)

        if len(source_nodes) != 1:
            # The node was not explicitly defined by the user, so we will add it implicitly based on the edge info
            if isinstance(source, Node):
                self.add_node(node=source)
            else:
                self.add_node(source, meta=source_meta)
            source_nodes = self.get_nodes(source)
        if len(destination_nodes) != 1:
            # The node was not explicitly defined by the user, so we will add it implicitly based on the edge info
            if isinstance(destination, Node):
                self.add_node(node=destination)
            else:
                self.add_node(destination, meta=destination_meta)
            destination_nodes = self.get_nodes(destination)
        if len(edges) != 0:
            # We don't allow implicit edge override. The user should delete the node or modify it.
            raise CausalGraphErrors.EdgeDuplicatedError(
                f'An edge already exists between {source} and {destination}. '
                f'Please modify or delete this and then create the new edge explicitly.'
            )
        return source_nodes[0], destination_nodes[0]

    def _set_edge(self, edge: Edge, validate: bool = True):
        """Set the edge in the graph."""
        source, destination = self._NodeCls.identifier_from(edge.source), self._NodeCls.identifier_from(
            edge.destination
        )

        self._edges_by_source[source][destination] = edge
        self._edges_by_destination[destination][source] = edge

        # only add edges to inbound / outbound edges of a node if the specified edge type is directed
        if edge._edge_type == EdgeType.DIRECTED_EDGE:
            self._nodes_by_identifier[destination]._add_inbound_edge(edge)
            self._nodes_by_identifier[source]._add_outbound_edge(edge)

        # check that there are no cycles of directed edges
        if validate:
            try:
                self._assert_node_does_not_depend_on_itself(destination)
            except AssertionError:
                self.delete_edge(source, destination)
                raise CausalGraphErrors.CyclicConnectionError(
                    f'Adding an edge from {source} to {destination} would create a cyclic connection.'
                )

    def _is_fully_directed(self) -> bool:
        """
        Check whether the `cai_causal_graph.causal_graph.CausalGraph` instance only contains directed edges, i.e.,
        all edges are directed.
        """
        if self._is_fully_directed_cached is None:
            self._is_fully_directed_cached = all(edge.get_edge_type() == EdgeType.DIRECTED_EDGE for edge in self.edges)
        return self._is_fully_directed_cached

    def _is_fully_undirected(self) -> bool:
        """
        Check whether the `cai_causal_graph.causal_graph.CausalGraph` instance only contains undirected edges, i.e.,
        all edges are undirected.
        """
        if self._is_fully_undirected_cached is None:
            self._is_fully_undirected_cached = all(
                edge.get_edge_type() == EdgeType.UNDIRECTED_EDGE for edge in self.edges
            )
        return self._is_fully_undirected_cached

    def _is_directed_and_or_undirected_error_message(self, is_one_kind_only: bool = True) -> str:
        """
        Validate that the `cai_causal_graph.causal_graph.CausalGraph` instance is fully undirected or fully directed
        and return where not.

        :param is_one_kind_only: If True, validates that all edges are one of directed or undirected only. If False,
            OK to have a mix of the two (directed + undirected), but still no other edge types.
        :return: String with information where there are issues. Can be used to raise an informative error.
        """
        error_msg = ''

        # Check if graph is all undirected or all directed. Cannot convert otherwise.
        if not self._is_fully_directed() and not self._is_fully_undirected():
            # Find the issue to give an informative error.
            bidirected_edges = self.get_bidirected_edges()
            if len(bidirected_edges) > 0:
                error_msg += f'CausalGraph contains the following bidirected edges: {bidirected_edges}.\n'
            unknown_edges = self.get_unknown_edges()
            if len(unknown_edges) > 0:
                error_msg += f'CausalGraph contains the following unknown edges: {unknown_edges}.\n'
            unknown_directed_edges = self.get_unknown_directed_edges()
            if len(unknown_directed_edges) > 0:
                error_msg += f'CausalGraph contains the following unknown-directed edges: {unknown_directed_edges}.\n'
            unknown_undirected_edges = self.get_unknown_undirected_edges()
            if len(unknown_undirected_edges) > 0:
                error_msg += (
                    f'CausalGraph contains the following unknown-undirected edges: {unknown_undirected_edges}.\n'
                )

            if is_one_kind_only:
                # See if mix of directed and undirected.
                directed_edges = self.get_directed_edges()
                undirected_edges = self.get_undirected_edges()
                if len(directed_edges) > 0 and len(undirected_edges) > 0:
                    # Won't list them and tell user to simply call method to get them.
                    error_msg += (
                        'CausalGraph contains a mix of directed and undirected edges. Call `get_directed_edges` and '
                        '`get_undirected_edges` to see them.\n'
                    )
                # No else needed as only one set of edge types.
            # No else needed as we don't care if it has a mix of directed and undirected.

        return error_msg

    def get_node(self, identifier: NodeLike) -> Node:
        """Return a node based on its identifier."""
        return self._nodes_by_identifier[self._NodeCls.identifier_from(identifier)]

    def get_nodes(self, identifier: Optional[Union[NodeLike, List[NodeLike]]] = None) -> List[Node]:
        """
        Return nodes matching the given identifier(s).

        If an identifier is provided then only that node will be returned (if it exists; an empty list will be returned
        otherwise). If a list of identifiers is provided, then a corresponding list of nodes will be returned. If the
        identifier is left as `None` (default) then all nodes will be returned.

        :param identifier: String or list of string identifier(s) of the node(s) to find, or `None` to return all nodes.
        :return: List of matching nodes. If a list of identifiers was provided, the order of nodes matches that of the
            provided identifiers. Otherwise, they are sorted alphabetically by identifier.
        """
        if identifier is not None:
            if isinstance(identifier, (str, HasIdentifier)):
                identifier = self._NodeCls.identifier_from(identifier)
                node_list = (
                    [self._nodes_by_identifier[identifier]] if identifier in self._nodes_by_identifier else list()
                )
            elif isinstance(identifier, list):
                node_list = list()
                for node_like_id in identifier:
                    node_id = self._NodeCls.identifier_from(node_like_id)
                    if node_id in self._nodes_by_identifier:
                        node_list.append(self._nodes_by_identifier[node_id])
                    else:
                        raise ValueError(
                            f'Provided a list of identifiers containing {node_id}, which is not in the causal graph.'
                        )
            else:
                raise TypeError(
                    f'Expected identifier to be a string, list of strings, or None. Got {type(identifier)}.'
                )
        else:
            node_list = list(map(self._nodes_by_identifier.__getitem__, sorted(list(self._nodes_by_identifier.keys()))))

        return node_list

    def get_node_names(self) -> List[str]:
        """Return a list of each node's identifier."""
        return sorted(self._nodes_by_identifier)  # sorting a dictionary returns a list of its keys sorted

    def get_inputs(self) -> List[Node]:
        """Get all nodes without ancestors."""
        return [node for node in self.get_nodes() if len(self.get_edges(destination=node.identifier)) == 0]

    def get_outputs(self) -> List[Node]:
        """Get all nodes without descendants."""
        return [node for node in self.get_nodes() if len(self.get_edges(source=node.identifier)) == 0]

    def node_exists(self, identifier: NodeLike) -> bool:
        """Check whether the specified node exists."""
        try:
            self.get_node(identifier)
            return True
        except KeyError:
            return False

    @reset_cached_attributes_decorator
    def add_node(
        self,
        /,
        identifier: Optional[NodeLike] = None,
        *,
        variable_type: NodeVariableType = NodeVariableType.UNSPECIFIED,
        meta: Optional[dict] = None,
        node: Optional[Node] = None,
    ) -> Node:
        """
        Add a node to the causal graph but do not connect it to anything.

        :param identifier: String that uniquely identifies the node within the causal graph. If a node with that
            identifier already exists within the network, an exception will be raised. Can be `None`, if a `node`
            parameter is specified.
        :param variable_type: The variable type that the node represents. The choices are available through the
            `cai_causal_graph.type_definitions.NodeVariableType` enum.
        :param meta: The meta values for the node. If passed explicitly, this metadata is not deepcopied.
        :param node: A `cai_causal_graph.graph_components.Node` node to be used to construct a new node. All the
            properties of the provided node will be deep copied to the constructed node, including metadata and
            variable type. If provided, then all other parameters to the method must not be specified. Default is
            `None`.
        :return: The created node.
        """
        if node is not None:
            assert (
                identifier is None and variable_type == NodeVariableType.UNSPECIFIED and meta is None
            ), 'If specifying `node` argument, all other arguments should not be specified.'
            identifier = node.identifier
            variable_type = node.variable_type
            meta = deepcopy(node.meta)
        else:
            assert identifier is not None, (
                'You must either specify an `identifier` of a node to add, '
                'or provide a constructed `Node` object using the `node` parameter.'
            )

        identifier = self._check_node_exists(identifier)

        node = self._NodeCls(identifier, meta=meta, variable_type=variable_type)

        self._nodes_by_identifier[identifier] = node

        return node

    @reset_cached_attributes_decorator
    def add_nodes_from(self, identifiers: List[NodeLike]):
        """
        A convenience method to add multiple nodes. Only allows to set up nodes with default setup.
        For more details how nodes are being set refer to CausalGraph.add_node() method.

        :param identifiers: List of valid node identifiers.
        """
        for identifier in identifiers:
            self.add_node(identifier)

    @reset_cached_attributes_decorator
    def add_fully_connected_nodes(self, inputs: List[NodeLike], outputs: List[NodeLike]):
        """Create directed edges between all inputs and all outputs."""
        for input_node in inputs:
            for output_node in outputs:
                self.add_edge(input_node, output_node, edge_type=EdgeType.DIRECTED_EDGE)

    @reset_cached_attributes_decorator
    def delete_node(self, identifier: NodeLike):
        """
        Delete a node from the causal graph. This also deletes all edges connecting to this node.

        :param identifier: String identifying the node to be deleted.
        """
        # get the identifier and node
        identifier = self._NodeCls.identifier_from(identifier)
        node = self.get_node(identifier)

        # get a list of edges to delete depending on whether they contain the node to be deleted
        edge_pairs_to_delete = [edge.get_edge_pair() for edge in self.edges if identifier in edge.get_edge_pair()]

        # delete edges irrespective of their edge type
        for edge_pair in edge_pairs_to_delete:
            self.delete_edge(*edge_pair)

        # delete the node
        self._nodes_by_identifier.pop(identifier)
        node.invalidate()

    @reset_cached_attributes_decorator
    def remove_node(self, identifier: NodeLike):
        """
        Remove a node from the causal graph. This also deletes all edges connecting to this node.

        :param identifier: String identifying the node to be deleted.
        """
        self.delete_node(identifier=identifier)

    @reset_cached_attributes_decorator
    def replace_node(
        self,
        /,
        node_id: NodeLike,
        new_node_id: Optional[NodeLike] = None,
        *,
        variable_type: NodeVariableType = NodeVariableType.UNSPECIFIED,
        meta: Optional[dict] = None,
    ):
        """
        Replace an existing node by a new one, copying over all inbound and outbound edges and removing original node.

        If any additional information, such as variable type or meta is specified, it is used for the new node.
        Otherwise, these are copied from the original node. If new node is not provided, original one is simply
        changed in-place.

        :param node_id: original node to be replaced and removed
        :param new_node_id: new node to replace the old one
        :param variable_type: The variable type that the node represents. The choices are available through the
            `cai_causal_graph.type_definitions.NodeVariableType` enum.
        :param meta: optional meta to be used for the new node
        """
        # ensure the new node does not exist and the original one does
        assert self.node_exists(node_id), f'Provided node {node_id} to be replaced does not exist'
        if new_node_id is not None:
            assert not self.node_exists(new_node_id), f'Cannot create a new node {new_node_id} as it already exists'
        else:
            # if new node is not passed, original node is simply edited in-place
            node = self.get_node(node_id)
            node.variable_type = variable_type if variable_type is not None else node.variable_type
            node.meta = meta if meta is not None else node.meta
            return

        # add a new node
        original_node = self.get_node(node_id)
        new_node = self.add_node(
            new_node_id,
            variable_type=variable_type if variable_type is not None else original_node.variable_type,
            meta=meta if meta is not None else original_node.meta,
        )

        # copy inbound edges
        for edge in self.get_edges(destination=original_node.identifier):
            self.add_edge(
                source=edge.source.identifier,
                destination=new_node.identifier,
                edge_type=edge.get_edge_type(),
                meta=edge.meta,
            )

        # copy outbound edges
        for edge in self.get_edges(source=original_node.identifier):
            self.add_edge(
                source=new_node.identifier,
                destination=edge.destination.identifier,
                edge_type=edge.get_edge_type(),
                meta=edge.meta,
            )

        # remove the original edge
        self.delete_node(original_node.identifier)

    def get_edge(self, /, source: NodeLike, destination: NodeLike, *, edge_type: Optional[EdgeType] = None) -> Edge:
        """Return an edge based on its source, destination and (optional) edge_type."""
        # get edge with the specified source and destination
        try:
            source_identifier = self._NodeCls.identifier_from(source)
            destination_identifier = self._NodeCls.identifier_from(destination)
            edge = self._edges_by_source[source_identifier][destination_identifier]
        except KeyError:
            raise CausalGraphErrors.EdgeDoesNotExistError(
                f'The specified edge '
                f'({self._NodeCls.identifier_from(source)}, {self._NodeCls.identifier_from(destination)}) '
                f'does not exist.'
            )

        # check if an edge type is specified and check that it matches
        if edge_type is not None:
            if edge_type == edge.get_edge_type():
                return edge
            else:
                raise CausalGraphErrors.EdgeDoesNotExistError(
                    f'The specified edge ({source_identifier} {edge_type} {destination_identifier}) does not exist. '
                    f'However, {edge.descriptor} does exist.'
                )
        else:
            return edge

    def get_edges(
        self,
        /,
        source: Optional[NodeLike] = None,
        destination: Optional[NodeLike] = None,
        *,
        edge_type: Optional[EdgeType] = None,
    ) -> List[Edge]:
        """
        Return edges matching the given identifiers.

        Both source and destination can either be provided or left as None. If either is provided, only edges
        originating from or terminating at that node will be returned. Otherwise, that identifier will match against
        edges originating from or terminating at any node. Specifying edge_type yields edges that only match the
        desired edge type. Otherwise, if set to None, edges with any edge type are returned.

        :param source: Identifier of the source node. If set to None (default), edges originating from any node are
            returned.
        :param destination: Identifier of the destination node. If set to None (default), edges terminating at any node
            are returned.
        :param edge_type: The edge type of the desired edges to be returned. If set to None (default), edges with any
            edge_type are returned.
        :return: List of matching edges, sorted alphabetically by source identifier and then by destination identifier.
        """
        if source is not None:
            source = self._NodeCls.identifier_from(source)
        if destination is not None:
            destination = self._NodeCls.identifier_from(destination)

        # retrieve edges based on source and destination
        edges: List[Edge]
        if (source is None) and (destination is None):
            edges = list()
            for source_node in sorted(self._edges_by_source.keys()):
                edges.extend(self.get_edges(source_node, None))
        elif (source is not None) and (destination is not None):
            edge = self._edges_by_source[source].get(destination)
            edges = [edge] if edge is not None else []
        elif (source is None) and (destination is not None):
            sorted_edge_ids = sorted(self._edges_by_destination[destination].keys())
            edges = [self._edges_by_destination[destination][edge] for edge in sorted_edge_ids]
        elif (source is not None) and (destination is None):
            sorted_edge_ids = sorted(self._edges_by_source[source].keys())
            edges = [self._edges_by_source[source][edge] for edge in sorted_edge_ids]
        else:
            raise CausalGraphErrors.EdgeInvalidError('Non-exhaustive conditions')

        # check that retrieved edges match the specified edge type
        if edge_type is not None:
            return [edge for edge in edges if edge.get_edge_type() == edge_type]
        else:
            return edges

    def get_edge_by_pair(self, pair: Tuple[NodeLike, NodeLike], edge_type: Optional[EdgeType] = None) -> Edge:
        """Return an edge based on a tuple of (source, destination) and an edge_type."""
        validate_pair_type(pair)
        return self.get_edge(pair[0], pair[1], edge_type=edge_type)

    def is_edge_by_pair(self, pair: Tuple[NodeLike, NodeLike], edge_type: Optional[EdgeType] = None) -> bool:
        """Check if a given edge exists by pair identifier."""
        validate_pair_type(pair)
        return self.edge_exists(pair[0], pair[1], edge_type=edge_type)

    def get_directed_edges(self) -> List[Edge]:
        """Returns a list of directed edges, e.g. `'X' -> 'Y'`, in the causal graph."""
        return self._get_edges_by_type(EdgeType.DIRECTED_EDGE)

    def get_undirected_edges(self) -> List[Edge]:
        """Returns a list of undirected edges, e.g. `'X' -- 'Y'`, in the causal graph."""
        return self._get_edges_by_type(EdgeType.UNDIRECTED_EDGE)

    def get_bidirected_edges(self) -> List[Edge]:
        """Returns a list of bidirectional edges, e.g. `'X' <-> 'Y'`,  in the causal graph."""
        return self._get_edges_by_type(EdgeType.BIDIRECTED_EDGE)

    def get_unknown_edges(self) -> List[Edge]:
        """Returns a list of edges that are unknown, e.g. `'X' oo 'Y'`, in the causal graph."""
        return self._get_edges_by_type(EdgeType.UNKNOWN_EDGE)

    def get_unknown_directed_edges(self) -> List[Edge]:
        """Returns a list of edges that are unknown-directed, e.g. `'X' o> 'Y'`, in the causal graph."""
        return self._get_edges_by_type(EdgeType.UNKNOWN_DIRECTED_EDGE)

    def get_unknown_undirected_edges(self) -> List[Edge]:
        """Returns a list of edges that are unknown-undirected, e.g. `'X' o- 'Y'`, in the causal graph."""
        return self._get_edges_by_type(EdgeType.UNKNOWN_UNDIRECTED_EDGE)

    def _get_edges_by_type(self, edge_type: EdgeType) -> List[Edge]:
        """
        Get a list of edges that have the provided type, e.g. `->`.

        :param edge_type: The type to query.
        """
        return [edge for edge in self.edges if edge.get_edge_type() == edge_type]

    def edge_exists(self, /, source: NodeLike, destination: NodeLike, *, edge_type: Optional[EdgeType] = None) -> bool:
        """Returns True if the edge exists. If edge_type is None (default), this ignores edge types."""
        try:
            edge = self.get_edge(source, destination)
            if edge_type is not None:
                return edge_type == edge.get_edge_type()
            else:
                return True
        except CausalGraphErrors.EdgeDoesNotExistError:
            return False

    def get_edge_pairs(self) -> List[PAIR_T]:
        """Return all edge pairs in the current graph."""
        return [edge.get_edge_pair() for edge in self.edges]

    @reset_cached_attributes_decorator
    def change_edge_type(self, source: NodeLike, destination: NodeLike, new_edge_type: EdgeType):
        """
        Change an edge type for a specific edge.

        Changing an edge type may affect the topology of the causal graph, as it changes the relationship between
        source and destination nodes.

        Please note that this method removes existing edge and replaces it with a new edge with an updated edge type,
        while keeping reference to all other attributes (e.g. metadata) unmodified.

        :param source: Source of the edge for which the type should be changed.
        :param destination: Destination of the edge for which the type should be changed.
        :param new_edge_type: New edge type of the edge. If the new edge type matches existing edge type, no action is
            performed.
        """
        source, destination = self._NodeCls.identifier_from(source), self._NodeCls.identifier_from(destination)
        edge = self.get_edge(source=source, destination=destination)
        if edge.get_edge_type() != new_edge_type:
            meta = edge.meta
            self.remove_edge(source=source, destination=destination, edge_type=edge.get_edge_type())
            self.add_edge(source=source, destination=destination, edge_type=new_edge_type, meta=meta)

    @reset_cached_attributes_decorator
    def add_edge(
        self,
        /,
        source: Optional[NodeLike] = None,
        destination: Optional[NodeLike] = None,
        *,
        edge_type: EdgeType = EdgeType.DIRECTED_EDGE,
        meta: Optional[dict] = None,
        edge: Optional[Edge] = None,
        validate: bool = True,
    ) -> Edge:
        """
        Add an edge from a source to a destination node with a specific edge type.

        If these two nodes are already connected in any way, then an error will be raised. An error will also be raised
        if the new edge would create a cyclic connection of directed edges. In this case, the
        `cai_causal_graph.causal_graph.CausalGraph` instance will be restored to its original state and the edge will
        not be added. It is possible to specify an edge type from source to destination as well, with the default being
        a forward directed edge, i.e., source -> destination.

        :param source: String identifying the node from which the edge will originate. Can be `None`, if an `edge`
            parameter is specified.
        :param destination: String identifying the node at which the edge will terminate. Can be `None`, if an `edge`
            parameter is specified.
        :param edge_type: The type of the edge to be added. Default is
            `cai_causal_graph.type_definitions.EdgeType.DIRECTED_EDGE`. See `cai_causal_graph.type_definitions.EdgeType`
            for the list of possible edge types.
        :param meta: The meta values for the edge. If passed explicitly, this metadata is not deepcopied.
        :param edge: A `cai_causal_graph.graph_components.Edge` edge to be used to construct a new edge. All the
            properties of the provided edge will be deep copied to the constructed edge, including metadata. If
            provided, then all other parameters to the method must not be specified. Default is `None`.
        :param validate: Whether to perform validation checks. The validation checks will raise if
            any cycles are introduced to the graph by adding the edge. There is no guarantees about the behavior of the
            resulting graph if this is disabled specifically to introduce cycles. This should only be used to speed up
            this method in situations where it is known the new edge will not add cycles, for example when copying a
            graph. Default is `True`.
        :return: The created edge object.
        """
        if edge is not None:
            assert (
                source is None and destination is None and edge_type == EdgeType.DIRECTED_EDGE and meta is None
            ), 'If specifying `edge` argument, all other arguments should not be specified.'
            source, destination = edge.source, edge.destination

            edge_type = edge.get_edge_type()
            meta = deepcopy(edge.meta)
        else:
            assert source is not None and destination is not None, (
                'You must either specify a `source` and `destination` of an edge to add, '
                'or provide a constructed `Edge` object using the `edge` parameter.'
            )

        source_node, destination_node = self._prepare_nodes(source, destination)

        edge = self._EdgeCls(source_node, destination_node, edge_type=edge_type)

        # Add any meta
        if meta is not None:
            edge.meta = meta

        self._set_edge(edge=edge, validate=validate)
        return edge

    @reset_cached_attributes_decorator
    def add_edges_from(self, pairs: List[Tuple[NodeLike, NodeLike]], validate: bool = True):
        """
        A convenience method to add multiple edges by specifying tuples of source and destination node identifiers.

        Only allows to set up edges with default setup. For more details on how edges are being set, refer to
        `cai_causal_graph.causal_graph.CausalGraph.add_edge` method.

        :param pairs: List of valid edge pairs, defined as tuples of `(source_identifier, destination_identifier)`.
        :param validate: Whether to perform validation checks. The validation checks will raise if
            any cycles are introduced to the graph by adding the edge. There is no guarantees about the behavior of the
            resulting graph if this is disabled specifically to introduce cycles. This should only be used to speed up
            this method in situations where it is known the new edge will not add cycles, for example when copying a
            graph. Default is `True`.
        """
        for pair in pairs:
            validate_pair_type(pair)
            self.add_edge(source=pair[0], destination=pair[1], validate=validate)

    @reset_cached_attributes_decorator
    def add_edges_from_paths(self, paths: Union[List[NodeLike], List[List[NodeLike]]], validate: bool = True):
        """
        A convenience method to add multiple edges by specifying a single or a list of paths.

        Importantly, the path can overlap with each other, or with existing causal graph structure. Meaning, if an
        edge specified on the path already exists, an error is not raised. For example, the following two
        `paths` parameters will add the same edges: `[['a','b','c'],['b','c','d']]` and `[['a','b','c'],['c','d']]`.

        However, conflicting paths will produce an error. For example, it is not possible to add the following two
        paths: `[['a','b','c'],['c','b','y']]`, because the first path defines an edge `('b','c')`, while the second
        defines the edge `('c','b')`.

        Only allows to set up edges with default setup. For more details on how edges are being set, refer to
        `cai_causal_graph.causal_graph.CausalGraph.add_edge` method.

        :param paths: A list of paths or a single path. A path is defined as a list of node identifiers, defining the
            causal path in a causal graph.
        :param validate: Whether to perform validation checks. The validation checks will raise if
            any cycles are introduced to the graph by adding the edge. There is no guarantees about the behavior of the
            resulting graph if this is disabled specifically to introduce cycles. This should only be used to speed up
            this method in situations where it is known the new edge will not add cycles, for example when copying a
            graph. Default is `True`.
        """
        assert len(paths) != 0, 'The `paths` parameter must not be an empty list.'
        if isinstance(paths[0], list):
            if not all(isinstance(path, list) for path in paths):
                raise TypeError(f'Expects `paths` to be either a list of paths or a single path. Got {paths}')

            for path in paths:
                self.add_edges_from_paths(paths=cast(list, path))
        else:
            # `paths` is guaranteed to be a single path by this point
            for source, destination in pairwise(paths):
                source, destination = cast(NodeLike, source), cast(NodeLike, destination)
                if not self.edge_exists(source=source, destination=destination):
                    validate_pair_type((source, destination))
                    self.add_edge(source=source, destination=destination, validate=validate)

    @reset_cached_attributes_decorator
    def add_edge_by_pair(
        self,
        pair: Tuple[NodeLike, NodeLike],
        edge_type: EdgeType = EdgeType.DIRECTED_EDGE,
        meta: Optional[dict] = None,
        validate: bool = True,
    ):
        """
        Add edge by pair identifier (source, destination).

        :param pair: Tuple to identify the source and destination of the edge.
        :param edge_type: The type of edge to add. Default is a directed edge
        :param meta: Any metadata to add to the edge. Default is `None` (no metadata).
        :param validate: Whether to perform validation checks. The validation checks will raise if
            any cycles are introduced to the graph by adding the edge. There is no guarantees about the behavior of the
            resulting graph if this is disabled specifically to introduce cycles. This should only be used to speed up
            this method in situations where it is known the new edge will not add cycles, for example when copying a
            graph. Default is `True`.
        """
        validate_pair_type(pair)
        self.add_edge(pair[0], pair[1], edge_type=edge_type, meta=meta, validate=validate)

    @reset_cached_attributes_decorator
    def remove_edge_by_pair(self, pair: Tuple[NodeLike, NodeLike], edge_type: Optional[EdgeType] = None):
        """Remove edge by pair identifier (source, destination)."""
        validate_pair_type(pair)
        self.delete_edge(pair[0], pair[1], edge_type=edge_type)

    @reset_cached_attributes_decorator
    def delete_edge(self, /, source: NodeLike, destination: NodeLike, *, edge_type: Optional[EdgeType] = None):
        """
        Delete an edge from the causal graph.

        This will raise an exception if the edge does not exist, or either of the nodes do not exist.

        :param source: Identifier of the node from which the edge originates.
        :param destination: Identifier of the node at which the edge terminates.
        :param edge_type: The edge type of the edge to be deleted. Default is None, in which case the type is ignored.
        """
        source, destination = Node.identifier_from(source), Node.identifier_from(destination)
        assert isinstance(source, str), 'Source identifier must be a string'
        assert isinstance(destination, str), 'Destination identifier must be a string'

        # Check if the source and destination nodes exist
        source_nodes = self.get_nodes(source)
        destination_nodes = self.get_nodes(destination)

        if len(source_nodes) != 1:
            raise CausalGraphErrors.NodeDoesNotExistError(f'Node not found: {source}')
        if len(destination_nodes) != 1:
            raise CausalGraphErrors.NodeDoesNotExistError(f'Node not found: {destination}')

        matching_edges = self.get_edges(source, destination, edge_type=edge_type)
        if len(matching_edges) != 1:
            if edge_type is None:
                raise CausalGraphErrors.EdgeDoesNotExistError(f'Edge not found: ({source}, {destination})')
            else:
                raise CausalGraphErrors.EdgeDoesNotExistError(f'Edge not found: ({source} {edge_type} {destination})')
        edge = matching_edges[0]

        # need to delete inbound / outbound edges if the edge type is ->
        if edge.get_edge_type() == EdgeType.DIRECTED_EDGE:
            edge.destination._delete_inbound_edge(edge)
            edge.source._delete_outbound_edge(edge)

        self._edges_by_source[source].pop(destination)
        self._edges_by_destination[destination].pop(source)

        self._clean_empty_edge_dictionaries()
        edge.invalidate()

    @reset_cached_attributes_decorator
    def remove_edge(self, /, source: NodeLike, destination: NodeLike, *, edge_type: Optional[EdgeType] = None):
        """Remove a specific edge by source and destination node identifiers, as well as edge type."""
        self.delete_edge(source=source, destination=destination, edge_type=edge_type)

    @reset_cached_attributes_decorator
    def replace_edge(
        self,
        /,
        source: NodeLike,
        destination: NodeLike,
        new_source: NodeLike,
        new_destination: NodeLike,
        *,
        edge_type: Optional[EdgeType] = None,
        meta: Optional[dict] = None,
    ):
        """
        Replace an existing edge by a new one.

        If any additional information, such as edge type or meta is specified, it is used for the new edge.
        Otherwise, these are copied from the original edge.

        Note that if you simply wish to change the type of an existing edge, you should use the
        `cai_causal_graph.causal_graph.CausalGraph.change_edge_type` method instead.

        :param source: Source of the edge to be replaced.
        :param destination: Destination of the edge to be replaced.
        :param new_source: Source of the new edge.
        :param new_destination: Destination of the new edge.
        :param edge_type: The edge type of the new edge. If `None` (default), the same edge type as the original edge
            will be used.
        :param meta: Optional meta to be used for the new edge. If `None` (default)`, the meta of the original edge
            will be used.
        """
        # ensure the new edge does not exist and the original one does
        if not self.edge_exists(source=source, destination=destination):
            raise CausalGraphErrors.EdgeDoesNotExistError(
                f'The provided edge ({source}, {destination}) to be replaced does not exist.'
            )
        if self.edge_exists(source=new_source, destination=new_destination):
            raise CausalGraphErrors.EdgeExistsError(
                f'Cannot create a new edge ({source}, {destination}) as it already exists.'
            )

        # get the existing edge and information
        source, destination = self._NodeCls.identifier_from(source), self._NodeCls.identifier_from(destination)
        edge = self.get_edge(source=source, destination=destination)
        edge_type = edge_type if edge_type is not None else edge.get_edge_type()
        meta = meta if meta is not None else edge.get_metadata()

        # remove the original edge first to avoid potential acyclicity errors with respect to the original edge
        self.delete_edge(source=source, destination=destination)

        # add a new edge
        self.add_edge(source=new_source, destination=new_destination, edge_type=edge_type, meta=meta)

    def get_neighbors(self, node: NodeLike) -> List[str]:
        """
        Get node identifiers for all neighbor nodes for a specific node.

        Note: It does not matter what the edge type is, as long as there is an edge between `node` and another node,
        that other node is considered its neighbor.
        """
        identifier = self._NodeCls.identifier_from(node)  # As subclasses override NodeCls, need correct identifier.

        assert identifier in self.get_node_names(), f'Node with identifier {identifier} does not exist in this graph.'

        # inbound undirected edges
        inbound = {e.destination.identifier for e in self.get_edges(source=identifier)}
        # outbound undirected edges
        outbound = {e.source.identifier for e in self.get_edges(destination=identifier)}

        return list(inbound.union(outbound) - {identifier})

    def get_neighbor_nodes(self, node: NodeLike) -> List[Node]:
        """
        Get all neighbor nodes for a specific node.

        Note: It does not matter what the edge type is, as long as there is an edge between `node` and another node,
        that other node is considered its neighbor.
        """
        return self.get_nodes(identifier=self.get_neighbors(node))  # type: ignore

    def get_children_nodes(self, node: NodeLike) -> List[Node]:
        """Get all children nodes for a specific node."""
        identifier = self._NodeCls.identifier_from(node)
        assert self.node_exists(node), (
            f'The provided node with identifier {identifier} is not present in the graph with nodes: '
            f'{self.get_node_names()}'
        )

        # get a list of outbound edges to the node
        outbound = self.get_node(identifier).get_outbound_edges()

        # Remove all nodes that are not connected to the query node (identify children nodes)
        return list({e.destination for e in outbound})

    def get_children(self, node: NodeLike) -> List[str]:
        """Get node identifiers for all children nodes for a specific node."""
        # Get identifier for each child node
        return [n.identifier for n in self.get_children_nodes(node)]

    def get_children_graph(self, node: NodeLike) -> CausalGraph:
        """
        Get a sub causal graph that only includes the children of a specific node and the node itself.

        This method returns a star graph, such that all edges are going from the provided node. Any edges between the
        children nodes are not present in the returned graph.
        """
        identifier = self._NodeCls.identifier_from(node)
        assert self.node_exists(node), (
            f'The provided node with identifier {identifier} is not present in the graph with nodes: '
            f'{self.get_node_names()}'
        )

        children_graph = self.copy()

        # get a list of outbound edges to the node
        outbound = children_graph.get_node(identifier).get_outbound_edges()

        # remove all the edges other than inbound edges
        for e in children_graph.edges:
            if e not in outbound:
                children_graph.delete_edge(*e.get_edge_pair())

        # Remove all nodes that are not connected to the query node (identify parent nodes)
        children_nodes = {identifier} | {e.destination.identifier for e in outbound}
        for n in children_graph.nodes:
            if n.identifier not in children_nodes:
                children_graph.delete_node(n.identifier)

        return children_graph

    def get_parent_nodes(self, node: NodeLike) -> List[Node]:
        """Get all parent nodes for a specific node."""
        identifier = self._NodeCls.identifier_from(node)
        assert self.node_exists(node), (
            f'The provided node with identifier {identifier} is not present in the graph with nodes: '
            f'{self.get_node_names()}'
        )

        # get a list of inbound edges to the node
        inbound = self.get_node(identifier).get_inbound_edges()

        # Remove all nodes that are not connected to the query node (identify parent nodes)
        return list({e.source for e in inbound})

    def get_parents(self, node: NodeLike) -> List[str]:
        """Get node identifiers for all parent nodes for a specific node."""
        # Get identifier for each parent node
        return [n.identifier for n in self.get_parent_nodes(node)]

    def get_parents_graph(self, node: NodeLike) -> CausalGraph:
        """
        Get a sub causal graph that only includes the parents of a specific node and the node itself.

        This method returns a star graph, such that all edges are going to the provided node. Any edges between the
        parents nodes are not present in the returned graph.
        """
        identifier = self._NodeCls.identifier_from(node)
        assert self.node_exists(node), (
            f'The provided node with identifier {identifier} is not present in the graph with nodes: '
            f'{self.get_node_names()}'
        )

        parents_graph = self.copy()

        # get a list of inbound edges to the node
        inbound = self.get_node(identifier).get_inbound_edges()

        # remove all the edges other than inbound edges
        for e in parents_graph.edges:
            if e not in inbound:
                parents_graph.delete_edge(*e.get_edge_pair())

        # Remove all nodes that are not connected to the query node (identify parent nodes)
        parent_nodes = {identifier} | {e.source.identifier for e in inbound}
        for n in parents_graph.nodes:
            if n.identifier not in parent_nodes:
                parents_graph.delete_node(n.identifier)

        return parents_graph

    def get_ancestors(self, node: NodeLike) -> Set[str]:
        """
        Get all ancestors of a node.

        This method is only applicable if the graph is a valid DAG (i.e., all edges are directed).
        """
        identifier = self._NodeCls.identifier_from(node)
        assert self.node_exists(node), (
            f'The provided node with identifier {identifier} is not present in the graph with nodes: '
            f'{self.get_node_names()}'
        )

        return networkx.ancestors(self.to_networkx(), identifier)

    def get_ancestral_graph(self, node: NodeLike) -> CausalGraph:
        """
        Get a sub causal graph that only includes the ancestors of a specific node and the node itself.
        """
        identifier = self._NodeCls.identifier_from(node)
        ancestors: List[str] = [*self.get_ancestors(node), identifier]

        ancestral_graph = self.copy()

        for i in ancestral_graph.nodes:
            if i.identifier not in ancestors:
                ancestral_graph.delete_node(i.identifier)

        return ancestral_graph

    def get_descendants(self, node: NodeLike) -> Set[str]:
        """
        Get all descendants of a node.

        This method is only applicable if the graph is a valid DAG (i.e., all edges are directed).
        """
        identifier = self._NodeCls.identifier_from(node)
        assert self.node_exists(node), (
            f'The provided node with identifier {identifier} is not present in the graph with nodes: '
            f'{self.get_node_names()}'
        )

        return networkx.descendants(self.to_networkx(), identifier)

    def get_descendant_graph(self, node: NodeLike) -> CausalGraph:
        """
        Get a sub causal graph that only includes the descendants of a specific node and the node itself.
        """
        identifier = self._NodeCls.identifier_from(node)
        descendants: List[str] = [*self.get_descendants(node), identifier]

        descendant_graph = self.copy()

        for i in descendant_graph.nodes:
            if i.identifier not in descendants:
                descendant_graph.delete_node(i.identifier)

        return descendant_graph

    def is_ancestor(
        self, ancestor_node: NodeLike, descendant_node: Union[NodeLike, Set[NodeLike], List[NodeLike]]
    ) -> bool:
        """
        Check whether there is a causal path between ancestor and descendant node(s).

        This method is only applicable if the graph is a valid DAG (i.e., all edges are directed).

        :param ancestor_node: node identifier-coercible object of a potential ancestor node.
        :param descendant_node: a single or a set/list of node identifier-coercible objects.
        :return: whether all nodes passed in the `'descendant_node'` parameter are descendants of the ancestor node.
            Hence, if some nodes are descendants and some are not, `False` is returned.
        """
        descendant_node_set = (
            {descendant_node}
            if not isinstance(descendant_node, (set, list))
            else set(descendant_node)
            if isinstance(descendant_node, list)
            else descendant_node
        )

        return descendant_node_set.issubset(self.get_descendants(ancestor_node))

    def is_descendant(
        self, descendant_node: NodeLike, ancestor_node: Union[NodeLike, Set[NodeLike], List[NodeLike]]
    ) -> bool:
        """
        Check whether there is a causal path between descendant and ancestor node(s).

        This method is only applicable if the graph is a valid DAG (i.e., all edges are directed).

        :param descendant_node: node identifier-coercible object of a potential descendant node.
        :param ancestor_node: a single or a set/list of node identifier-coercible objects.
        :return: whether all nodes passed in the `'ancestor_node'` parameter are ancestors of the descendant node.
            Hence, if some nodes are ancestors and some are not, `False` is returned.
        """

        ancestor_node_set = (
            {ancestor_node}
            if not isinstance(ancestor_node, (set, list))
            else set(ancestor_node)
            if isinstance(ancestor_node, list)
            else ancestor_node
        )

        return ancestor_node_set.issubset(self.get_ancestors(descendant_node))

    def get_common_ancestors(self, node_1: NodeLike, node_2: NodeLike) -> Set[str]:
        """
        Get all common ancestors for two nodes.

        This method is only applicable if the graph is a valid DAG (i.e., all edges are directed).

        If one of the provided nodes is an ancestor of another, it will not appear in the returned set.

        :param node_1: node identifier-coercible object specifying a node in the graph.
        :param node_2: node identifier-coercible object specifying a node in the graph.
        :return: set of all node identifiers which are ancestors of both nodes.
        """
        assert self.node_exists(node_1) and self.node_exists(node_2), 'Provided nodes are not present in this graph.'

        return self.get_ancestors(node_1).intersection(self.get_ancestors(node_2))

    def get_common_descendants(self, node_1: NodeLike, node_2: NodeLike) -> Set[str]:
        """
        Get all common descendants for two nodes.

        This method is only applicable if the graph is a valid DAG (i.e., all edges are directed).

        If one of the provided nodes is a descendant of another, it will not appear in the returned set.

        :param node_1: node identifier-coercible object specifying a node in the graph.
        :param node_2: node identifier-coercible object specifying a node in the graph.
        :return: set of all node identifiers which are descendants of both nodes.
        """
        assert self.node_exists(node_1) and self.node_exists(node_2), 'Provided nodes are not present in this graph.'

        return self.get_descendants(node_1).intersection(self.get_descendants(node_2))

    def get_d_separation_set(self, node_1: NodeLike, node_2: NodeLike) -> Set[str]:
        """
        Return a minimal d-separation set between two nodes.

        This method is only applicable if the graph is a valid DAG (i.e., all edges are directed).

        :param node_1: a single node-identifier coercible object.
        :param node_2: a single node-identifier coercible object.
        :return: a set of node identifiers for a minimal d-separation set between the provided nodes.
        """
        assert self.is_dag(), 'This method only works for DAGs but the current graph is not a DAG.'

        node_1 = self._NodeCls.identifier_from(node_1)
        node_2 = self._NodeCls.identifier_from(node_2)

        assert all(
            node in self.get_node_names() for node in [node_1, node_2]
        ), f'All nodes must be present in the graph. Got nodes "{node_1, node_2}".'

        assert not self.edge_exists(
            node_1, node_2
        ), 'Cannot identify a d-separation set between two nodes if an edge exists between them.'

        networkx_digraph = self.to_networkx()
        assert isinstance(networkx_digraph, networkx.DiGraph)  # Will be the case if graph is a DAG. Needed for linting.

        return networkx.minimal_d_separator(networkx_digraph, node_1, node_2)

    def is_d_separated(
        self,
        nodes_1: Union[Set[NodeLike], List[NodeLike], NodeLike],
        nodes_2: Union[Set[NodeLike], List[NodeLike], NodeLike],
        separation_set: Optional[Union[Set[NodeLike], List[NodeLike]]] = None,
    ) -> bool:
        """
        Check whether given sets of nodes are d-separated given the separation set.

        This method is only applicable if the graph is a valid DAG (i.e., all edges are directed).

        :param nodes_1: a set/list of or a single node-identifier coercible object(s).
        :param nodes_2: a set/list of or a single node-identifier coercible object(s).
        :param separation_set: a set/list of node-identifier coercible objects.
        :return: whether the separation set is d-separating for the provided nodes.
        """
        assert self.is_dag(), 'This method only works for DAGs but the current graph is not a DAG.'

        if separation_set is None:
            separation_set = set()

        nodes_1 = (
            {nodes_1}
            if not isinstance(nodes_1, (set, list))
            else set(nodes_1)
            if isinstance(nodes_1, list)
            else nodes_1
        )
        nodes_2 = (
            {nodes_2}
            if not isinstance(nodes_2, (set, list))
            else set(nodes_2)
            if isinstance(nodes_2, list)
            else nodes_2
        )

        nodes_1 = set(self._NodeCls.identifier_from(node) for node in nodes_1)
        nodes_2 = set(self._NodeCls.identifier_from(node) for node in nodes_2)
        separation_set = set(self._NodeCls.identifier_from(node) for node in separation_set)

        assert all(
            node in self.get_node_names() for node in [*nodes_1, *nodes_2, *separation_set]
        ), 'All nodes must be present in the graph.'

        return networkx.d_separated(self.to_networkx(), nodes_1, nodes_2, separation_set)

    def is_minimally_d_separated(
        self, node_1: NodeLike, node_2: NodeLike, separation_set: Optional[Set[NodeLike]] = None
    ) -> bool:
        """
        Check whether given nodes are minimally d-separated given the separation set.

        This method is only applicable if the graph is a valid DAG (i.e., all edges are directed).

        :param node_1: a single node-identifier coercible object.
        :param node_2: a single node-identifier coercible object.
        :param separation_set: a set/list of node-identifier coercible objects.
        :return: whether the separation set is d-separating for the provided nodes.
        """
        assert self.is_dag(), 'This method only works for DAGs but the current graph is not a DAG.'

        if separation_set is None:
            separation_set = set()

        node_1_id = self._NodeCls.identifier_from(node_1)
        node_2_id = self._NodeCls.identifier_from(node_2)
        separation_set_id = set(self._NodeCls.identifier_from(node) for node in separation_set)

        assert all(
            node in self.get_node_names() for node in [node_1_id, node_2_id, *separation_set_id]
        ), 'All nodes must be present in the graph.'

        networkx_digraph = self.to_networkx()
        assert isinstance(networkx_digraph, networkx.DiGraph)  # Will be the case if graph is a DAG. Needed for linting.

        # is_minimal_d_separator does not check if set is indeed separating so need to confirm
        return networkx.is_minimal_d_separator(
            networkx_digraph, node_1_id, node_2_id, separation_set_id
        ) and self.is_d_separated(
            node_1_id, node_2_id, separation_set_id  # type: ignore
        )

    def get_topological_order(self, return_all: bool = False) -> Union[List[str], List[List[str]]]:
        """
        Return either a single or all topological orders of the graph.

        A topological order is a non-unique permutation of the nodes such that an edge from `'A'` to `'B'` implies
        that `'A'` appears before `'B'` in the topological sort order. Generating all possible topological orders may
        be expensive for large graphs.

        It is only possible to get topological order if the graph is a valid DAG.

        :param return_all: whether to generate all topological orders.
        :return: either a list of strings identifying a single topological order, or a list of lists identifying all
            possible topological orders.
        """
        assert self.is_dag(), 'This method only works for DAGs but the current graph is not a DAG.'
        if return_all:
            return list(networkx.all_topological_sorts(self.to_networkx()))
        else:
            return list(networkx.topological_sort(self.to_networkx()))

    def get_all_causal_paths(self, source: NodeLike, destination: NodeLike) -> List[List[str]]:
        """
        Get all causal paths between the provided source and destination.

        This method is only applicable if the graph is a valid DAG (i.e., all edges are directed).

        :param source: source node-identifier coercible object.
        :param destination: destination node-identifier coercible object.
        :return: a list of lists specifying the node identifiers on the path between source and destination (this is
            inclusive of the source and destination node identifiers). An empty list is returned if no paths are
            available.
        """

        assert self.is_dag(), 'This method only works for DAGs but the current graph is not a DAG.'

        source = self._NodeCls.identifier_from(source)
        destination = self._NodeCls.identifier_from(destination)

        assert all(node in self.get_node_names() for node in [source, destination])

        # In networkx 3.3 this will return a path of length 1 with just that node in it. We don't want to consider this
        # particularly as there is no self loop from that node to itself. One could make an edge from source to source
        # but then it would no longer be a DAG as we confirm in the first step of this method.
        if source == destination:
            return []

        return list(networkx.all_simple_paths(self.to_networkx(), source, destination))

    def directed_path_exists(self, source: NodeLike, destination: NodeLike) -> bool:
        """
        Check whether there exists a directed path between the source and destination.

        Unlike the `CausalGraph.get_all_causal_paths` method, this works for mixed causal graphs as well (note that
        non-directed edges are ignored).

        :param source: source node-identifier coercible object.
        :param destination: destination node-identifier coercible object.
        :return: True if there exists a directed path between the source and destination, False otherwise.
        """
        # get proper node identifiers and verify that they exist in the graph
        source = self._NodeCls.identifier_from(source)
        destination = self._NodeCls.identifier_from(destination)

        assert all(node in self.get_node_names() for node in [source, destination])

        # get the children of an outbound node; need to ignore any edges other than directed
        children = [edge.destination.identifier for edge in self.get_node(source).get_outbound_edges()]

        # return True if the set of children contains the destination
        if destination in children:
            return True

        # perform a recursive depth-first search to see if a directed path exists
        for child in children:
            if self.directed_path_exists(child, destination):
                return True

        return False

    @staticmethod
    def coerce_to_nodelike(node: Union[NodeLike, int]) -> NodeLike:
        """
        Coerce an object to be used for node construction.

        If the provided node is already `cai_causal_graph.type_definitions.NodeLike`, it is simply returned.
        If provided node is an integer, it is converted to a string and returned.
        If anything else, it will raise a `TypeError`.
        """
        if isinstance(node, (str, HasIdentifier)):
            return node
        elif isinstance(node, int):
            return str(node)
        else:
            raise TypeError(
                f'Cannot coerce {type(node)} to node like. See coerce_to_nodelike method docstring for details.'
            )

    def _assert_node_does_not_depend_on_itself(self, identifier: NodeLike):
        """Check that the given node does not depend on itself, and raise an error if it does."""
        identifier = self._NodeCls.identifier_from(identifier)

        checked: Set[str] = set()
        to_check = [identifier]

        while len(to_check) > 0:
            current = to_check.pop()
            if current == identifier and len(checked) > 0:
                # Do not fail if the identifier is being checked on the first iteration
                raise AssertionError(f'Node {identifier} depends upon itself')
            if current not in checked:
                checked.add(current)
                for edge in self._nodes_by_identifier[current].get_inbound_edges():
                    to_check.append(edge.source.identifier)

    def _clean_empty_edge_dictionaries(self):
        """Remove any dictionaries that have no entries from the edge map."""
        by_source_keys = list(self._edges_by_source.keys())
        for key in by_source_keys:
            if len(self._edges_by_source[key]) == 0:
                self._edges_by_source.pop(key)

        by_destination_keys = list(self._edges_by_destination.keys())
        for key in by_destination_keys:
            if len(self._edges_by_destination[key]) == 0:
                self._edges_by_destination.pop(key)

    def to_dict(self, include_meta: bool = True) -> dict:
        """
        Serialize a `cai_causal_graph.causal_graph.CausalGraph` instance to a dictionary.

        :param include_meta: Whether to include meta information about the graph in the dictionary. Default is `True`.
        :return: The dictionary representation of the `cai_causal_graph.causal_graph.CausalGraph` instance.
        """
        nodes = {node.identifier: node.to_dict(include_meta=include_meta) for node in self.nodes}

        edges: Dict[str, Dict[str, dict]] = dict()
        for edge in self.edges:
            source, destination = edge.get_edge_pair()
            if source not in edges:
                edges[source] = dict()
            edges[source][destination] = edge.to_dict(include_meta=include_meta)

        d = {'nodes': nodes, 'edges': edges, 'version': CAUSAL_GRAPH_VERSION}

        if include_meta:
            d['meta'] = self.meta.copy()

        return d

    def to_networkx(self) -> networkx.Graph:
        """
        Return a `networkx.Graph` or `networkx.DiGraph` corresponding to the `cai_causal_graph.causal_graph.CausalGraph`
        instance.

        :return: `networkx.Graph` representation of the `CausalGraph`. `networkx.DiGraph` is a subclass of
            `networkx.Graph` and will be returned when all edges in the `CausalGraph` instance are directed. When all
            edges in the `CausalGraph` instance are undirected, a `networkx.Graph` will be returned. When there is a mix
            of directed and undirected edges or there are more elaborate edge types, a
            `cai_causal_graph.exceptions.CausalGraphErrors.GraphConversionError` will be raised.
        """
        if self._networkx is not None:
            return deepcopy(self._networkx)
        # Save these off as they need to be called more than once.
        is_fully_directed = self._is_fully_directed()
        is_fully_undirected = self._is_fully_undirected()

        error_msg = (
            'CausalGraph cannot be converted to a networkx.Graph unless it is undirected (all edges must be '
            'undirected) and it cannot be converted to a networkx.DiGraph unless it is directed (all edges must be '
            'directed).\n'
        )

        # Check if graph is all undirected or all directed. Cannot convert otherwise.
        if not is_fully_directed and not is_fully_undirected:
            raise CausalGraphErrors.GraphConversionError(
                error_msg + self._is_directed_and_or_undirected_error_message(is_one_kind_only=True)
            )
        elif is_fully_directed:
            # Set type to convert networkx to.
            networkx_type = networkx.DiGraph
        elif is_fully_undirected:
            # Set type to convert networkx to.
            networkx_type = networkx.Graph
        else:
            # Should not hit this block so raise error if it does.
            raise CausalGraphErrors.GraphConversionError(error_msg)

        # Convert to the appropriate networkx object. DiGraph extends Graph so Graph return type is valid for both.
        networkx_graph = networkx.empty_graph(n=self.get_node_names(), create_using=networkx_type)
        for source, destinations in self._edges_by_source.items():
            for target in destinations.keys():
                networkx_graph.add_edge(source, target)

        self._networkx = networkx_graph

        return networkx_graph

    def to_numpy(self) -> Tuple[numpy.ndarray, List[str]]:
        """
        Return a numpy array that represents the adjacency matrix of the `cai_causal_graph.causal_graph.CausalGraph`
        instance.

        To avoid confusion, this method also returns a list of variables that corresponds to the order of columns in
        the numpy array.
        """
        for edge in self.edges:
            if edge.get_edge_type() not in [EdgeType.DIRECTED_EDGE, EdgeType.UNDIRECTED_EDGE]:
                raise TypeError(
                    'Cannot convert a CausalGraph instance to a numpy array if it contains edges other than directed '
                    f'and undirected edges. Got {edge.get_edge_type()} for the edge {edge.descriptor}.'
                )

        return self.adjacency_matrix, self.get_node_names()

    def to_gml_string(self) -> str:
        """
        Return a Graph Modelling Language (GML) string representative of the
        `cai_causal_graph.causal_graph.CausalGraph` instance.
        """
        error_msg = self._is_directed_and_or_undirected_error_message(is_one_kind_only=False)
        if len(error_msg) > 0:
            raise CausalGraphErrors.GraphConversionError(
                'CausalGraph cannot be converted to a GML string unless it only contains directed and undirected '
                'edges.\n' + error_msg
            )
        return '\n'.join(networkx.generate_gml(self.to_networkx()))

    @classmethod
    def from_dict(cls, d: dict, validate: bool = True) -> CausalGraph:
        """
        Construct a `cai_causal_graph.causal_graph.CausalGraph` instance from a Python dictionary.

        The contents of the passed dictionary are deepcopied when constructing a new `CausalGraph` instance. This
        includes metadata of each node and edge, as well as the graph itself.

        :param d: Dictionary to build a graph from.
        :param validate: Whether to perform validation checks. The validation checks will raise if
            any cycles are introduced to the graph by adding the edge. This should only be disabled to speed up
            this method in situations where it is known that the serialized graph is valid. Default is `True`.
        """
        # deepcopy the metadata if provided in the dict. This is consistent with metadata in nodes/edges which
        # is deepcopied by the `add_node`/`add_edge` methods.
        graph = cls(meta=deepcopy(d.get('meta', None)))

        for identifier, node_dict in d['nodes'].items():
            node = cls._NodeCls.from_dict(node_dict)
            graph.add_node(node=node)

        for source, destinations in d['edges'].items():
            for destination, edge_dict in destinations.items():
                edge = cls._EdgeCls.from_dict(edge_dict)
                graph.add_edge(edge=edge, validate=validate)

        return graph

    @classmethod
    def from_networkx(cls, g: networkx.Graph) -> CausalGraph:
        """Construct a `cai_causal_graph.causal_graph.CausalGraph` instance from a `networkx.Graph` instance."""
        # Check graph type.
        if isinstance(g, networkx.MultiGraph) or isinstance(g, networkx.MultiDiGraph):
            raise CausalGraphErrors.InvalidNetworkXError(
                f'{cls.__class__} cannot be constructed from networkx.MultiGraph or networkx.MultiDiGraph. However, '
                f'the provided graph is of type: {type(g)}.'
            )
        # Convert node names to strings.
        node_names: List[str] = [cls._NodeCls.identifier_from(cls.coerce_to_nodelike(node)) for node in g.nodes()]
        return cls.from_adjacency_matrix(networkx.to_numpy_array(g), node_names)  # type: ignore

    @classmethod
    def from_skeleton(cls, skeleton: Skeleton) -> CausalGraph:
        """
        Construct a `cai_causal_graph.causal_graph.CausalGraph` instance from a
        `cai_causal_graph.causal_graph.Skeleton` instance.
        """
        assert isinstance(skeleton, Skeleton), f'Expected skeleton to be of type Skeleton, but got {type(skeleton)}.'
        return cls.from_networkx(skeleton.to_networkx())

    @classmethod
    def from_gml_string(cls, gml: str) -> CausalGraph:
        """
        Return an instance of `cai_causal_graph.causal_graph.CausalGraph` constructed from the provided Graph Modelling
        Language (GML) string.
        """
        g = networkx.parse_gml(gml)
        return cls.from_networkx(g)

    @classmethod
    def from_adjacency_matrix(
        cls, adjacency: numpy.ndarray, node_names: Optional[List[Union[NodeLike, int]]] = None
    ) -> CausalGraph:
        """
        Construct a `cai_causal_graph.causal_graph.CausalGraph` instance from an adjacency matrix and optionally a list
        of node names.

        If a list of node names is provided it is used to identify the columns and rows in the adjacency matrix.
        If no node names are provided, these are autogenerated to be ['node_0','node_1'...].

        :param adjacency: A square binary numpy adjacency array.
        :param node_names: A list of strings, `cai_causal_graph.interfaces.HasIdentifier`, and/or integers which can be
            coerced to `cai_causal_graph.graph_components.Node`.
        :return: A `cai_causal_graph.causal_graph.CausalGraph` object.
        """
        # check that adjacency matrix is a square matrix
        if not len(adjacency.shape) == 2:
            raise CausalGraphErrors.InvalidAdjacencyMatrixError(
                f'Expected a two dimensional adjacency matrix, got a matrix with shape {adjacency.shape}.'
            )
        if not adjacency.shape[0] == adjacency.shape[1]:
            raise CausalGraphErrors.InvalidAdjacencyMatrixError(
                f'Expected a square adjacency matrix, got a matrix with shape {adjacency.shape}.'
            )

        # check adjacency matrix is binary
        if not numpy.array_equal(adjacency, adjacency.astype(bool)):
            raise CausalGraphErrors.InvalidAdjacencyMatrixError(f'Expected a binary adjacency matrix, got {adjacency}.')

        # ensure that if node_names are provided, they match up with matrix dimensions
        if node_names is not None:
            assert len(node_names) == adjacency.shape[0], CausalGraphErrors.InvalidAdjacencyMatrixError(
                f'The provided node names ({node_names} do not match up with the provided adjacency matrix dimensions.'
            )
        else:
            # if no node names are provided, autogenerate them
            node_names = [f'node_{i}' for i in range(adjacency.shape[0])]

        # coerce node names into NodeLike to obtain the identifiers of the created nodes
        nodes = [cls.coerce_to_nodelike(node) for node in node_names]  # type: ignore

        # Add edges. Any conversion from BasicFeature or BasicTarget is handled by the add_edge method.
        graph = cls()
        graph.add_nodes_from(nodes)
        for i, j in itertools.combinations(range(len(nodes)), 2):
            if adjacency[i, j] != 0 and adjacency[j, i] == 0:
                graph.add_edge(nodes[i], nodes[j], edge_type=EdgeType.DIRECTED_EDGE)
            elif adjacency[i, j] == 0 and adjacency[j, i] != 0:
                graph.add_edge(nodes[j], nodes[i], edge_type=EdgeType.DIRECTED_EDGE)
            elif adjacency[i, j] != 0 and adjacency[j, i] != 0:
                graph.add_edge(nodes[i], nodes[j], edge_type=EdgeType.UNDIRECTED_EDGE)

        return graph

    def copy(self, include_meta: bool = True) -> CausalGraph:
        """
        Return a copy of the `cai_causal_graph.causal_graph.CausalGraph` instance.

        :param include_meta: if `True` (default), the metadata will be copied as well.
        :return: A copy of the `cai_causal_graph.causal_graph.CausalGraph` instance.
        """
        graph_dict = self.to_dict(include_meta=include_meta)
        # validate=False as we know it will be an exact copy
        new_graph = self.__class__.from_dict(graph_dict, validate=False)
        assert isinstance(new_graph, self.__class__)  # for linting and sanity check
        return new_graph

    def _reset_cached_attributes(self):
        """Reset cached internal attributes."""
        self._is_dag = None
        self._networkx = None
        self._is_fully_directed_cached = None
        self._is_fully_undirected_cached = None
        self._adjacency = None

    def __repr__(self) -> str:
        """Return a string description of the `cai_causal_graph.causal_graph.CausalGraph` instance."""
        return (
            f'{self.__class__.__name__}(num_nodes={len(self.nodes)}, num_edges={len(self.edges)}, '
            f'id={self.__hash__()}, is_dag={self.is_dag()})'
            f'\n'
            f'Nodes: {self.nodes}\nEdges: {self.edges}'
        )

    def details(self) -> str:
        """Return a detailed string description of the `cai_causal_graph.causal_graph.CausalGraph` instance."""
        node_details = '\t' + '\n\t'.join([n.details().replace('\n', '\n\t') for n in self.nodes])
        edge_details = '\t' + '\n\t'.join([e.details().replace('\n', '\n\t') for e in self.edges])
        return f'{self.__repr__()}\n' f'Node Details:\n{node_details}\nEdge Details:\n{edge_details}'

    def __hash__(self) -> int:
        """Return a hash representation of the `cai_causal_graph.causal_graph.CausalGraph` instance."""
        return hash(repr(self.to_dict()))
