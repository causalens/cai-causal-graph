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

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy

from cai_causal_graph import CausalGraph
from cai_causal_graph.graph_components import Edge, Node, get_name_from_lag, get_variable_name_and_lag
from cai_causal_graph.type_definitions import EDGE_T, NodeLike, NodeVariableType

logger = logging.getLogger(__name__)

# TODO: we can do one general for many other things as well
def reset_attributes(func):
    """Decorator to reset the summary graph."""

    # TODO: make this more clever as it is not said that we need to reset the summary graph etc
    def wrapper(self, *args, **kwargs):
        function = func(self, *args, **kwargs)
        self._summary_graph = None
        self._order = None
        self._variables = None
        return function

    return wrapper


def extract_names_and_lags(
    node_names: List[NodeLike],
) -> Tuple[List[Dict[str, int]], int]:
    """
    Extract the names and lags from a list of node names.
    This is useful for converting a list of node names into a list of node names and lags.

    Example:
        ['X', 'Y', 'Z lag(n=2)'] -> [{'X': 0}, {'Y': 0}, {'Z': 2}]

    :param node_names: List of node names.
    :return: List of dictionaries with node names and lags sorted by lag and the maximum lag.
    """
    names_and_lags = []
    for node_name in node_names:
        variable_name, lag = get_variable_name_and_lag(node_name)
        names_and_lags.append({variable_name: lag})
    sorted_names_and_lags = sorted(names_and_lags, key=lambda x: max(x.values()), reverse=False)
    maxlag = next(iter(sorted_names_and_lags[0].values()))
    return sorted_names_and_lags, maxlag


class TimeSeriesCausalGraph(CausalGraph):
    """
    A causal graph for time series data.

    The node in a time series causal graph will have additional metadata that
    gives the time information of the node together with the variable name.
    The two additional metadata are:
    - 'timedelta': the time difference with respect the reference time 0
    - 'variable_name': the name of the variable (without the lag information).
    """

    def __init__(
        self,
        input_list: Optional[List[NodeLike]] = None,
        output_list: Optional[List[NodeLike]] = None,
        fully_connected: bool = True,
        store_graphs: bool = False,
    ):
        """Initialize the time series causal graph."""
        super().__init__(input_list, output_list, fully_connected)

        self._store_graphs = store_graphs
        # autoregressive order of the graph (max lag)
        self._order: Optional[int] = None
        # list of variables in the graph, i.e. discarding the lags (X1(t-1) and X1 are the same variable)
        self._variables: Optional[List[str]] = None
        self._summary_graph: Optional[CausalGraph] = None
        self._minimum_graph: Optional[TimeSeriesCausalGraph] = None
        self._meta_time_name = 'time_lag'
        self._meta_variable_name = 'variable_name'

    def __eq__(self, other: object) -> bool:
        """Return True if the graphs are equal."""
        if not isinstance(other, TimeSeriesCausalGraph):
            return False

        # now check if the graphs are equal. Since TimeSeriesCausalGraph is a subclass of CausalGraph,
        # we can use the CausalGraph.__eq__ method and then check for the metadata
        are_equal = super().__eq__(other)
        # no need to check the metadata if the graphs are not equal
        if not are_equal:
            return False

        # now check the metadata timedelta in the nodes
        for node in self.get_nodes():
            other_node = other.get_node(node.identifier)
            # check the variable name
            node_metadata_variable = node.get_metadata()[self._meta_variable_name]
            other_node_metadata_td = other_node.get_metadata()[self._meta_variable_name]

            if node_metadata_variable != other_node_metadata_td:
                return False

            # check the time delta
            node_metadata_td = node.get_metadata()[self._meta_time_name]
            other_node_metadata_td = other_node.get_metadata()[self._meta_time_name]

            if node_metadata_td != other_node_metadata_td:
                return False

        return True

    def copy(self, include_meta: bool = True) -> TimeSeriesCausalGraph:
        """Return a copy of the graph."""
        graph = super().copy(include_meta=include_meta)
        # cast the graph to TimeSeriesCausalGraph to have the correct metadata
        return TimeSeriesCausalGraph.from_causal_graph(graph)

    def get_minimum_graph(self) -> TimeSeriesCausalGraph:
        """
        Return a minimum graph.

        The minimum graph is the graph with the minimum number of edges that is equivalent to the original graph.
        In other words, it is a graph that has no edges whose destination is not time delta 0.
        """
        minimum_cg = TimeSeriesCausalGraph()

        # first step
        # go through all the nodes and add them to the minimum graph if they have time delta 0
        for node in self.get_nodes():
            if node.get_metadata()[self._meta_time_name] == 0:
                # add if node is not already in the graph
                if not minimum_cg.node_exists(node.identifier):
                    minimum_cg.add_node(node=node)

        # second step: add nodes that are connected to the nodes in the minimum graph
        # go through all the nodes in the minimum graph and add the nodes that are entering them
        for node in minimum_cg.get_nodes():
            for edge in self.get_edges():
                # could also check just edge.destination == node (avoiding the metadata check)
                if edge.destination.identifier == node.identifier:
                    # add the node if it is not already in the graph
                    if not minimum_cg.node_exists(edge.source):
                        minimum_cg.add_node(node=self.get_node(edge.source))
                    minimum_cg.add_edge(edge=edge)

        return minimum_cg

    def is_minimum_graph(self) -> bool:
        """Return True if the graph is minimum."""
        return self == self.get_minimum_graph()

    def get_summary_graph(self) -> CausalGraph:
        """
        Return a summary graph.

        Collapse graph in time into a single node per variable (column name).
        This can become cyclic and bi-directed as X-1 -> Y and Y-1 -> X would become X <-> Y.

        There are several cases to consider. Assume the edge in consideration is called B.
        - if the edge is not in the summary graph, add it
        - if there's an edge A is in the summary graph, then:
            - if A and B have the same direction, keep the direction
            - if A and B have different directions, make it bi-directed
            - if one of the two is already bi-directed, keep it bi-directed

        :return: The summary graph as a CausalGraph object.
        """
        if self._summary_graph is None:
            summary_graph = CausalGraph()
            # now check as described above (assume egdes are already directed)
            edges = self.get_edges()
            for edge in edges:
                # first we need to extract the variable names from the nodes as the summary graph
                # will have the variable names as nodes
                source_node = edge.source
                destination_node = edge.destination
                source_variable_name = source_node.get_metadata()[self._meta_variable_name]
                destination_variable_name = destination_node.get_metadata()[self._meta_variable_name]

                assert source_variable_name is not None, 'Source variable name is None, cannot create summary graph.'
                assert (
                    destination_variable_name is not None
                ), 'Destination variable name is None, cannot create summary graph.'

                if source_variable_name != destination_variable_name and not summary_graph.is_edge_by_pair(
                    (source_variable_name, destination_variable_name)
                ):
                    summary_graph.add_edge_by_pair((source_variable_name, destination_variable_name))

            self._summary_graph = summary_graph

        return self._summary_graph

    def extend_graph(
        self, backward_steps: Optional[int] = None, forward_steps: Optional[int] = None
    ) -> TimeSeriesCausalGraph:
        """
        Return an extended graph.
        Extend the graph in time by adding nodes for each variable at each time step from backward_steps to forward_steps.
        If a backward step of n is specified, it means that the graph will be extended in order to include nodes at time -n.
        For example, if t is a reference time, if backward_steps is 2, the graph will be extended to include nodes at time t-2.
        This does not mean that the graph will be extended to only include nodes at time t-2, but rather that it will include
        nodes at time t-2 and all the nodes that are connected to them as specified by the minimum graph.

        If both backward_steps and forward_steps are None, return the original graph.

        :param backward_steps: Number of steps to extend the graph backwards in time. If None, do not extend backwards.
        :param forward_steps: Number of steps to extend the graph forwards in time. If None, do not extend forwards.
        :return: Extended graph with nodes for each variable at each time step from backward_steps to forward_steps.
        """
        # check steps are valid (positive integers) if not None
        assert backward_steps is None or backward_steps > 0
        assert forward_steps is None or forward_steps > 0

        # first get the minimum graph
        minimum_graph = self.get_minimum_graph()

        # create a new graph by copying the minimum graph
        extended_graph = minimum_graph.copy()

        if backward_steps is not None:
            # start from 1 as 0 is already defined
            # we cannot start directly from maxlag as it may be possible
            # that not all the nodes from 1 to -maxlag are defined (as they were not
            # needed in the mimimal graph)
            maxlag = minimum_graph.order
            assert maxlag is not None

            # create all the nodes from 1 to maxlag
            for lag in range(1, backward_steps + 1):
                # add nodes
                neglag = -lag
                for node in minimum_graph.get_nodes():
                    # create the node with -lag
                    # create the new identifier from the lag
                    lagged_identifier = get_name_from_lag(node.identifier, neglag)
                    lagged_node = Node(
                        identifier=lagged_identifier,
                        meta={
                            'variable_name': node.identifier,
                            'time_lag': neglag,
                        },
                    )
                    # if node does not exist, add it
                    if not extended_graph.node_exists(lagged_node.identifier):
                        extended_graph.add_node(node=lagged_node)

            for lag in range(1, backward_steps + 1):
                neglag = -lag
                for node in minimum_graph.get_nodes():
                    # add in-bound edges (here it could beyond the backward_steps as dictated by the minimum graph)
                    for in_edge in minimum_graph.get_edges(destination=node.identifier):
                        if in_edge.source.time_lag + neglag >= -backward_steps:
                            # create the edge with -lag
                            # create the new identifier from the lag
                            lagged_source_identifier = get_name_from_lag(
                                in_edge.source.identifier, in_edge.source.time_lag + neglag
                            )
                            lagged_identifier = get_name_from_lag(node.identifier, neglag)

                            # check if lagged_source_identifier is in the graph and if the edge is not already there
                            if extended_graph.node_exists(lagged_source_identifier) and not extended_graph.edge_exists(
                                lagged_source_identifier, lagged_identifier
                            ):
                                lagged_node = Node(
                                    identifier=lagged_identifier,
                                    meta={
                                        'variable_name': node.identifier,
                                        'time_lag': neglag,
                                    },
                                )
                                lagged_source_node = extended_graph.get_node(lagged_source_identifier)
                                lagged_edge = Edge(
                                    source=lagged_source_node,
                                    destination=lagged_node,
                                )

                                extended_graph.add_edge(edge=lagged_edge)

            # add a waring if the backward_steps is smaller than the maximum lag in the graph
            if backward_steps < maxlag:
                ramaining_nodes = []
                for node in minimum_graph.get_nodes():
                    if abs(node.time_lag) > backward_steps:
                        ramaining_nodes.append(node.identifier)
                logger.warning(
                    'backward_steps is smaller than the maximum lag in the graph, the following nodes will not be added: %s',
                    list(set(ramaining_nodes)),
                )

        if forward_steps is not None:
            # start from 1 as 0 is already defined
            # with the forward extension, the maximum positive lag is forward_steps
            assert forward_steps > 0, 'forward_steps must be positive.'
            for lag in range(1, forward_steps + 1):
                for node in minimum_graph.get_nodes():
                    # create the node with +lag (if it does not exist)
                    # create the new identifier from the lag
                    lagged_identifier = get_name_from_lag(node.identifier, lag)
                    lagged_node = Node(
                        identifier=lagged_identifier,
                        meta={
                            'variable_name': node.identifier,
                            'time_lag': lag,
                        },
                    )
                    # if node does not exist, add it
                    if not extended_graph.node_exists(lagged_node.identifier):
                        extended_graph.add_node(node=lagged_node)

                    # add all the in-bound edges corresponding the the previous lag

                    # get the identifier of the node with lag -1
                    lagged_previous_identifier = get_name_from_lag(lagged_identifier, lag - 1)
                    # get the node with lag -1

                    for in_edge in extended_graph.get_edges(destination=lagged_previous_identifier):
                        # create the edge with +lag
                        # create the new identifier from the lag
                        lagged_source_identifier = get_name_from_lag(
                            in_edge.source.identifier, in_edge.source.time_lag + lag
                        )
                        if not extended_graph.edge_exists(lagged_source_identifier, lagged_identifier):
                            # check if the source node exists
                            if not extended_graph.node_exists(lagged_source_identifier):
                                # if it does not exist, create it
                                lagged_source_node = Node(
                                    identifier=lagged_source_identifier,
                                    meta={
                                        'variable_name': in_edge.source.identifier,
                                        'time_lag': in_edge.source.time_lag + lag,
                                    },
                                )
                                extended_graph.add_node(node=lagged_source_node)
                            else:
                                lagged_source_node = extended_graph.get_node(lagged_source_identifier)
                            lagged_edge = Edge(
                                source=lagged_source_node,
                                destination=lagged_node,
                            )

                            extended_graph.add_edge(edge=lagged_edge)

        return extended_graph

    @staticmethod
    def get_variable_names_from_node_names(node_names: List[str]):
        """
        Return a list of variable names from a list of node names.
        This is useful for converting a list of node names into a list of variable names.
        Variables are the elementary (unique) units of a time series causal graph.

        Example:
            ['X', 'X lag(n=1)', 'Y', 'Z lag(n=2)'] -> ['X', 'Y', 'Z']

        :param node_names: Can be a list of node names or a list of dictionaries with node names and lags.
        :return: List of variable names.
        """
        assert isinstance(node_names, list)

        for node_name in node_names:
            vname, _ = get_variable_name_and_lag(node_name)
            if vname not in node_names:
                node_names.append(vname)

        return sorted(node_names)

    @reset_attributes
    def add_node(
        self,
        /,
        identifier: Optional[NodeLike] = None,
        *,
        variable_type: NodeVariableType = NodeVariableType.UNSPECIFIED,
        meta: Optional[dict] = None,
        node: Optional[Node] = None,
        **kwargs,
    ) -> Node:
        node = super().add_node(identifier, variable_type=variable_type, meta=meta, node=node, **kwargs)
        # populate the metadata for each node
        vname, lag = get_variable_name_and_lag(node.identifier)
        node.meta[self._meta_variable_name] = vname
        node.meta[self._meta_time_name] = lag
        return node

    @reset_attributes
    def replace_node(
        self,
        /,
        node_id: NodeLike,
        new_node_id: Optional[NodeLike] = None,
        *,
        variable_type: NodeVariableType = NodeVariableType.UNSPECIFIED,
        meta: Optional[dict] = None,
    ):
        super().replace_node(node_id, new_node_id, variable_type=variable_type, meta=meta)

    @reset_attributes
    def delete_node(self, identifier: NodeLike):
        super().delete_node(identifier)

    @reset_attributes
    def delete_edge(self, source: NodeLike, destination: NodeLike):
        super().delete_edge(source, destination)

    @reset_attributes
    def add_edge(
        self,
        /,
        source: Optional[NodeLike] = None,
        destination: Optional[NodeLike] = None,
        *,
        edge_type: EDGE_T = EDGE_T.DIRECTED_EDGE,
        meta: Optional[dict] = None,
        edge: Optional[Edge] = None,
        **kwargs,
    ) -> Edge:
        edge = super().add_edge(source, destination, edge_type=edge_type, meta=meta, edge=edge, **kwargs)
        # populate the metadata for each node
        for node in self.get_nodes():
            # extract variable name and lag from node name
            vname, lag = get_variable_name_and_lag(node.identifier)
            node.meta[self._meta_variable_name] = vname
            node.meta[self._meta_time_name] = lag

        return edge

    @classmethod
    def from_causal_graph(cls, causal_graph: CausalGraph, store_graphs: bool = False) -> TimeSeriesCausalGraph:
        """
        Return a time series causal graph from a causal graph.
        This is useful for converting a causal graph from a single time step into a time series causal graph.
        """
        # create an instance without calling __init__
        obj = cls.__new__(cls)

        # copy all attributes from causal_graph to the new object
        for name, value in vars(causal_graph).items():
            setattr(obj, name, value)

        # set all the attributes related to time series

        # create a temporary TimeSeriesCausalGraph object to get default values for new attributes
        temp_obj = cls(None, None, True, store_graphs)  # Assuming __init__ takes three arguments

        # Copy all attributes from the temporary object to the new object
        for name, value in vars(temp_obj).items():
            if not hasattr(obj, name):
                setattr(obj, name, value)

        # set the store_graphs attribute
        obj._store_graphs = store_graphs

        variables = []
        maxlag = 0
        # now for each node in tsgraph set the metadata
        for node in obj.get_nodes():
            # get the variable name and lag from the node name
            variable_name, lag = get_variable_name_and_lag(node.identifier)
            maxlag = max(maxlag, lag)
            # set the variable name and lag in the metadata
            node.variable_name = variable_name
            node.time_lag = lag
            variables.append(variable_name) if variable_name not in variables else variables

        obj._order = maxlag
        obj._variables = sorted(variables)
        return obj

    @staticmethod
    def from_adjacency_matrix(
        adjacency_matrix: numpy.ndarray,
        node_names: Optional[List[Union[NodeLike, int]]] = None,
    ) -> TimeSeriesCausalGraph:
        """
        Return a time series causal graph from an adjacency matrix.
        This is useful for converting an adjacency matrix into a time series causal graph.
        """
        graph = CausalGraph.from_adjacency_matrix(adjacency_matrix, node_names=node_names)
        return TimeSeriesCausalGraph.from_causal_graph(graph)

    @staticmethod
    def from_adjacency_matrices(
        adjacency_matrices: Dict[int, numpy.ndarray],
        variable_names: Optional[List[Union[NodeLike, int]]] = None,
    ) -> TimeSeriesCausalGraph:
        """
        Return a time series causal graph from a dictionary of adjacency matrices. Keys are the time delta.
        This is useful for converting a list of adjacency matrices into a time series causal graph.

        For example, the adjacency matrix with time delta -1 is stored in adjacency_matrices[-1] as would correspond to X-1 -> X,
        where X is the set of nodes.

        Example:
        >>> adjacency_matrices = {
        ...     -2: numpy.array([[0, 0, 0], [1, 0, 0], [0, 0, 1]]),
        ...     -1: numpy.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
        ...     0: numpy.array([[0, 1, 0], [0, 0, 1], [0, 1, 0]]),
        ... }

        Let the nodes be X,Y,Z. The corresponding edges are:
        >>> edges = [
        ...     (Z-2, Y),
        ...     (X-1, X),
        ...     (Y-1, X),
        ...     (Y-1, Y),
        ...     (Z-1, Z),
        ...     (X, Z),
        ...     (Z, Y),
        ... ]

        :param adjacency_matrices: A dictionary of adjacency matrices. Keys are the time delta.
        :param variable_names: A list of variable names. If not provided, the variable names are integers starting from 0.
            Node names must correspond to the variable names and must not contain the lag.
        :return: A time series causal graph.
        """
        assert isinstance(adjacency_matrices, dict)
        # keys must be integers or str that can be converted to integers
        assert all(isinstance(key, (int, str)) and int(key) == key for key in adjacency_matrices)
        if variable_names is not None:
            assert len(variable_names) == adjacency_matrices[0].shape[0], (
                'The number of variable names must be equal to the number of nodes in the adjacency matrix.'
                f'Got {len(variable_names)} variable names and {adjacency_matrices[0].shape[0]} nodes.'
            )
        else:
            variable_names = [f'node_{i}' for i in range(adjacency_matrices[0].shape[0])]

        # we could create the full adjacency matrix from the adjacency matrices by stacking them according to the time delta
        # but if we have many time deltas, this could be very memory intensive
        # so we create the graph by adding the edges one by one for each time delta

        # create the empty graph
        tsgraph = TimeSeriesCausalGraph()

        for time_delta, adjacency_matrix in adjacency_matrices.items():
            # create the edges
            edges = []
            # get the edges from the adjacency matrix by getting the indices of the non-zero elements
            for row, column in zip(*numpy.where(adjacency_matrix)):
                edges.append(
                    (
                        get_name_from_lag(variable_names[row], time_delta),
                        variable_names[column],
                    )
                )
            # add the edges to the graph
            tsgraph.add_edges_from(edges)

        return tsgraph

    @property
    def adjacency_matrices(self) -> Dict[int, numpy.ndarray]:
        """
        Return the adjacence matrix dictionry of the `cai_causal_graph.causal_graph.Skeleton` instance.

        The keys are the time deltas and the values are the adjacency matrices.
        """
        adjacency_matrices: Dict[int, numpy.ndarray] = {}
        # get the minimum graph
        graph = self.get_minimum_graph()

        if self.variables is None:
            return adjacency_matrices

        for edge in graph.edges:
            # get the source and destination of the edge
            # extract the variable name and lag from the node attributes
            source_variable_name, source_lag = (
                edge.source.meta[self._meta_variable_name],
                edge.source.meta[self._meta_time_name],
            )
            if source_lag not in adjacency_matrices:
                adjacency_matrices[source_lag] = numpy.zeros((len(self.variables), len(self.variables)))

            destination_variable_name, _ = (
                edge.destination.meta[self._meta_variable_name],
                edge.destination.meta[self._meta_time_name],
            )

            # we only have undirected edges in the Skeleton, so no need to check for other types
            adjacency_matrices[source_lag][
                self.variables.index(source_variable_name), self.variables.index(destination_variable_name)
            ] = 1
        return adjacency_matrices

    @property
    def order(self) -> Optional[int]:
        """Return the order of the graph."""
        # get the maximum lag of the nodes in the graph
        if self._order is None:
            self._order = max([abs(node.get_metadata()[self._meta_time_name]) for node in self.get_nodes()])
        return self._order

    @property
    def variables(self) -> Optional[List[str]]:
        """Return the variables in the graph."""
        if self._variables is None:
            self._variables = [get_variable_name_and_lag(node.identifier)[0] for node in self.get_nodes()]
        return self._variables

    def _update_meta_from_node_names(self):
        """
        Update the metadata from the node names.
        This is just for internal use.
        """
        for node in self.nodes:
            # get the variable name and lag from the node name
            variable_name, lag = get_variable_name_and_lag(node.identifier)
            # set the variable name and lag in the metadata
            node.meta[self._meta_variable_name] = variable_name
            node.meta[self._meta_time_name] = lag
