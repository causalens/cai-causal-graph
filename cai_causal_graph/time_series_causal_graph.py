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
from functools import wraps
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy

from cai_causal_graph import CausalGraph
from cai_causal_graph.graph_components import Edge, Node
from cai_causal_graph.interfaces import HasIdentifier
from cai_causal_graph.type_definitions import EDGE_T, TIME_LAG, VARIABLE_NAME, NodeLike, NodeVariableType
from cai_causal_graph.utils import name_with_lag, get_variable_name_and_lag

logger = logging.getLogger(__name__)

# TODO: we can do one general for many other things as well
def _reset_attributes(func):
    """
    Decorator to reset attributes such as  summary graph, miminal graph etc.

    Whenever a function is called that changes the graph, we need to reset the summary graph etc.
    """
    # TODO: make this more clever as it is not said that we need to reset the summary graph etc
    @wraps(func)
    def wrapper(self: TimeSeriesCausalGraph, *args, **kwargs):
        function = func(self, *args, **kwargs)
        self._summary_graph = None
        self._maxlag = None
        self._variables = None
        return function

    return wrapper


class TimeSeriesNode(Node):
    """Time series node.
    
    A node in a time series causal graph will have additional metadata and attributes that 
    gives the time information of the node together with the variable name.
    
    The two additional metadata are:
    - 'time_lag': the time difference with respect the reference time 0
    - 'variable_name': the name of the variable (without the lag information).
    """
    def __init__(self, identifier: str, meta: Optional[Dict[str, Any]] = None, variable_type: NodeVariableType = NodeVariableType.UNSPECIFIED):
        super().__init__(identifier, meta, variable_type)

    @property
    def time_lag(self) -> int:
        """Return the time lag of the node from the metadata."""
        return self.meta.get('time_lag', 0)

    @property
    def variable_name(self) -> Optional[str]:
        """Return the variable name of the node from the metadata."""
        return self.meta.get('variable_name', None)

class TimeSeriesCausalGraph(CausalGraph):
    """
    A causal graph for time series data.

    The node in a time series causal graph will have additional metadata that
    gives the time information of the node together with the variable name.
    The two additional metadata are:
    - 'time_lag': the time difference with respect the reference time 0
    - 'variable_name': the name of the variable (without the lag information).
    """

    def __init__(
        self,
        input_list: Optional[List[NodeLike]] = None,
        output_list: Optional[List[NodeLike]] = None,
        fully_connected: bool = True,
    ):
        """
        Initialize the time series causal graph.

        :param input_list: list of input nodes. Default is None.
        :param output_list: list of output nodes. Default is None.
        :param fully_connected: if True, the graph will be fully connected.
            Default is True.

        Example usage:
            >>> from cai_causal_graph import CausalGraph, TimeSeriesCausalGraph
            >>>
            >>>
            >>> # How to initialize a TimeSeriesCausalGraph directly
            >>> ts_cg = TimeSeriesCausalGraph()
            >>> ts_cg.add_edge('X1 lag(n=1)', 'X1', edge_type=EDGE_T.DIRECTED_EDGE)
            >>> ts_cg.add_edge('X2 lag(n=1)', 'X2', edge_type=EDGE_T.DIRECTED_EDGE)
            >>> ts_cg.add_edge('X1', 'X3', edge_type=EDGE_T.DIRECTED_EDGE)
            >>>
            >>>
            >>> # How to initialize a TimeSeriesCausalGraph from a CausalGraph
            >>> cg = CausalGraph()
            >>> cg.add_edge('X1 lag(n=1)', 'X1', edge_type=EDGE_T.DIRECTED_EDGE)
            >>> cg.add_edge('X2 lag(n=1)', 'X2', edge_type=EDGE_T.DIRECTED_EDGE)
            >>> cg.add_edge('X1', 'X3', edge_type=EDGE_T.DIRECTED_EDGE)
            >>>
            >>> # The time series causal graph will have the same nodes and edges as the causal graph
            >>> # but it is aware of the time information so 'X1 lag(n=1)' and 'X1' represent the same
            >>> # variable but at different times.
            >>> ts_cg = TimeSeriesCausalGraph.from_causal_graph(cg)
        """
        super().__init__(input_list, output_list, fully_connected)

        # autoregressive order of the graph (max lag)
        self._maxlag: Optional[int] = None
        # list of variables in the graph, i.e. discarding the lags (X1(t-1) and X1 are the same variable)
        self._variables: Optional[List[str]] = None
        self._summary_graph: Optional[CausalGraph] = None
        self._minimal_graph: Optional[TimeSeriesCausalGraph] = None

    def __eq__(self, other: object) -> bool:
        """
        Return True if the graphs are equal.

        Two graphs are equal if they have the same nodes and edges and the same metadata.
        """
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
            node_metadata_variable = node.variable_name
            other_node_metadata_variable = other_node.variable_name

            if node_metadata_variable != other_node_metadata_variable:
                return False

            # check the time delta
            node_metadata_td = node.time_lag
            other_node_metadata_td = other_node.time_lag

            if node_metadata_td != other_node_metadata_td:
                return False

        return True

    def copy(self, include_meta: bool = True) -> TimeSeriesCausalGraph:
        """
        Return a copy of the graph.

        :param include_meta: if True, the metadata will be copied as well. Default is True.
        """
        graph = super().copy(include_meta=include_meta)
        # cast the graph to TimeSeriesCausalGraph to have the correct metadata
        return TimeSeriesCausalGraph.from_causal_graph(graph)

    def get_minimal_graph(self) -> TimeSeriesCausalGraph:
        """
        Return a minimal graph.

        The minimal graph is the graph with the minimal number of edges that is equivalent to the original graph.
        In other words, it is a graph that has no edges whose destination is not time delta 0.

        Example:
        Input graph:
            - X1(t-2)-> X1(t-1) -> X2(t-1) -> X2(t), X1(t) -> X2(t), X1(t-1) -> X2(t-1)
        Minimal graph:
            - X1(t-1) -> X1(t) -> X2(t)
        """
        minimal_cg = TimeSeriesCausalGraph()

        for edge in self.get_edges():
            # get the relative time delta
            time_delta = edge.destination.time_lag - edge.source.time_lag
            # add the edge if the time delta is 0 (no need to extract the new names)
            if time_delta == 0 and not minimal_cg.edge_exists(edge.source.variable_name, edge.destination.variable_name):
                minimal_cg.add_edge(
                    edge.source.variable_name,
                    edge.destination.variable_name,
                )
            # otherwise if the time delta is not 0, we may have X[t-2]->X[t-1] and 
            # we must add X[t-1]->X[t]
            else:
                # get the new names according to the time delta
                destination_name = name_with_lag(edge.destination.identifier, 0)
                source_name = name_with_lag(edge.source.identifier, -time_delta)
                if not minimal_cg.edge_exists(source_name, destination_name):
                    # add the edge
                    minimal_cg.add_edge(
                        source_name,
                        destination_name,
                    )

        return minimal_cg

    def is_minimal_graph(self) -> bool:
        """
        Return True if the graph is minimal.

        See `get_minimal_graph` for more details.
        """
        return self == self.get_minimal_graph()

    def get_summary_graph(self) -> CausalGraph:
        """
        Return a summary graph.

        Collapse graph in time into a single node per variable (column name).
        This can become cyclic and bi-directed as X-1 -> Y and Y-1 -> X would become X <-> Y.

        There are several cases to consider. Assume the edge in consideration is called B.
        - if the edge is not in the summary graph, add it
        - if there is an edge A is in the summary graph, then:
            - if A and B have the same direction, keep the direction
            - if A and B have different directions, make it bi-directed
            - if one of the two is already bi-directed, keep it bi-directed

        :return: The summary graph as a `CausalGraph` object.
        """
        if self._summary_graph is None:
            summary_graph = CausalGraph()
            # now check as described above (assume edges are already directed)
            edges = self.get_edges()
            for edge in edges:
                # first we need to extract the variable names from the nodes as the summary graph
                # will have the variable names as nodes
                source_node = edge.source
                destination_node = edge.destination
                source_variable_name = source_node.variable_name
                destination_variable_name = destination_node.variable_name

                assert (
                    source_variable_name is not None
                ), 'Source variable name is None, cannot create summary graph. The edge is: {}'.format(edge)

                assert (
                    destination_variable_name is not None
                ), 'Destination variable name is None, cannot create summary graph. The edge is: {}'.format(edge)

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
        nodes at time t-2 and all the nodes that are connected to them as specified by the minimal graph.

        If both backward_steps and forward_steps are None, return the original graph.

        :param backward_steps: Number of steps to extend the graph backwards in time. If None, do not extend backwards.
        :param forward_steps: Number of steps to extend the graph forwards in time. If None, do not extend forwards.
        :return: Extended graph with nodes for each variable at each time step from backward_steps to forward_steps.
        """
        # check steps are valid (positive integers) if not None
        assert backward_steps is None or backward_steps > 0
        assert forward_steps is None or forward_steps > 0

        # first get the minimal graph
        minimal_graph = self.get_minimal_graph()

        # create a new graph by copying the minimal graph
        extended_graph = minimal_graph.copy()

        if backward_steps is not None:
            assert backward_steps > 0, 'backward_steps must be a positive integer.'
            # start from 1 as 0 is already defined
            # we cannot start directly from maxlag as it may be possible
            # that not all the nodes from 1 to -maxlag are defined (as they were not
            # needed in the mimimal graph)
            maxlag = minimal_graph.maxlag
            assert maxlag is not None

            for lag in range(1, backward_steps + 1):
                for edge in minimal_graph.get_edges():
                    time_delta = edge.destination.time_lag - edge.source.time_lag
                    
                    lagged_destination_node = self._get_lagged_node(node=edge.destination, lag=-lag)
                    # check if the new source node would go beyond the backward_steps
                    if -lag -time_delta < -backward_steps:
                        # add the destination node if it does not exist (e.g. floating nodes)
                        if not extended_graph.node_exists(lagged_destination_node.identifier):
                            extended_graph.add_node(node=lagged_destination_node)
                        continue
                    
                    lagged_source_node = self._get_lagged_node(node=edge.source, lag=-lag - time_delta)
                    
                    # add the lagged nodes
                    if not extended_graph.node_exists(lagged_source_node.identifier):
                        extended_graph.add_node(node=lagged_source_node)

                    if not extended_graph.node_exists(lagged_destination_node.identifier):
                        extended_graph.add_node(node=lagged_destination_node)

                    # add the lagged edge
                    if not extended_graph.edge_exists(lagged_source_node.identifier, lagged_destination_node.identifier):
                        extended_graph.add_edge(
                            source=lagged_source_node,
                            destination=lagged_destination_node,
                            meta=edge.meta,
                        )

            # add a waring if the backward_steps is smaller than the maximum lag in the graph
            if backward_steps < maxlag:
                ramaining_nodes = []
                for node in minimal_graph.get_nodes():
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

            # first create all the nodes from 1 to forward_steps
            for lag in range(1, forward_steps + 1):
                for node in minimal_graph.get_nodes():
                    # create the node with +lag (if it does not exist)
                    lagged_node = self._get_lagged_node(node=node, lag=lag)
                    # if node does not exist, add it
                    if not extended_graph.node_exists(lagged_node.identifier):
                        extended_graph.add_node(node=lagged_node)

            for lag in range(1, forward_steps + 1):
                for edge in minimal_graph.get_edges():
                    # create the edge with +lag
                    source = edge.source
                    destination = edge.destination

                    # now lag the source and destination of +lag
                    lagged_source = self._get_lagged_node(node=source, lag=source.time_lag + lag)
                    lagged_dest = self._get_lagged_node(node=destination, lag=destination.time_lag + lag)

                    lagged_edge = Edge(
                        source=extended_graph.get_node(lagged_source.identifier),  # make sure the node exists
                        destination=extended_graph.get_node(lagged_dest.identifier),
                        edge_type=edge.get_edge_type(),
                        meta=edge.meta,  # copy the meta from the original edge
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

    @staticmethod
    def _get_lagged_node(identifier: Optional[NodeLike] = None, node: Optional[Node] = None, lag: Optional[int] = None):
        """
        Return the lagged node of a node with a given lag. Lag is overwritten if the node is already lagged.

        For example, if the node is X and the lag is -1, the node will be X lag(n=1).
        If the node is X lag(n=1) and the lag is -2, the node will be X lag(n=2).

        Moreover, if you want to make sure to copy all the metadata from the original node, you have to provide the node.

        :param identifier: The identifier of the node. Default is None.
        :param node: The node. If provided, the identifier is ignored. Default is None.
        :param lag: The lag of the node. Default is None. If None, the lag is set to 0.
        :return: The lagged node.
        """
        # either identifier or node must be provided
        assert identifier is not None or node is not None, 'Either identifier or node must be provided.'

        if lag is None:
            lag = 0

        if node is not None:
            identifier = node.identifier
        elif isinstance(identifier, HasIdentifier):
            identifier = identifier.identifier

        assert isinstance(identifier, str), 'The identifier must be a string. Got %s.' % type(identifier)

        lagged_identifier = name_with_lag(identifier, lag)

        if node is None:
            # extract the variable name from the identifier
            variable_name, _ = get_variable_name_and_lag(identifier)

            meta = {
                'variable_name': variable_name,
                'time_lag': lag,
            }
        else:
            # modify the identifier of the provided node
            # update the meta information
            meta = node.meta.copy()
            # meta['time_lag'] = lag
            # meta['variable_name'] = node.variable_name
            node = Node(identifier=lagged_identifier, meta=meta)
            # update the meta information
            TimeSeriesCausalGraph._update_node_meta(node, node.variable_name, lag)
            return node

    @_reset_attributes
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
        """
        Add a node to the graph. See `cai_causal_graph.causal_graph.CausalGraph.add_node` for more details.

        In addition to the `CausalGraph.add_node` method, this method also populates the metadata of the node with
        the variable name and the time lag.

        :param identifier: The identifier of the node.
        :param variable_type: The type of the variable.
        :param meta: The metadata of the node.
        :param node: The node to add.
        :param kwargs: Additional keyword arguments.
        :return: The added node.
        """
        node = super().add_node(identifier, variable_type=variable_type, meta=meta, node=node, **kwargs)
        # populate the metadata for each node
        vname, lag = get_variable_name_and_lag(node.identifier)
        TimeSeriesCausalGraph._update_node_meta(node, vname, lag)
        return node

    @_reset_attributes
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
        Replace a node in the graph.

        See `cai_causal_graph.causal_graph.CausalGraph.replace_node` for more details.
        """
        super().replace_node(node_id, new_node_id, variable_type=variable_type, meta=meta)
        vname, lag = get_variable_name_and_lag(new_node_id.identifier)
        TimeSeriesCausalGraph._update_node_meta(new_node_id, vname, lag)

    @_reset_attributes
    def delete_node(self, identifier: NodeLike):
        """
        Delete a node from the graph.

        See `cai_causal_graph.causal_graph.CausalGraph.delete_node` for more details.
        """
        super().delete_node(identifier)

    @_reset_attributes
    def delete_edge(self, source: NodeLike, destination: NodeLike):
        """
        Delete an edge from the graph.

        See `cai_causal_graph.causal_graph.CausalGraph.delete_edge` for more details.
        """
        super().delete_edge(source, destination)

    @_reset_attributes
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
        """
        Add an edge to the graph. See `cai_causal_graph.causal_graph.CausalGraph.add_edge` for more details.

        In addition to the `CausalGraph.add_edge` method, this method also populates the metadata of the nodes with
        the variable name and the time lag.
        """
        edge = super().add_edge(source, destination, edge_type=edge_type, meta=meta, edge=edge, **kwargs)
        
        # just need to update the metadata of the source destination node
        # the metadata of the other node is not affected
        for node in [edge.source, edge.destination]:
            # extract variable name and lag from node name
            vname, lag = get_variable_name_and_lag(node.identifier)
            TimeSeriesCausalGraph._update_node_meta(node, vname, lag)

        return edge

    @classmethod
    def from_causal_graph(cls, causal_graph: CausalGraph) -> TimeSeriesCausalGraph:
        """
        Return a time series causal graph from a causal graph.

        This is useful for converting a causal graph from a single time step into a time series causal graph.

        :param causal_graph: The causal graph.
        """
        # create an instance without calling __init__
        obj = cls.__new__(cls)

        # copy all attributes from causal_graph to the new object
        for name, value in vars(causal_graph).items():
            setattr(obj, name, value)

        # set all the attributes related to time series

        # create a temporary TimeSeriesCausalGraph object to get default values for new attributes
        temp_obj = cls(input_list=None, output_list=None, fully_connected=True)

        # Copy all attributes from the temporary object to the new object
        for name, value in vars(temp_obj).items():
            if not hasattr(obj, name):
                setattr(obj, name, value)

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

        obj._maxlag = maxlag
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
            variable_names_str: List[Union[str, int]] = []
            assert len(variable_names) == adjacency_matrices[0].shape[0], (
                'The number of variable names must be equal to the number of nodes in the adjacency matrix.'
                f'Got {len(variable_names)} variable names and {adjacency_matrices[0].shape[0]} nodes.'
            )
            # convert the variable names to strings if they are not strings
            for variable_name in variable_names:
                if isinstance(variable_name, HasIdentifier):
                    variable_names_str.append(variable_name.identifier)
                else:
                    variable_names_str.append(variable_name)

        else:
            variable_names_str = [f'node_{i}' for i in range(adjacency_matrices[0].shape[0])]

        # we could create the full adjacency matrix from the adjacency matrices by stacking them according to the time delta
        # but if we have many time deltas, this could be very memory intensive
        # so we create the graph by adding the edges one by one for each time delta

        # create the empty graph
        tsgraph = TimeSeriesCausalGraph()

        for time_delta, adjacency_matrix in adjacency_matrices.items():
            # create the edges
            edges: List[Tuple[str, str]] = []
            # get the edges from the adjacency matrix by getting the indices of the non-zero elements
            for row, column in zip(*numpy.where(adjacency_matrix)):
                edges.append(
                    (
                        name_with_lag(variable_names_str[row], time_delta),
                        variable_names_str[column],
                    )
                )
            # add the edges to the graph
            tsgraph.add_edges_from(edges)   # type: ignore

        return tsgraph

    @property
    def adjacency_matrices(self) -> Dict[int, numpy.ndarray]:
        """
        Return the adjacency matrix dictionary of the minimal causal graph.

        The keys are the time deltas and the values are the adjacency matrices.
        """
        adjacency_matrices: Dict[int, numpy.ndarray] = {}
        # get the minimal graph
        graph = self.get_minimal_graph()

        if self.variables is None:
            return adjacency_matrices

        for edge in graph.edges:
            # get the source and destination of the edge
            # extract the variable name and lag from the node attributes
            source_variable_name, source_lag = (
                edge.source.meta[VARIABLE_NAME],
                edge.source.meta[TIME_LAG],
            )
            if source_lag not in adjacency_matrices:
                adjacency_matrices[source_lag] = numpy.zeros((len(self.variables), len(self.variables)))

            destination_variable_name, _ = (
                edge.destination.meta[VARIABLE_NAME],
                edge.destination.meta[TIME_LAG],
            )

            if edge.get_edge_type() == EDGE_T.DIRECTED_EDGE:
                adjacency_matrices[source_lag][
                    self.variables.index(source_variable_name), self.variables.index(destination_variable_name)
                ] = 1
            elif edge.get_edge_type() == EDGE_T.UNDIRECTED_EDGE:
                adjacency_matrices[source_lag][
                    self.variables.index(source_variable_name), self.variables.index(destination_variable_name)
                ] = 1
                adjacency_matrices[source_lag][
                    self.variables.index(destination_variable_name), self.variables.index(source_variable_name)
                ] = 1
            else:
                raise TypeError(
                    f'Adjacency matrices can only be computed if the CausalGraph instance solely contains directed and '
                    f'undirected edges. Got {edge.get_edge_type()} for the edge {edge.descriptor}.'
                )
        return adjacency_matrices

    @property
    def maxlag(self) -> Optional[int]:
        """
        Return the autoregressive order of the graph.

        The autoregressive order of the graph is the maximum lag of the nodes in the minimal graph.
        """
        # get the maximum lag of the nodes in the graph
        if self._maxlag is None:
            self._maxlag = max([abs(node.time_lag) for node in self.get_minimal_graph().get_nodes()])
        return self._maxlag

    @property
    def variables(self) -> Optional[List[str]]:
        """
        Return the variables in the graph.

        Variables differ from nodes in that they do not contain the lag.
        For example, if the graph contains the node "X1 lag(n=2)", the variable is "X1".
        """
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
            self._update_node_meta(node, variable_name, lag)

    @staticmethod
    def _update_node_meta(node: Node, variable_name: Optional[str] = None, lag: Optional[int] = None):
        """
        Update the metadata of a node.

        This is just for internal use.
        """
        node.meta[VARIABLE_NAME] = variable_name
        node.meta[TIME_LAG] = lag
