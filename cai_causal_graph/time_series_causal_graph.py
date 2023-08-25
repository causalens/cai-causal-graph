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
from copy import deepcopy
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy

from cai_causal_graph import CausalGraph
from cai_causal_graph.exceptions import CausalGraphErrors
from cai_causal_graph.graph_components import Edge, Node
from cai_causal_graph.interfaces import HasIdentifier, HasMetadata
from cai_causal_graph.type_definitions import EDGE_T, TIME_LAG, VARIABLE_NAME, NodeLike, NodeVariableType
from cai_causal_graph.utils import get_name_with_lag, get_variable_name_and_lag

logger = logging.getLogger(__name__)


# TODO: we can do one general for many other things as well
def _reset_attributes(func):
    """
    Decorator to reset attributes such as summary graph, minimal graph, etc.

    Whenever a function is called that changes the graph, we need to reset the summary graph etc.
    """
    # TODO: make this more clever as it is not said that we need to reset the summary graph etc
    @wraps(func)
    def wrapper(self: TimeSeriesCausalGraph, *args, **kwargs):
        function = func(self, *args, **kwargs)
        self._minimal_graph = None
        self._summary_graph = None
        self._maxlag = None
        self._variables = None
        return function

    return wrapper


class TimeSeriesNode(Node):
    """
    Time series node.

    A node in a time series causal graph will have additional metadata and attributes that gives the time information
    of the node together with the variable name.

    The two additional metadata are:
    - `cai_causal_graph.type_definitions.TIME_LAG`: the time difference with respect to the reference time 0
    - `cai_causal_graph.type_definitions.VARIABLE_NAME`: the name of the variable (without the lag information)
    """

    def __init__(
        self,
        identifier: Optional[NodeLike] = None,
        time_lag: Optional[int] = None,
        variable_name: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        variable_type: NodeVariableType = NodeVariableType.UNSPECIFIED,
    ):
        """
        Initialize the time series node.

        :param identifier: the identifier of the node. If the identifier is provided, the time_lag and
            variable_name will be extracted from the identifier. Default is None.
        :param time_lag: the time lag of the node. If `time_lag` is provided, then `variable_name` must be provided
            to set the identifier. If both `time_lag` and `variable_name` are provided, the identifier must be None.
            Default is None.
        :param variable_name: the variable name of the node. If `variable_name` is provided, then `time_lag` must be
            provided to set the identifier. If both `time_lag` and `variable_name` are provided, the identifier must
            be None. Default is None.
        :param meta: the metadata of the node. Default is None.
        :param variable_type: the variable type of the node. Default is NodeVariableType.UNSPECIFIED.
        """
        if time_lag is not None and variable_name is not None:
            assert identifier is None, 'If time_lag and variable_name are provided, identifier must be None.'
            identifier = get_name_with_lag(variable_name, time_lag)
        elif identifier is not None:
            assert (
                time_lag is None and variable_name is None
            ), 'If identifier is provided, `time_lag` and `variable_name` must be None.'
            identifier = Node.identifier_from(identifier)
            variable_name, time_lag = get_variable_name_and_lag(identifier)
        else:
            raise ValueError(
                'Either identifier or both time_lag and variable_name must be provided to initialize a node.'
            )

        # populate the metadata for each node
        if meta is not None:
            meta.update({TIME_LAG: time_lag, VARIABLE_NAME: variable_name})
        else:
            meta = {TIME_LAG: time_lag, VARIABLE_NAME: variable_name}

        # populate the metadata for each node
        super().__init__(identifier, meta, variable_type)

    @property
    def time_lag(self) -> int:
        """Return the time lag of the node from the metadata."""
        return self.meta.get(TIME_LAG, 0)

    @property
    def variable_name(self) -> Optional[str]:
        """Return the variable name of the node from the metadata."""
        return self.meta.get(VARIABLE_NAME, None)

    def __repr__(self) -> str:
        return super().__repr__().replace('Node', 'TimeSeriesNode')


class TimeSeriesCausalGraph(CausalGraph):
    """
    A causal graph for time series data.

    The node in a time series causal graph will have additional metadata that
    gives the time information of the node together with the variable name.
    The two additional metadata are:
    - `cai_causal_graph.type_definitions.TIME_LAG`: the time difference with respect to the reference time 0
    - `cai_causal_graph.type_definitions.VARIABLE_NAME`: the name of the variable (without the lag information)
    """

    def __init__(
        self,
        input_list: Optional[List[NodeLike]] = None,
        output_list: Optional[List[NodeLike]] = None,
        fully_connected: bool = False,
    ):
        """
        Initialize the time series causal graph.

        :param input_list: list of input nodes. Default is None.
        :param output_list: list of output nodes. Default is None.
        :param fully_connected: if `True`, the graph will be fully connected from inputs to outputs.
            Default is `False`.

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
            >>> # The time series causal graph will have the same nodes and edges as the causal graph,
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
            assert isinstance(other_node, TimeSeriesNode)

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
            assert isinstance(edge.source, TimeSeriesNode)
            assert isinstance(edge.destination, TimeSeriesNode)
            assert isinstance(edge.source.time_lag, int)
            assert isinstance(edge.destination.time_lag, int)
            assert edge.source.variable_name is not None
            assert edge.destination.variable_name is not None

            time_delta = edge.destination.time_lag - edge.source.time_lag
            # add the edge if the time delta is 0 (no need to extract the new names)
            if time_delta == 0 and not minimal_cg.edge_exists(
                edge.source.variable_name, edge.destination.variable_name
            ):
                minimal_cg.add_edge(
                    edge.source.variable_name,
                    edge.destination.variable_name,
                )
            # otherwise if the time delta is not 0, we may have X[t-2]->X[t-1] and
            # we must add X[t-1]->X[t]
            else:
                # get the new names according to the time delta
                destination_name = get_name_with_lag(edge.destination.identifier, 0)
                source_name = get_name_with_lag(edge.source.identifier, -time_delta)
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
                assert isinstance(source_node, TimeSeriesNode)
                assert isinstance(destination_node, TimeSeriesNode)

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

        Extend the graph in time by adding nodes for each variable at each time step from `backward_steps` to
        `forward_steps`. If a backward step of n is specified, it means that the graph will be extended in order to
        include nodes back to time -n. For example, if t is the reference time, if `backward_steps` is 2, the graph
        will be extended to include nodes back to time t-2. This does not mean that the graph will be extended to only
        include nodes at time t-2, but rather that it will include nodes at time t-2 and all the nodes that are
        connected to them as specified by the minimal graph.

        If both `backward_steps` and `forward_steps` are None, the original graph is returned.

        :param backward_steps: Number of steps to extend the graph backwards in time. If None, do not extend backwards.
        :param forward_steps: Number of steps to extend the graph forwards in time. If None, do not extend forwards.
        :return: Extended graph with nodes for each variable at each time step from `backward_steps` to `forward_steps`.
        """
        # check steps are valid (positive integers) if not None
        if backward_steps is not None:
            assert backward_steps == int(backward_steps), f'backward_steps must be an integer. Got {backward_steps}.'
            assert backward_steps > 0, f'backward_steps must be a positive integer. Got {backward_steps}.'
        if forward_steps is not None:
            assert forward_steps == int(forward_steps), f'backward_steps must be an integer. Got {forward_steps}.'
            assert forward_steps > 0, f'forward_steps must be a positive integer. Got {forward_steps}.'

        # first get the minimal graph
        minimal_graph = self.get_minimal_graph()

        # create a new graph by copying the minimal graph
        extended_graph = minimal_graph.copy()

        if backward_steps is not None:
            # Start from 1 as 0 is already defined.
            # We cannot start directly from maxlag as it may be possible that not all the nodes from 1 to -maxlag are
            # defined (as they were not needed in the minimal graph).
            maxlag = minimal_graph.maxlag
            assert maxlag is not None

            for lag in range(1, backward_steps + 1):
                for edge in minimal_graph.get_edges():
                    assert isinstance(edge.destination, TimeSeriesNode)
                    assert isinstance(edge.source, TimeSeriesNode)
                    time_delta = edge.destination.time_lag - edge.source.time_lag

                    lagged_destination_node = self._get_lagged_node(node=edge.destination, lag=-lag)
                    # check if the new source node would go beyond the backward_steps
                    if -lag - time_delta < -backward_steps:
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
                    if not extended_graph.edge_exists(
                        lagged_source_node.identifier, lagged_destination_node.identifier
                    ):
                        extended_graph.add_edge(
                            source=lagged_source_node,
                            destination=lagged_destination_node,
                            meta=edge.meta,
                        )

            # Log a warning if the backward_steps is smaller than the maximum lag in the graph.
            if backward_steps < maxlag:
                remaining_nodes = []
                for node in minimal_graph.get_nodes():
                    if abs(node.time_lag) > backward_steps:
                        remaining_nodes.append(node.identifier)
                logger.warning(
                    'backward_steps is smaller than the maximum lag in the graph, the following nodes will not be added: %s',
                    list(set(remaining_nodes)),
                )

        if forward_steps is not None:
            # Start from 1 as 0 is already defined.
            # With the forward extension, the maximum positive lag is forward_steps.

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
                    assert isinstance(source, TimeSeriesNode)
                    assert isinstance(destination, TimeSeriesNode)

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
    def _get_lagged_node(
        identifier: Optional[NodeLike] = None, node: Optional[TimeSeriesNode] = None, lag: Optional[int] = None
    ):
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

        if node is None:
            # extract the variable name from the identifier
            variable_name, _ = get_variable_name_and_lag(identifier)

            meta = {
                VARIABLE_NAME: variable_name,
                TIME_LAG: lag,
            }
        else:
            # modify the identifier of the provided node
            # update the meta information
            meta = node.meta.copy()

        assert node is not None, 'The node must be valid. Got None.'
        node = TimeSeriesNode(variable_name=node.variable_name, time_lag=lag, meta=meta)
        return node

    @_reset_attributes
    def add_node(
        self,
        /,
        identifier: Optional[NodeLike] = None,
        variable_name: Optional[str] = None,
        time_lag: Optional[int] = None,
        *,
        variable_type: NodeVariableType = NodeVariableType.UNSPECIFIED,
        meta: Optional[dict] = None,
        node: Optional[TimeSeriesNode] = None,
        **kwargs,
    ) -> TimeSeriesNode:
        """
        Add a node to the time series graph. See `cai_causal_graph.causal_graph.CausalGraph.add_node` for more details.

        In addition to the `CausalGraph.add_node` method, this method also populates the metadata of the node with
        the variable name and the time lag.

        :param identifier: The identifier of the time series node.
        :param variable_name: The variable name of the time series node.
        :param time_lag: The time lag of the time series node.
        :param variable_type: The type of the variable.
        :param meta: The metadata of the time series node.
        :param node: The node to add.
        :param kwargs: Additional keyword arguments.
        :return: The added node.
        """
        if node is not None:
            assert (
                identifier is None
                and variable_name is None
                and time_lag is None
                and variable_type == NodeVariableType.UNSPECIFIED
                and meta is None
                and len(kwargs) == 0
            ), 'If specifying `node` argument, all other arguments should not be specified.'
            identifier = node.identifier
            variable_type = node.variable_type
            meta = deepcopy(node.meta)
        else:
            # either identifier or (variable_name and time_lag) must be provided
            assert identifier is not None or (
                variable_name is not None and time_lag is not None
            ), 'Either identifier or (variable_name and time_lag) must be provided.'

        node = TimeSeriesNode(
            identifier=identifier,
            time_lag=time_lag,
            variable_name=variable_name,
            variable_type=variable_type,
            meta=meta,
        )
        identifier = node.identifier
        if identifier in self._nodes_by_identifier:
            raise CausalGraphErrors.NodeDuplicatedError(f'Node already exists: {identifier}')

        self._nodes_by_identifier[identifier] = node

        return node

    @_reset_attributes
    def replace_node(
        self,
        /,
        node_id: NodeLike,
        new_node_id: Optional[NodeLike] = None,
        time_lag: Optional[int] = None,
        variable_name: Optional[str] = None,
        *,
        variable_type: NodeVariableType = NodeVariableType.UNSPECIFIED,
        meta: Optional[dict] = None,
    ):
        """
        Replace a node in the graph.

        See `cai_causal_graph.causal_graph.CausalGraph.replace_node` for more details.
        """
        # either new_node_id or (variable_name and time_lag) must be provided
        assert new_node_id is not None or (
            variable_name is not None and time_lag is not None
        ), 'Either new_node_id or (variable_name and time_lag) must be provided.'

        if new_node_id is None:
            # update name from the variable name and time lag
            new_node_id = get_name_with_lag(variable_name, time_lag)   # type: ignore
        else:
            new_node_id = Node.identifier_from(new_node_id)
            variable_name, time_lag = get_variable_name_and_lag(Node.identifier_from(new_node_id))

        if meta is not None:
            meta = meta.copy()
            meta.update({VARIABLE_NAME: variable_name, TIME_LAG: time_lag})
        else:
            meta = {VARIABLE_NAME: variable_name, TIME_LAG: time_lag}

        super().replace_node(node_id, new_node_id, variable_type=variable_type, meta=meta)

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
        # if the source and destination time series nodes do not exist, create them
        if source is not None and not self.node_exists(source):
            source_meta = None
            if isinstance(source, HasMetadata):
                source_meta = source.get_metadata()
                source = self.add_node(source, meta=source_meta)

        if destination is not None and not self.node_exists(destination):
            destination_meta = None
            if isinstance(destination, HasMetadata):
                destination_meta = destination.get_metadata()
                destination = self.add_node(destination, meta=destination_meta)

        edge = super().add_edge(source, destination, edge_type=edge_type, meta=meta, edge=edge, **kwargs)

        return edge

    @classmethod
    def from_causal_graph(cls, causal_graph: CausalGraph) -> TimeSeriesCausalGraph:
        """
        Return a time series causal graph from a causal graph.

        This is useful for converting a causal graph from a single time step into a time series causal graph.

        :param causal_graph: The causal graph.
        """

        sepsets = deepcopy(causal_graph._sepsets)

        # copy nodes and make them TimeSeriesNodes
        ts_cg = cls()
        for node in causal_graph.get_nodes():
            meta = deepcopy(node.meta)
            # get the variable name and lag from the node name
            variable_name, lag = get_variable_name_and_lag(node.identifier)
            node = TimeSeriesNode(variable_name=variable_name, time_lag=lag, meta=meta)
            ts_cg.add_node(node)

        # copy edges
        for edge in causal_graph.get_edges():
            source = ts_cg.get_node(edge.source)
            destination = ts_cg.get_node(edge.destination)
            ts_cg.add_edge(source, destination, meta=edge.meta)

        ts_cg._sepsets = sepsets

        return ts_cg

    @staticmethod
    def from_adjacency_matrix(
        adjacency: numpy.ndarray,
        node_names: Optional[List[Union[NodeLike, int]]] = None,
    ) -> TimeSeriesCausalGraph:
        """
        Construct a `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` instance from an adjacency matrix
        and optionally a list of node names.

        :param adjacency: A square binary numpy adjacency array.
        :param node_names: A list of strings, `cai_causal_graph.interfaces.HasIdentifier`, and/or integers which can be
            coerced to `cai_causal_graph.graph_components.Node`.
        :return: A `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` object.
        """
        graph = CausalGraph.from_adjacency_matrix(adjacency, node_names=node_names)
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
        ...     0: numpy.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]]),
        ... }

        Let the nodes be X,Y,Z. The corresponding edges are:
        >>> edges = [
        ...     (Z-2, Y),
        ...     (X-1, X),
        ...     (Y-1, X),
        ...     (Y-1, Y),
        ...     (Z-1, Z),
        ...     (X, Y),
        ...     (X, Z),
        ...     (Y, Z),
        ... ]

        :param adjacency_matrices: A dictionary of adjacency matrices. Keys are the time delta.
        :param variable_names: A list of variable names. If not provided, the variable names are integers starting
            from 0. Node names must correspond to the variable names and must not contain the lag.
        :return: A time series causal graph.
        """
        assert isinstance(adjacency_matrices, dict)
        # keys must be integers or str that can be converted to integers
        assert all(isinstance(key, (int, str)) and int(key) == key for key in adjacency_matrices)

        # get the shape of the adjacency matrices (get the first key as 0 may not be present)
        shape = adjacency_matrices[next(iter(adjacency_matrices))].shape

        if variable_names is not None:
            variable_names_str: List[Union[str, int]] = []
            assert len(variable_names) == shape[0], (
                'The number of variable names must be equal to the number of nodes in the adjacency matrix.'
                f'Got {len(variable_names)} variable names and {shape[0]} nodes.'
            )
            # convert the variable names to strings if they are not strings
            for variable_name in variable_names:
                if isinstance(variable_name, HasIdentifier):
                    variable_names_str.append(variable_name.identifier)
                else:
                    variable_names_str.append(variable_name)

        else:
            variable_names_str = [f'node_{i}' for i in range(shape[0])]

        # We could create the full adjacency matrix from the adjacency matrices by stacking them according to the time
        # delta but if we have many time deltas, this could be very memory intensive. Therefore, we create the graph by
        # adding the edges one by one for each time delta.

        # create the empty graph
        tsgraph = TimeSeriesCausalGraph()

        for time_delta, adjacency_matrix in adjacency_matrices.items():
            # create the edges
            edges: List[Tuple[str, str]] = []
            # get the edges from the adjacency matrix by getting the indices of the non-zero elements
            for row, column in zip(*numpy.where(adjacency_matrix)):
                edges.append(
                    (
                        get_name_with_lag(variable_names_str[row], time_delta),
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
            self._maxlag = max([abs(node.time_lag) for node in self.get_minimal_graph().get_nodes()])   # type: ignore
        return self._maxlag

    @property
    def variables(self) -> Optional[List[str]]:
        """
        Return the variables in the graph.

        Variables differ from nodes in that they do not contain the lag.
        For example, if the graph contains the node "X1 lag(n=2)", the variable is "X1".
        """
        if self._variables is None:
            variables = []
            for node in self.get_nodes():
                assert node.variable_name is not None
                variables.append(node.variable_name)
            self._variables = sorted(list(set(variables)))
        return self._variables

    def get_nodes(self, identifier: Optional[Union[NodeLike, List[NodeLike]]] = None) -> List[TimeSeriesNode]:  # type: ignore
        """
        Return the time series nodes in the graph.

        :param identifier: The identifier of the node(s) to return.
        :return: A list of nodes.
        """
        nodes = super().get_nodes(identifier)
        # check all nodes are TimeSeriesNode
        assert all(isinstance(node, TimeSeriesNode) for node in nodes)
        return cast(List[TimeSeriesNode], nodes)
