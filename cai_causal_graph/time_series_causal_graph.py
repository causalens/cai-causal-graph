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
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import networkx
import numpy

from cai_causal_graph import CausalGraph, Skeleton
from cai_causal_graph.graph_components import Edge, Node, TimeSeriesNode
from cai_causal_graph.interfaces import HasIdentifier, HasMetadata
from cai_causal_graph.type_definitions import TIME_LAG, VARIABLE_NAME, EdgeType, NodeLike, NodeVariableType
from cai_causal_graph.utils import get_name_with_lag, get_variable_name_and_lag

logger = logging.getLogger(__name__)


def _reset_ts_graph_attributes(func: Callable) -> Callable:
    """
    Decorator to reset attributes of TimeSeriesCausalGraph such as summary graph, minimal graph, etc.

    Whenever a function is called that changes the graph, we need to reset these attributes.
    """

    # TODO - CAUSALAI-3369: Improve this decorator to remove need to reset the summary graph and other attributes
    @wraps(func)
    def wrapper(self: TimeSeriesCausalGraph, *args, **kwargs) -> Any:
        function = func(self, *args, **kwargs)
        self._minimal_graph = None
        self._summary_graph = None
        self._stationary_graph = None
        self._variables = None
        return function

    return wrapper


class TimeSeriesCausalGraph(CausalGraph):
    """
    A causal graph for time series data.

    The node in a time series causal graph will have additional metadata that provides the time information of the
    node together with the variable name.
    The two additional metadata are:
    - `cai_causal_graph.type_definitions.TIME_LAG`: the time difference with respect to the reference time 0
    - `cai_causal_graph.type_definitions.VARIABLE_NAME`: the name of the variable (without the lag information)
    """

    _NodeCls: Type[TimeSeriesNode] = TimeSeriesNode
    _EdgeCls: Type[Edge] = Edge
    _SummaryGraphCls: Type[CausalGraph] = CausalGraph

    def __init__(
        self,
        input_list: Optional[List[NodeLike]] = None,
        output_list: Optional[List[NodeLike]] = None,
        fully_connected: bool = False,
    ):
        """
        Initialize the time series causal graph.

        Example usage:
            >>> from cai_causal_graph import CausalGraph, TimeSeriesCausalGraph
            >>>
            >>>
            >>> # How to initialize a TimeSeriesCausalGraph directly
            >>> ts_cg = TimeSeriesCausalGraph()
            >>> ts_cg.add_edge('X1 lag(n=1)', 'X1', edge_type=EdgeType.DIRECTED_EDGE)
            >>> ts_cg.add_edge('X2 lag(n=1)', 'X2', edge_type=EdgeType.DIRECTED_EDGE)
            >>> ts_cg.add_edge('X1', 'X3', edge_type=EdgeType.DIRECTED_EDGE)
            >>>
            >>>
            >>> # How to initialize a TimeSeriesCausalGraph from a CausalGraph
            >>> cg = CausalGraph()
            >>> cg.add_edge('X1 lag(n=1)', 'X1', edge_type=EdgeType.DIRECTED_EDGE)
            >>> cg.add_edge('X2 lag(n=1)', 'X2', edge_type=EdgeType.DIRECTED_EDGE)
            >>> cg.add_edge('X1', 'X3', edge_type=EdgeType.DIRECTED_EDGE)
            >>>
            >>> # The time series causal graph will have the same nodes and edges as the causal graph,
            >>> # but it is aware of the time information so 'X1 lag(n=1)' and 'X1' represent the same
            >>> # variable but at different times.
            >>> ts_cg = TimeSeriesCausalGraph.from_causal_graph(cg)

        :param input_list: List of objects coercable to `cai_causal_graph.graph_components.TimeSeriesNode`. Each
            element is treated as an input node, if `full_connected` parameter is `True`. Otherwise, the nodes will
            simply be added to the graph with no edges.
        :param output_list:  List of objects coercable to `cai_causal_graph.graph_components.TimeSeriesNode`. Each
            element is treated as an output node, if `fully_connected` parameter is `True`. Otherwise, the nodes will
            simply be added to the graph with no edges.
        :param fully_connected: If set to `True`, create a fully-connected bipartite directed graph, with all
            inputs connected to all outputs. If no `input_list` and no `output_list` is provided, an empty graph will
            be created. If either or both are provided, but this is `False` (default), then the nodes will be added but
            not connected by edges.
        """
        super().__init__(input_list, output_list, fully_connected)

        # list of variables in the graph, i.e. discarding the lags (X1(t-1) and X1 are the same variable)
        self._variables: Optional[List[str]] = None
        self._summary_graph: Optional[CausalGraph] = None
        self._stationary_graph: Optional[TimeSeriesCausalGraph] = None
        self._minimal_graph: Optional[TimeSeriesCausalGraph] = None

    def __eq__(self, other: object, deep: bool = False) -> bool:
        """
        Return True if the graphs are equal.

        Two graphs are equal if they have the same nodes and edges with the same time-specific metadata.

        :param other: The other graph to compare to.
        :param deep: If `True`, also does deep equality checks on all the nodes and edges. Default is `False`.
        :return: `True` if the graphs are equal, `False` otherwise.
        """
        if not isinstance(other, TimeSeriesCausalGraph):
            return False

        # now check if the graphs are equal. Since TimeSeriesCausalGraph is a subclass of CausalGraph,
        # we can use the CausalGraph.__eq__ method and then check for the metadata
        if not super().__eq__(other, deep):
            return False

        # now check the metadata timedelta in the nodes
        for node in self.get_nodes():
            other_node = other.get_node(node.identifier)
            assert isinstance(other_node, TimeSeriesNode)  # for linting

            # check the variable name
            if node.variable_name != other_node.variable_name:
                return False

            # check the time delta
            if node.time_lag != other_node.time_lag:
                return False

        return True

    def __ne__(self, other: object) -> bool:
        """Check if the graph is not equal to another graph."""
        return not (self == other)

    def is_stationary_graph(self) -> bool:
        """
        Check if the graph is stationary. That is, if the graph is time invariant.

        If there exists the edge X(t-1) -> X(t), then there must be the same edge X(t-2) -> X(t-1), etc.

        :return: True if the graph is stationary, False otherwise.
        """
        if not self.is_dag():
            logger.warning('The graph is not a DAG. The stationarity check is not valid.')
            return False

        stationary_graph = self.get_stationary_graph()

        # now check if the stationary graph is equal to the current graph
        return stationary_graph == self

    def get_stationary_graph(self) -> TimeSeriesCausalGraph:
        """
        Make the graph stationary by adding the missing edges if needed.

        If there exists the edge X(t-1) -> X(t), then there must be the edge X(t-2) -> X(t-1), etc.
        """
        if self._stationary_graph is not None:
            return self._stationary_graph

        # extract the minimal graph
        minimal_graph = self.get_minimal_graph()

        # extract the negative and positive lag of the original graph
        lags = sorted([node.time_lag for node in self.get_nodes()])
        neg_lag, pos_lag = lags[0], lags[-1]

        # now extend the minimal graph to the current max and min lag to match the current graph
        self._stationary_graph = minimal_graph.extend_graph(-neg_lag, pos_lag)
        return self._stationary_graph

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

        :return: The minimal graph as a `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` object.
        """
        minimal_cg = self.__class__()

        for edge in self.get_edges():
            # copy edge
            edge = self._EdgeCls.from_dict(edge.to_dict(include_meta=True))

            # get the relative time delta; asserts are needed for linting
            source = edge.source
            destination = edge.destination
            assert isinstance(source, TimeSeriesNode)
            assert isinstance(destination, TimeSeriesNode)
            assert isinstance(source.time_lag, int)
            assert isinstance(destination.time_lag, int)
            assert source.variable_name is not None
            assert destination.variable_name is not None

            time_delta = destination.time_lag - source.time_lag

            # add the edge if the time delta is 0 (no need to extract the new names)
            # copy the edge type to the minimal graph
            if time_delta == 0 and not minimal_cg.edge_exists(source.variable_name, destination.variable_name):
                # create the time series nodes
                # update the node meta data to make sure the time lag is correct
                source.meta[TIME_LAG] = 0
                destination.meta[TIME_LAG] = 0

                source = self._NodeCls(
                    identifier=source.variable_name, meta=source.meta, variable_type=source.variable_type
                )

                destination = self._NodeCls(
                    identifier=destination.variable_name,
                    meta=destination.meta,
                    variable_type=destination.variable_type,
                )

                edge = self._EdgeCls(source=source, destination=destination, edge_type=edge.edge_type, meta=edge.meta)

                # Do not check for acyclicity as an acyclic minimal graph will always produce an acyclic extended graph
                minimal_cg.add_edge(edge=edge, validate=False)

            # otherwise if the time delta is not 0, we may have X[t-2]->X[t-1] and
            # we must add X[t-1]->X[t]
            else:
                # get the new names according to the time delta
                destination_name = get_name_with_lag(destination.identifier, 0)
                source_name = get_name_with_lag(source.identifier, -time_delta)
                if not minimal_cg.edge_exists(source_name, destination_name):
                    # update the node meta data to make sure the time lag is correct
                    source.meta[TIME_LAG] = -time_delta
                    destination.meta[TIME_LAG] = 0

                    # create the nodes
                    source = self._NodeCls(
                        identifier=source_name,
                        meta=source.meta,
                        variable_type=source.variable_type,
                    )

                    destination = self._NodeCls(
                        identifier=destination_name,
                        meta=destination.meta,
                        variable_type=destination.variable_type,
                    )

                    # create the edge
                    edge = self._EdgeCls(
                        source=source, destination=destination, edge_type=edge.edge_type, meta=edge.meta
                    )
                    # Do not check for acyclicity as an acyclic minimal graph will always produce an acyclic extended
                    # graph
                    minimal_cg.add_edge(edge=edge, validate=False)

        # check for floating nodes
        if self.variables is not None and len(self.variables) > 0:
            for variable in self.variables:
                if (
                    minimal_cg.variables is None or variable not in minimal_cg.variables
                ) and not minimal_cg.node_exists(variable):
                    minimal_cg.add_node(variable)

        return minimal_cg

    def _get_time_topological_order(self, ordered_nodes: List[str]) -> List[str]:
        """Return the provided list if it is ordered in time; otherwise, an empty list is returned."""
        # Iterate through the list to ensure it respects the time ordering.
        for j, node in enumerate(ordered_nodes):
            if j == 0:
                continue
            assert isinstance(node, str)  # for linting
            # check if the time delta is correct
            if self.get_node(ordered_nodes[j - 1]).time_lag > self.get_node(ordered_nodes[j]).time_lag:  # type: ignore
                return []

        return ordered_nodes

    def get_topological_order(
        self, return_all: bool = False, respect_time_ordering: bool = True
    ) -> Union[List[str], List[List[str]]]:
        """
        Return either a single or all topological orders of the graph.

        A topological order is a non-unique permutation of the nodes such that an edge from `'A'` to `'B'` implies
        that `'A'` appears before `'B'` in the topological sort order. Generating all possible topological orders may
        be expensive for large graphs.

        It is only possible to get topological order if the graph is a valid DAG.

        For more details, see `cai_causal_graph.causal_graph.CausalGraph.get_topological_order`.

        :param return_all: If `True`, return all the possible topological orders. Default is `False`.
        :param respect_time_ordering: If `True`, return the topological order that is ordered in time. Default is
            `True`. For example, if the graph is `'Y lag(n=1)' -> 'Y' <- 'X'`, then `['X', 'Y lag(n=1)', 'Y']` and
            `['Y lag(n=1)', 'X', 'Y']` are both valid topological orders. However, only the second one would respect time
            ordering. If both `return_all` and `respect_time_ordering` are `True`, then only all topological orders
            that respect time are returned, not all valid topological orders.
        :return: either a list of strings identifying a single topological order, or a list of lists identifying all
            possible topological orders.
        """
        if respect_time_ordering:
            if return_all:
                ordered_nodes_list = super().get_topological_order(return_all=return_all)
                # ordered_nodes_list must be a list of lists of strings
                assert isinstance(ordered_nodes_list, list)
                assert all(isinstance(x, list) for x in ordered_nodes_list)

                ordered_nodes: List[List[str]] = []

                for i in range(len(ordered_nodes_list)):
                    assert isinstance(ordered_nodes_list[i], list)
                    tmp = self._get_time_topological_order(ordered_nodes_list[i])  # type: ignore
                    if len(tmp) > 0:
                        ordered_nodes.append(tmp)

                return ordered_nodes
            else:
                # check it is a dag. Not needed in the other cases as it is already checked in the super method
                assert self.is_dag(), 'The graph is not a DAG. The topological order is not valid.'
                return list(
                    networkx.lexicographical_topological_sort(
                        self.to_networkx(), key=lambda x: self.get_node(x).time_lag  # type: ignore
                    )
                )

        else:
            return super().get_topological_order(return_all=return_all)

    def is_minimal_graph(self) -> bool:
        """
        Return `True` if the graph is minimal.

        See `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph.get_minimal_graph` for more details.
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

        :return: The summary graph as a `cai_causal_graph.causal_graph.CausalGraph` object.
        """
        if self._summary_graph is None:
            summary_graph = self._SummaryGraphCls()
            # now check as described above (assume edges are already directed)
            edges = self.get_edges()

            # check if the graph is a DAG
            assert self.is_dag(), 'This method only works for DAGs but the current graph is not a DAG.'

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
                    # create the nodes
                    source = self._NodeCls(
                        identifier=source_variable_name,
                        meta=source_node.meta,
                        variable_type=source_node.variable_type,
                    )
                    destination = self._NodeCls(
                        identifier=destination_variable_name,
                        meta=destination_node.meta,
                        variable_type=destination_node.variable_type,
                    )

                    # create the edge
                    edge = self._EdgeCls(
                        source=source, destination=destination, edge_type=edge.edge_type, meta=edge.meta
                    )
                    summary_graph.add_edge(edge=edge)

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
            assert backward_steps >= 0, f'backward_steps must be a non-negative integer. Got {backward_steps}.'
        if forward_steps is not None:
            assert forward_steps == int(forward_steps), f'backward_steps must be an integer. Got {forward_steps}.'
            assert forward_steps >= 0, f'forward_steps must be a non-negative integer. Got {forward_steps}.'

        # first get the minimal graph
        minimal_graph = self.get_minimal_graph()

        if minimal_graph.is_empty():
            return minimal_graph

        # create a new graph by copying the minimal graph
        extended_graph = minimal_graph.copy()
        assert isinstance(extended_graph, TimeSeriesCausalGraph)  # for linting

        if backward_steps is not None:
            # Start from 1 as 0 is already defined.
            # We cannot start directly from maxlag as it may be possible that not all the nodes from 1 to -maxlag are
            # defined (as they were not needed in the minimal graph).
            maxlag = minimal_graph.max_backward_lag
            assert maxlag is not None

            # create all the nodes from 1 to backward_steps
            for lag in range(0, backward_steps + 1):
                for node in minimal_graph.get_nodes():
                    # create the node with -lag (if it does not exist)
                    lagged_node = self._get_lagged_node(node=node, lag=-lag)
                    # if node does not exist, add it
                    if not extended_graph.node_exists(lagged_node.identifier):
                        extended_graph.add_node(node=lagged_node)

            for lag in range(1, backward_steps + 1):
                for edge in minimal_graph.get_edges():
                    assert isinstance(edge.destination, TimeSeriesNode)
                    assert isinstance(edge.source, TimeSeriesNode)
                    time_delta = edge.destination.time_lag - edge.source.time_lag

                    lagged_destination_node = self._get_lagged_node(node=edge.destination, lag=-lag)

                    # check if the new source node would go beyond the backward_steps
                    if -lag - time_delta < -backward_steps:
                        continue

                    lagged_source_node = self._get_lagged_node(node=edge.source, lag=-lag - time_delta)

                    # add the lagged edge
                    if not extended_graph.edge_exists(
                        lagged_source_node.identifier, lagged_destination_node.identifier
                    ):
                        # No need to validate as edge will be as valid as in the minimal graph
                        extended_graph.add_edge(
                            source=lagged_source_node,
                            destination=lagged_destination_node,
                            edge_type=edge.get_edge_type(),
                            meta=edge.meta,
                            validate=False,
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
            for lag in range(0, forward_steps + 1):
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

                    # add the lagged nodes
                    if not extended_graph.node_exists(lagged_source.identifier):
                        extended_graph.add_node(node=lagged_source)

                    if not extended_graph.node_exists(lagged_dest.identifier):
                        extended_graph.add_node(node=lagged_dest)
                    # No need to validate as edge will be as valid as in the minimal graph
                    extended_graph.add_edge(
                        source=extended_graph.get_node(lagged_source.identifier),
                        destination=extended_graph.get_node(lagged_dest.identifier),
                        edge_type=edge.get_edge_type(),
                        meta=edge.meta,
                        validate=False,
                    )

        return extended_graph

    @staticmethod
    def get_variable_names_from_node_names(node_names: List[str]) -> List[str]:
        """
        Return a list of variable names from a list of node names.

        This is useful for converting a list of node names into a list of variable names.
        Variables are the elementary (unique) units of a time series causal graph.

        Example:
            ['X', 'X lag(n=1)', 'Y', 'Z lag(n=2)'] -> ['X', 'Y', 'Z']

        :param node_names: A list of node names.
        :return: A sorted list of variable names.
        """
        assert isinstance(node_names, list)

        var_names = []

        for node_name in node_names:
            var_name, _ = get_variable_name_and_lag(node_name)
            if var_name not in var_names:
                var_names.append(var_name)

        return sorted(var_names)

    def _get_lagged_node(
        self, identifier: Optional[NodeLike] = None, node: Optional[TimeSeriesNode] = None, lag: Optional[int] = None
    ) -> TimeSeriesNode:
        """
        Return the lagged node of a node with a given lag. Lag is overwritten if the node is already lagged.

        For example, if the node is X and the lag is -1, the node will be X lag(n=1).
        If the node is X lag(n=1) and the lag is -2, the node will be X lag(n=2).

        Moreover, if you want to make sure to copy all the metadata from the original node, you have to provide the
        node instance and not the identifier.

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

        variable_type = None

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
            variable_type = node.variable_type

        assert node is not None, 'The node must be valid. Got None.'

        node = self._NodeCls(
            variable_name=node.variable_name,
            time_lag=lag,
            meta=meta,
            variable_type=variable_type if variable_type is not None else NodeVariableType.UNSPECIFIED,
        )
        return node

    @_reset_ts_graph_attributes
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
    ) -> TimeSeriesNode:
        """
        Add a node to the time series graph. See `cai_causal_graph.causal_graph.CausalGraph.add_node` for more details.

        In addition to the `cai_causal_graph.causal_graph.CausalGraph.add_node` method, this method also populates the
        metadata of the node with the variable name and the time lag.

        :param identifier: The identifier of the time series node.
        :param variable_name: The variable name of the time series node.
        :param time_lag: The time lag of the time series node.
        :param variable_type: The type of the variable.
        :param meta: The metadata of the time series node.
        :param node: The node to add.
        :return: The added node.
        """
        if node is not None:
            assert (
                identifier is None
                and variable_name is None
                and time_lag is None
                and variable_type == NodeVariableType.UNSPECIFIED
                and meta is None
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

        identifier = self._check_node_exists(node)
        self._nodes_by_identifier[identifier] = node

        return node

    @_reset_ts_graph_attributes
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
            new_node_id = get_name_with_lag(variable_name, time_lag)  # type: ignore
        else:
            new_node_id = Node.identifier_from(new_node_id)
            variable_name, time_lag = get_variable_name_and_lag(Node.identifier_from(new_node_id))

        if meta is not None:
            meta = meta.copy()
            meta.update({VARIABLE_NAME: variable_name, TIME_LAG: time_lag})
        else:
            meta = {VARIABLE_NAME: variable_name, TIME_LAG: time_lag}

        super().replace_node(node_id, new_node_id, variable_type=variable_type, meta=meta)

    @_reset_ts_graph_attributes
    def delete_node(self, identifier: NodeLike):
        """
        Delete a node from the graph.

        See `cai_causal_graph.causal_graph.CausalGraph.delete_node` for more details.
        """
        super().delete_node(identifier)

    @_reset_ts_graph_attributes
    def delete_edge(self, /, source: NodeLike, destination: NodeLike, *, edge_type: Optional[EdgeType] = None):
        """
        Delete an edge from the graph.

        See `cai_causal_graph.causal_graph.CausalGraph.delete_edge` for more details.
        """
        super().delete_edge(source, destination, edge_type=edge_type)

    def _check_nodes_and_edge(
        self,
        /,
        source: Optional[NodeLike] = None,
        destination: Optional[NodeLike] = None,
        *,
        edge_type: EdgeType = EdgeType.DIRECTED_EDGE,
        edge: Optional[Edge] = None,
    ):
        """Check that the source and destination nodes exist and that the edge is valid."""
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

        # For directed edge, confirm time of destination is greater than or equal to time of source.
        if edge is not None:
            source_node = edge.source
            destination_node = edge.destination
            assert isinstance(source_node, TimeSeriesNode)  # for linting
            assert isinstance(destination_node, TimeSeriesNode)  # for linting

            if edge.get_edge_type() == EdgeType.DIRECTED_EDGE:
                time_source = source_node.time_lag
                time_destination = destination_node.time_lag
                if time_destination < time_source:
                    raise ValueError(
                        f'For a directed edge, the time at the destination must be greater than or equal to the time at '
                        f'the source. The time lag for the source and destination are {time_source} and '
                        f'{time_destination}, respectively.'
                    )
        elif edge_type == EdgeType.DIRECTED_EDGE:
            # Check these are not None before trying to get node.
            assert source is not None
            assert destination is not None
            source_node = self.get_node(source)
            destination_node = self.get_node(destination)
            assert isinstance(source_node, TimeSeriesNode)  # for linting
            assert isinstance(destination_node, TimeSeriesNode)  # for linting
            time_source = source_node.time_lag
            time_destination = destination_node.time_lag
            if time_destination < time_source:
                raise ValueError(
                    f'For a directed edge, the time at the destination must be greater than or equal to the time at '
                    f'the source. The time lag for the source and destination are {time_source} and '
                    f'{time_destination}, respectively.'
                )
        # No else needed as we don't need to check time direction for other edge types.

    @_reset_ts_graph_attributes
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
        In addition to the `cai_causal_graph.causal_graph.CausalGraph.add_edge` method, this method also populates the
        metadata of the nodes with the variable name and the time lag.

        If these two nodes are already connected in any way, then an error will be raised. An error will also be raised
        if the new edge would create a cyclic connection of directed edges. In this case, the
        `cai_causal_graph.causal_graph.TimeSeriesCausalGraph` instance will be restored to its original state and the
        edge will not be added. It is possible to specify an edge type from source to destination as well,
        with the default being a forward directed edge, i.e., source -> destination.

        :param source: String identifying the node from which the edge will originate. Can be `None`, if an `edge`
            parameter is specified.
        :param destination: String identifying the node at which the edge will terminate. Can be `None`, if an `edge`
            parameter is specified.
        :param edge_type: The type of the edge to be added. Default is
            `cai_causal_graph.type_definitions.EdgeType.DIRECTED_EDGE`. See `cai_causal_graph.type_definitions.EdgeType`
            for the list of possible edge types.
        :param meta: The meta values for the edge.
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
            assert isinstance(edge, self._EdgeCls), f'Edge class must be of type {self._EdgeCls.__name__}.'
            # Check source node type
            source_node = deepcopy(edge.source)
            if isinstance(source_node, Node) and not isinstance(source_node, self._NodeCls):
                source_node = self._NodeCls.from_dict(source_node.to_dict())
            # Check destination node type
            destination_node = deepcopy(edge.destination)
            if isinstance(destination_node, Node) and not isinstance(destination_node, self._NodeCls):
                destination_node = self._NodeCls.from_dict(destination_node.to_dict())
            # See if either node was recreated, if yes, then create new edge with appropriate nodes
            if source_node != edge.source or destination_node != edge.destination:
                if edge_type != EdgeType.DIRECTED_EDGE:
                    # if time_lag for source is greater than destination, then swap source and destination
                    assert isinstance(source, TimeSeriesNode) and isinstance(destination, TimeSeriesNode)
                    if source_node.time_lag > destination_node.time_lag:
                        destination_node, source_node = source_node, destination_node
                edge = self._EdgeCls(
                    source_node, destination_node, edge_type=edge.get_edge_type(), meta=edge.get_metadata()
                )

        else:
            # we need the nodes to be in the correct format to extract the time_lag. We only need to do this if the edge
            # is non-directed.
            if edge_type != EdgeType.DIRECTED_EDGE:
                if isinstance(source, Node) and not isinstance(source, self._NodeCls):
                    tmp_source = self._NodeCls.from_dict(source.to_dict())
                    source_time_lag = tmp_source.time_lag
                elif isinstance(source, str):
                    source_time_lag = get_variable_name_and_lag(source)[1]
                else:
                    assert isinstance(source, TimeSeriesNode)  # for linting
                    source_time_lag = source.time_lag

                if isinstance(destination, Node) and not isinstance(destination, self._NodeCls):
                    tmp_destination = self._NodeCls.from_dict(destination.to_dict())
                    destination_time_lag = tmp_destination.time_lag
                elif isinstance(source, str):
                    assert destination is not None  # for linting
                    destination_time_lag = get_variable_name_and_lag(destination)[1]
                else:
                    assert isinstance(destination, TimeSeriesNode)  # for linting
                    destination_time_lag = destination.time_lag

                if source_time_lag > destination_time_lag:
                    source, destination = destination, source

        self._check_nodes_and_edge(source=source, destination=destination, edge_type=edge_type, edge=edge)
        edge = super().add_edge(
            source=source,
            destination=destination,
            edge_type=edge_type,
            meta=meta,
            edge=edge,
            validate=validate,
        )

        return edge

    @_reset_ts_graph_attributes
    def add_time_edge(
        self,
        /,
        source_variable: str,
        source_time: int,
        destination_variable: str,
        destination_time: int,
        *,
        meta: Optional[dict] = None,
        validate: bool = True,
    ) -> Edge:
        """
        Add a time edge to the graph from the variable at the source time to the variable at the destination time.

        :param source_variable: The name of the source variable.
        :param source_time: The time of the source variable.
        :param destination_variable: The name of the destination variable.
        :param destination_time: The time of the destination variable.
        :param meta: The metadata for the edge.
        :param validate: Whether to perform validation checks. The validation checks will raise if
            any cycles are introduced to the graph by adding the edge. There is no guarantees about the behavior of the
            resulting graph if this is disabled specifically to introduce cycles. This should only be used to speed up
            this method in situations where it is known the new edge will not add cycles, for example when copying a
            graph. Default is `True`.
        :return: The edge that was added.

        Example:
            - `add_time_edge('x', -2, 'x', 2)` will add an edge from the variable `x` at time -2 to the variable `x` at
                time 2.
        """

        assert source_variable is not None
        assert source_time is not None
        assert destination_variable is not None
        assert destination_time is not None

        source = get_name_with_lag(source_variable, source_time)
        destination = get_name_with_lag(destination_variable, destination_time)

        return self.add_edge(source, destination, meta=meta, validate=validate)

    @classmethod
    def from_causal_graph(cls, causal_graph: CausalGraph) -> TimeSeriesCausalGraph:
        """
        Instantiate a `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` from a
        `cai_causal_graph.causal_graph.CausalGraph` object. If the `cai_causal_graph.causal_graph.CausalGraph` is
        already a `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph`, it is returned as is.

        This is useful for converting a `cai_causal_graph.causal_graph.CausalGraph` from a single time step into a
        `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph`.

        :param causal_graph: The causal graph as a `cai_causal_graph.causal_graph.CausalGraph` object.
        :return: A `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` object.
        """
        if isinstance(causal_graph, cls):
            return causal_graph

        sepsets = deepcopy(causal_graph._sepsets)

        # copy nodes and make them TimeSeriesNodes
        ts_cg = cls()
        for node in causal_graph.get_nodes():
            meta = deepcopy(node.meta)
            # get the variable name and lag from the node name
            variable_name, lag = get_variable_name_and_lag(node.identifier)
            node = TimeSeriesNode(variable_name=variable_name, time_lag=lag, meta=meta)
            ts_cg.add_node(node=node)

        # copy edges
        for edge in causal_graph.get_edges():
            source = ts_cg.get_node(edge.source)
            destination = ts_cg.get_node(edge.destination)
            assert isinstance(source, TimeSeriesNode)  # for linting
            assert isinstance(destination, TimeSeriesNode)  # for linting

            # No need to validate, the new graph will be as valid as the input graph
            ts_cg.add_edge(source, destination, meta=edge.meta, edge_type=edge.get_edge_type(), validate=False)

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
            coerced to `cai_causal_graph.graph_components.TimeSeriesNode`.
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
        Return a time series causal graph from a dictionary of adjacency matrices. Keys are the time deltas.
        This is useful for converting a list of adjacency matrices into a time series causal graph.

        For example, the adjacency matrix with time delta -1 is stored in adjacency_matrices[-1] and would correspond
        to X-1 -> X, where X is the set of nodes.

        Moreover, if floating nodes are present, only the contemporaneous nodes will be added to the graph.

        Example:
        >>> import numpy
        >>> from cai_causal_graph import TimeSeriesCausalGraph
        >>>
        >>> # define the adjacency matrices
        >>> adjacency_matrices = {
        >>>     -2: numpy.array([[0, 0, 0], [1, 0, 0], [0, 0, 1]]),
        >>>     -1: numpy.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
        >>>     0: numpy.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]]),
        >>> }
        >>>
        >>> # construct a time series causal graph from the adjacency matrices
        >>> graph = TimeSeriesCausalGraph.from_adjacency_matrices(adjacency_matrices, variable_names=['X', 'Y', 'Z'])
        >>>
        >>> # query the edges of the constructed time series causal graph
        >>> graph.get_edges()
        >>> # output:
        >>> # [
        >>> #   Edge("X", "Y", type=->),
        >>> #   Edge("X", "Z", type=->),
        >>> #   Edge("X lag(n=1)", "Y", type=->),
        >>> #   Edge("Y", "Z", type=->),
        >>> #   Edge("Y lag(n=1)", "X", type=->),
        >>> #   Edge("Y lag(n=2)", "X", type=->),
        >>> #   Edge("Z lag(n=2)", "Z", type=->)
        >>> # ]

        :param adjacency_matrices: A dictionary of adjacency matrices. Keys are the time delta.
        :param variable_names: A list of variable names. If not provided, the variable names are integers starting
            from 0. Node names must correspond to the variable names and must not contain the lag.
        :return: A time series causal graph.
        """
        assert isinstance(adjacency_matrices, dict)
        # Keys must be integers or str that can be converted to integers as they represent the time deltas.
        assert all(isinstance(key, (int, str)) and int(key) == key for key in adjacency_matrices)
        assert all(isinstance(adj, numpy.ndarray) for adj in adjacency_matrices.values())

        # Confirm shape of all adjacency matrices are the same.
        shapes = [adj.shape for adj in adjacency_matrices.values()]
        assert (
            len(set(shapes)) == 1
        ), f'The shape of all the adjacency matrices must be the same. Got the following shapes: {list(set(shapes))}.'
        shape = shapes[0]

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

        # first add all the contemporaneous nodes (there could be floating nodes)
        for variable_name in variable_names_str:
            tsgraph.add_node(variable_name=variable_name, time_lag=0)

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
            tsgraph.add_edges_from(edges)  # type: ignore

        return tsgraph

    @staticmethod
    def from_skeleton(skeleton: Skeleton) -> TimeSeriesCausalGraph:
        """
        Construct a `cai_causal_graph.causal_graph.TimeSeriesCausalGraph` instance from a
        `cai_causal_graph.causal_graph.Skeleton` instance.
        """
        return TimeSeriesCausalGraph.from_causal_graph(CausalGraph.from_skeleton(skeleton))

    @staticmethod
    def from_networkx(g: networkx.Graph) -> TimeSeriesCausalGraph:
        """
        Construct a `cai_causal_graph.causal_graph.TimeSeriesCausalGraph` instance from a
        `networkx.Graph` instance.
        """
        return TimeSeriesCausalGraph.from_causal_graph(CausalGraph.from_networkx(g))

    @staticmethod
    def from_gml_string(gml: str) -> TimeSeriesCausalGraph:
        """
        Return an instance of `cai_causal_graph.causal_graph.TimeSeriesCausalGraph` constructed from the provided Graph Modelling
        Language (GML) string.
        """
        g = networkx.parse_gml(gml)
        return TimeSeriesCausalGraph.from_networkx(g)

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

            if edge.get_edge_type() == EdgeType.DIRECTED_EDGE:
                adjacency_matrices[source_lag][
                    self.variables.index(source_variable_name), self.variables.index(destination_variable_name)
                ] = 1
            elif edge.get_edge_type() == EdgeType.UNDIRECTED_EDGE:
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
        Return the absolute maximum backward time lag of the graph. Retained for backwards compatibility.

        For example, if the graph is X lag(n=2) -> Y future(n=1), the maximum backward lag is 2.

        If the graph is empty or only forward lags are present, None is returned. Otherwise, the maximum backward lag
        is a non-negative integer where 0 is returned if only contemporaneous nodes are present.
        """
        # get the maximum lag of the nodes in the graph
        return self.max_backward_lag

    @property
    def max_forward_lag(self) -> Optional[int]:
        """
        Return the maximum forward time lag of the graph.

        For example, if the graph is X lag(n=2) -> Y future(n=1), the maximum forward lag is 1.

        If the graph is empty or only backward lags are present, None is returned. Otherwise, the maximum forward lag
        is a non-negative integer where 0 is returned if only contemporaneous nodes are present.
        """
        if len(self.nodes) == 0:
            return None
        # get the maximum lag of the nodes in the graph
        pos_time_lags = [node.time_lag for node in self.get_nodes() if node.time_lag >= 0]
        max_forward_lag = max(pos_time_lags) if len(pos_time_lags) > 0 else None
        return max_forward_lag

    @property
    def max_backward_lag(self) -> Optional[int]:
        """
        Return the absolute maximum backward time lag of the graph.

        For example, if the graph is X lag(n=2) -> Y future(n=1), the maximum backward lag is 2.

        If the graph is empty or only forward lags are present, None is returned. Otherwise, the maximum backward lag
        is a non-negative integer where 0 is returned if only contemporaneous nodes are present.
        """
        if len(self.nodes) == 0:
            return None
        neg_time_lags = [node.time_lag for node in self.get_nodes() if node.time_lag <= 0]
        max_backward_lag = abs(min(neg_time_lags)) if len(neg_time_lags) > 0 else None
        return max_backward_lag

    @property
    def variables(self) -> Optional[List[str]]:
        """
        Return the variables in the graph.

        Variables differ from nodes in that they do not contain the lag. For example, if the graph contains the node
        'X1 lag(n=2)', the variable is 'X1'.
        """
        if self._variables is None:
            variables = []
            for node in self.get_nodes():
                assert node.variable_name is not None
                variables.append(node.variable_name)
            self._variables = sorted(list(set(variables)))
        return self._variables

    def get_nodes(  # type: ignore
        self, identifier: Optional[Union[NodeLike, List[NodeLike]]] = None
    ) -> List[TimeSeriesNode]:  # type: ignore
        """
        Return the time series nodes in the graph.

        :param identifier: The identifier of the node(s) to return.
        :return: A list of nodes.
        """
        nodes = super().get_nodes(identifier)
        # check all nodes are TimeSeriesNode
        assert all(isinstance(node, TimeSeriesNode) for node in nodes)
        return nodes  # type: ignore

    def to_numpy_by_lag(self) -> Tuple[Dict[int, numpy.ndarray], List[str]]:
        """
        Return the adjacency matrices of the time series causal graph ordered by the time delta.

        Different time deltas are represented by different adjacency matrices.
        The keys of the dictionary are the time deltas and the values are the adjacency matrices.

        :return: A tuple containing the dictionary for the adjacency matrices and the variable names.
        """
        # get the adjacency matrix
        adjacency_matrices = self.adjacency_matrices

        assert self.variables is not None
        return adjacency_matrices, self.variables

    def get_nodes_at_lag(self, time_lag: int = 0) -> List[TimeSeriesNode]:
        """Return all nodes at time delta `time_lag`."""
        return [node for node in self.get_nodes() if node.time_lag == time_lag]

    def get_contemporaneous_nodes(self, node: NodeLike) -> List[TimeSeriesNode]:
        """Return all nodes that are contemporaneous (i.e. have the same time_lag) to the provided node."""
        assert node is not None, 'The `node` cannot be None.'
        if isinstance(node, str):
            node = self.get_node(node)

        assert isinstance(node, TimeSeriesNode), 'The node must be a `TimeSeriesNode`.'
        cont_nodes = self.get_nodes_at_lag(node.time_lag)
        return [n for n in cont_nodes if n != node]

    def __hash__(self) -> int:
        """
        Return a hash representation of the `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` instance.
        """
        return super().__hash__()
