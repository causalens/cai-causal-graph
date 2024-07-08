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
from collections import defaultdict
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Type, Union, cast

import networkx
import numpy
from mypy_extensions import Arg, DefaultArg

from cai_causal_graph.causal_graph import CausalGraph, Skeleton, reset_cached_attributes_decorator
from cai_causal_graph.graph_components import Edge, TimeSeriesEdge, TimeSeriesNode
from cai_causal_graph.interfaces import HasIdentifier, HasMetadata
from cai_causal_graph.type_definitions import TIME_LAG, EdgeType, NodeLike, NodeVariableType
from cai_causal_graph.utils import get_name_with_lag, get_variable_name_and_lag

logger = logging.getLogger(__name__)


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
    _EdgeCls: Type[Edge] = TimeSeriesEdge
    _SummaryGraphCls: Type[CausalGraph] = CausalGraph
    _lag_to_nodes: Dict[int, List[TimeSeriesNode]]
    _variable_name_to_nodes: Dict[str, List[TimeSeriesNode]]

    # Overwrite type annotations for linting and code completion.
    from_adjacency_matrix: Callable[  # type: ignore
        [
            Arg(numpy.ndarray, 'adjacency'),
            DefaultArg(Optional[List[Union[NodeLike, int]]], 'node_names'),
        ],
        TimeSeriesCausalGraph,
    ]
    from_skeleton: Callable[[Arg(Skeleton, 'skeleton')], TimeSeriesCausalGraph]  # type: ignore
    from_networkx: Callable[[Arg(networkx.Graph, 'g')], TimeSeriesCausalGraph]  # type: ignore
    from_gml_string: Callable[[Arg(str, 'gml')], TimeSeriesCausalGraph]  # type: ignore

    nodes: List[TimeSeriesNode]   # type: ignore
    edges: List[TimeSeriesEdge]   # type: ignore
    get_nodes: Callable[  # type: ignore
        [DefaultArg(Optional[Union[NodeLike, List[NodeLike]]], 'identifier')], List[TimeSeriesNode]
    ]
    get_node: Callable[[Arg(Optional[Union[NodeLike, List[NodeLike]]], 'identifier')], TimeSeriesNode]  # type: ignore

    def __init__(
        self,
        input_list: Optional[List[NodeLike]] = None,
        output_list: Optional[List[NodeLike]] = None,
        fully_connected: bool = False,
        meta: Optional[dict] = None,
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
        :param meta: Any metadata defined on the graph. The keys must be strings, but no requirement is placed on the
            values of the dictionary. Default is `None`. If passed, meta is shallow-copied.
        """
        # Initialize caches for fast lookups
        self._lag_to_nodes = defaultdict(list)
        self._variable_name_to_nodes = defaultdict(list)
        super().__init__(input_list=input_list, output_list=output_list, fully_connected=fully_connected, meta=meta)

        # list of variables in the graph, i.e. discarding the lags (X1(t-1) and X1 are the same variable)
        self._variables: Optional[List[str]] = None
        self._is_minimal_graph: Optional[bool] = None
        self._is_stationary_graph: Optional[bool] = None

    def is_stationary_graph(self) -> bool:
        """
        Check if the graph is stationary. That is, if the graph is time invariant.

        If there exists the edge `X(t-1) -> X(t)`, then there must be the same edge `X(t-2) -> X(t-1)`, etc.

        :return: True if the graph is stationary, False otherwise.
        """
        if not self.is_dag():
            logger.warning('The graph is not a DAG. The stationarity check is not valid.')
            return False

        if self._is_stationary_graph is None:
            stationary_graph = self.get_stationary_graph()

            # now check if the stationary graph is equal to the current graph
            self._is_stationary_graph = stationary_graph == self

        return self._is_stationary_graph

    def get_stationary_graph(self) -> TimeSeriesCausalGraph:
        """
        Make the graph stationary by adding the missing edges if needed.

        If there exists the edge `X(t-1) -> X(t)`, then there must be the edge `X(t-2) -> X(t-1)`, etc.
        """
        # extract the minimal graph
        minimal_graph = self.get_minimal_graph()

        # extract the negative and positive lag of the original graph
        lags = sorted([node.time_lag for node in self.get_nodes()])
        neg_lag, pos_lag = lags[0], lags[-1]

        # now extend the minimal graph to the current max and min lag to match the current graph
        return minimal_graph.extend_graph(-neg_lag, pos_lag, include_all_parents=False)

    def get_minimal_graph(self) -> TimeSeriesCausalGraph:
        """
        Return a minimal time series causal graph.

        The minimal graph is the graph with the minimal number of edges that is equivalent to the original graph.
        In other words, it is a graph that has no edges whose destination is not time delta 0.

        Example:
        Input graph:
            - X1(t-2)-> X1(t-1) -> X2(t-1) -> X2(t), X1(t) -> X2(t), X1(t-1) -> X2(t-1)
        Minimal graph:
            - X1(t-1) -> X1(t) -> X2(t)

        :return: The minimal graph as a `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` object.
        """
        minimal_cg = self.__class__(meta=deepcopy(self.meta))

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
                # update the node metadata to make sure the time lag is correct
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
                    # update the node metadata to make sure the time lag is correct
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
                    # Guaranteed there is at least one node, take the first for simplicity.
                    # This aligns with how edges are chosen for minimal graph.
                    # Only issue is if they have different meta. TODO: CAUSALAI-4784
                    original_floating_node = self.get_nodes_for_variable_name(variable)[0]
                    new_floating_node = self._NodeCls(
                        identifier=variable,
                        meta=original_floating_node.meta,
                        variable_type=original_floating_node.variable_type,
                    )
                    minimal_cg.add_node(node=new_floating_node)

        return minimal_cg

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
        if self._is_minimal_graph is None:
            self._is_minimal_graph = self == self.get_minimal_graph()

        return self._is_minimal_graph

    def get_summary_graph(self) -> CausalGraph:
        """
        Return a summary graph.

        Collapse graph in time into a single node per variable (column name).
        This can become cyclic and bi-directed as `X(t-1) -> Y` and `Y(t-1) -> X` would become `X <-> Y`.

        There are several cases to consider. Assume the edge in consideration is called B.
        - if the edge is not in the summary graph, add it
        - if there is an edge A is in the summary graph, then:
            - if A and B have the same direction, keep the direction
            - if A and B have different directions, make it bi-directed
            - if one of the two is already bi-directed, keep it bi-directed

        :return: The summary graph as a `cai_causal_graph.causal_graph.CausalGraph` object.
        """
        # check if the graph is a DAG
        assert self.is_dag(), 'This method only works for DAGs but the current graph is not a DAG.'

        summary_graph = self._SummaryGraphCls(meta=deepcopy(self.meta))
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
                edge = self._EdgeCls(source=source, destination=destination, edge_type=edge.edge_type, meta=edge.meta)
                summary_graph.add_edge(edge=edge)

        return summary_graph

    def extend_graph(
        self,
        backward_steps: Optional[int] = None,
        forward_steps: Optional[int] = None,
        include_all_parents: bool = True,
    ) -> TimeSeriesCausalGraph:
        """
        Return an extended graph.

        Extend the graph in time by adding nodes for each variable at each time step from `backward_steps` to
        `forward_steps`. If a backward step of n is specified, it means that the graph will be extended in order to
        include nodes back to time -n. For example, if t is the reference time, if `backward_steps` is 2, the graph
        will be extended to include nodes back to time t-2. This does not mean that the graph will be extended to only
        include nodes at time t-2, but rather that it will include nodes at time t-2 and all the nodes that are
        connected to them as specified by the minimal graph.

        By default, this will also add all parents of newly added nodes. This is done to ensure that all nodes
        corresponding to the same variable name have consistent parents, which is particularly useful for
        causal modeling tasks. This can be controlled by the `include_all_parents` parameter. If set to `False`, then
        only nodes up-to `backward_steps` back in time are added.

        If both `backward_steps` and `forward_steps` are None, the original graph is returned.

        :param backward_steps: Number of steps to extend the graph backwards in time. If None, do not extend backwards.
        :param forward_steps: Number of steps to extend the graph forwards in time. If None, do not extend forwards.
        :param include_all_parents: If `True`, any nodes and edges required to predict nodes up to `backward_steps` ago
            will also be added, in addition to the nodes/edges normally added by this method. This may mean that nodes
            further back in time than `backward_steps` will be added, if they are parents of any nodes up to
            `backward_steps` ago. Default is `True`, meaning this extra nodes/edges are added.

            `include_all_parents` is only valid when specifying a `backward_steps`, and will have no effect on the
            logic of `forward_steps` since by definition all the parents of the added nodes in the future will either
            already exist in the minimal graph or be added.
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

                    # check if the new source node would go beyond the backward_steps, and do not add if
                    # include_all_parents is False
                    if (-lag - time_delta < -backward_steps) and not include_all_parents:
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

    def get_all_variable_names(self) -> List[str]:
        """
        Return all variable names in the provided causal graph.

        The returned list is sorted by variable name.
        """
        return self.get_variable_names_from_node_names(node_names=self.get_node_names())

    @reset_cached_attributes_decorator
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
        if node is None:
            try:
                node = self._NodeCls(
                    identifier=identifier,
                    variable_name=variable_name,
                    time_lag=time_lag,
                    variable_type=variable_type,
                    meta=meta,
                )
            except Exception as e:
                # Raise matching error type, and point to the original error
                if isinstance(e, (ValueError, AssertionError)):
                    raise e.__class__(f'Cannot add a node using the specified parameters.') from e
                else:
                    # In case more complex errors are raised, raise them directly.
                    raise e

            # Add explicitly rather than a node to ensure metadata is not deepcopied
            node = cast(
                TimeSeriesNode,
                super().add_node(identifier=node.identifier, meta=node.meta, variable_type=node.variable_type),
            )
        else:
            assert (
                identifier is None
                and variable_name is None
                and time_lag is None
                and meta is None
                and variable_type is NodeVariableType.UNSPECIFIED
            ), 'If specifying `node` argument, all other arguments should not be specified.'

            node = cast(
                TimeSeriesNode,
                super().add_node(node=node),
            )

        self._add_node_to_cache(node)

        return node

    @reset_cached_attributes_decorator
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
        # either new_node_id or (time_lag and variable_name) must be provided, but not both
        if new_node_id is not None:
            # If new node id is provided, then it must set variable name and time lag, hence these must be None
            assert (
                time_lag is None and variable_name is None
            ), f'Cannot provide both a new_node_id and (time_lag and variable_name).'
        elif time_lag is not None or variable_name is not None:
            # if new_node_id is not provided, but variable name or time lag is, then construct a new node id
            # if one of variable name or time lag is not provided, infer it from existing node id. This enables
            # changing the node variable name/time lag
            derived_variable_name, derived_time_lag = get_variable_name_and_lag(node_id)
            if time_lag is None:
                time_lag = derived_time_lag

            if variable_name is None:
                variable_name = derived_variable_name

            new_node_id = get_name_with_lag(variable_or_node_name=variable_name, lag=time_lag)

        super().replace_node(node_id=node_id, new_node_id=new_node_id, variable_type=variable_type, meta=meta)

    @reset_cached_attributes_decorator
    def delete_node(self, identifier: NodeLike):
        """
        Delete a node from the graph.

        See `cai_causal_graph.causal_graph.CausalGraph.delete_node` for more details.
        """
        node = self.get_node(self._NodeCls.identifier_from(identifier))
        assert isinstance(node, TimeSeriesNode)  # for linting
        self._remove_node_from_cache(node)
        super().delete_node(identifier)

    @reset_cached_attributes_decorator
    def delete_edge(self, /, source: NodeLike, destination: NodeLike, *, edge_type: Optional[EdgeType] = None):
        """
        Delete an edge from the graph.

        See `cai_causal_graph.causal_graph.CausalGraph.delete_edge` for more details.
        """
        super().delete_edge(source=source, destination=destination, edge_type=edge_type)

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

    @reset_cached_attributes_decorator
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

        assert (
            source_variable is not None
            and source_time is not None
            and destination_variable is not None
            and destination_time is not None
        ), f'When adding an edge source and destination variable and time lags must be specified.'

        source = get_name_with_lag(source_variable, source_time)
        destination = get_name_with_lag(destination_variable, destination_time)

        return self.add_edge(source=source, destination=destination, meta=meta, validate=validate)

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

        # This also deepcopies all the metadata
        ts_cg = cast(TimeSeriesCausalGraph, cls.from_dict(d=causal_graph.to_dict(include_meta=True), validate=False))
        ts_cg._sepsets = sepsets

        return ts_cg

    @classmethod
    def from_adjacency_matrices(
        cls,
        adjacency_matrices: Dict[int, numpy.ndarray],
        variable_names: Optional[List[Union[NodeLike, int]]] = None,
        construct_minimal: bool = True,
    ) -> TimeSeriesCausalGraph:
        """
        Instantiate a `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` from a dictionary of
        adjacency matrices.

        Keys are the time deltas. This is useful for converting a list of adjacency matrices into a
        `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph`.

        For example, the adjacency matrix with time delta -1 is stored in adjacency_matrices[-1] and would correspond
        to `X(t-1) -> X`, where `X` is the set of nodes.

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
        :param construct_minimal: Whether to return a minimal time series graph. Default is `True`.
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

        # if 0 in adjacency_matrices keys, then we need to add the contemporaneous nodes
        if 0 not in adjacency_matrices:
            adjacency_matrices = adjacency_matrices.copy()
            adjacency_matrices[0] = numpy.zeros((shape[0], shape[0]))

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

        # create the full matrix from the dictionary of adjacency matrices
        # get the nodes first the full adjacency matrix has the shape NT x NT, where N is the number of nodes and T
        # is the number of time steps.
        adjacency_matrix_full = numpy.zeros((shape[0] * len(adjacency_matrices), shape[0] * len(adjacency_matrices)))

        # create the node names list to match the variable names
        node_names = []
        for variable_name in variable_names_str:
            for time_delta in adjacency_matrices:
                node_names.append(get_name_with_lag(str(variable_name), time_delta))

        # create a map between time delta and the index in the full adjacency matrix
        time_delta_to_index = {time_delta: i for i, time_delta in enumerate(adjacency_matrices)}
        n_time_delta = len(adjacency_matrices)

        for time_delta, adjacency_matrix in adjacency_matrices.items():
            for row, column in zip(*numpy.where(adjacency_matrix)):
                # add 1 to the row and column to account for the time delta
                adjacency_matrix_full[
                    time_delta_to_index[time_delta] + (n_time_delta * row),
                    time_delta_to_index[0] + (n_time_delta * column),
                ] = 1

        graph = cls.from_adjacency_matrix(adjacency_matrix_full, node_names)  # type: ignore

        if construct_minimal:
            graph = graph.get_minimal_graph()

        return graph

    @property
    def adjacency_matrices(self) -> Dict[int, numpy.ndarray]:
        """
        Return the adjacency matrix dictionary of the minimal time series causal graph.

        The keys are the time deltas and the values are the adjacency matrices.
        """
        adjacency_matrices: Dict[int, numpy.ndarray] = {}

        # get the minimal graph
        graph = self.get_minimal_graph()

        if graph.variables is None:
            return adjacency_matrices

        for edge in graph.edges:
            # get the source and destination of the edge
            # extract the variable name and lag from the node attributes
            source_variable_name, source_lag = (
                edge.source.variable_name,
                edge.source.time_lag,
            )
            if source_lag not in adjacency_matrices:
                adjacency_matrices[source_lag] = numpy.zeros((len(graph.variables), len(graph.variables)))

            destination_variable_name, _ = (
                edge.destination.variable_name,
                edge.destination.time_lag,
            )

            if edge.get_edge_type() == EdgeType.DIRECTED_EDGE:
                adjacency_matrices[source_lag][
                    graph.variables.index(source_variable_name), graph.variables.index(destination_variable_name)
                ] = 1
            elif edge.get_edge_type() == EdgeType.UNDIRECTED_EDGE:
                adjacency_matrices[source_lag][
                    graph.variables.index(source_variable_name), graph.variables.index(destination_variable_name)
                ] = 1
                adjacency_matrices[source_lag][
                    graph.variables.index(destination_variable_name), graph.variables.index(source_variable_name)
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
        return self._variables.copy()

    def to_numpy_by_lag(self) -> Tuple[Dict[int, numpy.ndarray], List[str]]:
        """
        Return the adjacency matrices of the minimal time series causal graph ordered by the time delta.

        Different time deltas are represented by different adjacency matrices.
        The keys of the dictionary are the time deltas and the values are the adjacency matrices.

        :return: A tuple containing the dictionary for the adjacency matrices and the variable names.
        """
        # get the adjacency matrix
        adjacency_matrices = self.adjacency_matrices

        # Adjacency matrices
        graph = self.get_minimal_graph()

        assert graph.variables is not None
        return adjacency_matrices, graph.variables

    def get_nodes_at_lag(self, time_lag: int = 0) -> List[TimeSeriesNode]:
        """
        Return all nodes at time delta `time_lag`.

        :param time_lag: Time lag to return nodes for. Default is `0`.
        """
        return self._lag_to_nodes[time_lag]

    def get_nodes_for_variable_name(self, variable_name: str) -> List[TimeSeriesNode]:
        """
        Return all nodes for the variable `variable_name`.

        :param variable_name: Variable name to return nodes for.
        """
        return self._variable_name_to_nodes[variable_name]

    def get_contemporaneous_nodes(self, node: NodeLike) -> List[TimeSeriesNode]:
        """Return all nodes that are contemporaneous (i.e. have the same time_lag) to the provided node."""
        assert node is not None, 'The `node` cannot be None.'
        if isinstance(node, str):
            node = self.get_node(node)

        assert isinstance(node, TimeSeriesNode), 'The node must be a `TimeSeriesNode`.'
        cont_nodes = self.get_nodes_at_lag(node.time_lag)
        return [n for n in cont_nodes if n != node]

    def _add_node_to_cache(self, node: TimeSeriesNode):
        """Add a node to node caches."""
        self._lag_to_nodes[node.time_lag].append(node)
        self._variable_name_to_nodes[node.variable_name].append(node)

    def _remove_node_from_cache(self, node: TimeSeriesNode):
        """Remove a node to node caches."""
        try:
            self._lag_to_nodes[node.time_lag].remove(node)
            if not self._lag_to_nodes[node.time_lag]:
                del self._lag_to_nodes[node.time_lag]
        except ValueError as e:
            raise ValueError(
                f'Tried to remove node {node.identifier} from `TimeSeriesCausalGraph._lag_to_nodes` cache but the '
                f'node was not found!'
            ) from e

        try:
            self._variable_name_to_nodes[node.variable_name].remove(node)
            if not self._variable_name_to_nodes[node.variable_name]:
                del self._variable_name_to_nodes[node.variable_name]
        except ValueError as e:
            raise ValueError(
                f'Tried to remove node {node.identifier} from `TimeSeriesCausalGraph.variable_name_to_nodes` cache '
                f'but the node was not found!'
            ) from e

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

    def _get_lagged_node(self, node: TimeSeriesNode, lag: int = 0) -> TimeSeriesNode:
        """
        Return a version of a node with a given lag.

        For example, if the node is X and the lag is -1, the node will be X lag(n=1).
        If the node is X lag(n=1) and the lag is -2, the node will be X lag(n=2).

        Metadata of the provided node will be shallow-copied to the newly constructed node. It is not deepcopied here
        because metadata will be deepcopied when adding a node as a node of a causal graph.

        Even if the provided lag matches existing lag of a node, a new instance of a node is constructed and returned.

        :param node: Original node used to construct a lagged copy.
        :param lag: The lag of the node. Default is zero.
        :return: The lagged node.
        """
        return self._NodeCls(
            variable_name=node.variable_name,
            time_lag=lag,
            meta=node.meta,  # meta is shallow copied by the node constructor
            variable_type=node.variable_type,
        )

    def _reset_cached_attributes(self):
        """Reset cached internal attributes."""
        self._variables = None
        self._is_minimal_graph = None
        self._is_stationary_graph = None
        super()._reset_cached_attributes()
