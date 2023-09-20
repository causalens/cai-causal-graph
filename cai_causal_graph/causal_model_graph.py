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
import copy
from typing import Any, Dict, List, Optional, Union

from cai_causal_graph import CausalGraph
from cai_causal_graph.causal_graph import Edge, Node
from cai_causal_graph.definitions import EdgeActivation, NodeAggregation
from cai_causal_graph.exceptions import CausalGraphErrors
from cai_causal_graph.interfaces import HasMetadata
from cai_causal_graph.type_definitions import PAIR_T, EdgeType, NodeLike, NodeVariableType
from cai_causal_graph.utils import to_list

_AGG_DICT_T = Dict[str, Union[bool, int, float, str, list, dict]]
_ACT_DICT_T = Dict[str, Union[bool, int, float, str, list, dict]]
_Aggregation = Union[str, _AGG_DICT_T]
_Activation = Union[str, _ACT_DICT_T, List[Union[str, _ACT_DICT_T]]]

DEFAULT_NODE_AGGREGATION = NodeAggregation.SUM_WITH_BIAS
DEFAULT_NODE_ACTIVATION = EdgeActivation.IDENTITY
DEFAULT_EDGE_ACTIVATION = EdgeActivation.LINEAR


class CausalModelNode(Node):
    """A utility class that manages the state of a model node."""

    def __init__(
        self,
        identifier: Optional[NodeLike] = None,
        meta: Optional[Dict[str, Any]] = None,
        aggregation: Optional[_Aggregation] = None,
        activations: Optional[Union[_Activation, List[_Activation]]] = None,
        variable_type: Union[str, NodeVariableType] = NodeVariableType.UNSPECIFIED,
    ):
        """
        :param identifier: String that uniquely identifies the node within the causal graph.
        :param meta: The meta values for the node.
        :param aggregation: Default node aggregation method to be used. If set to None, the current
            default found in cl_core.library.causal_graph.DEFAULT_NODE_AGGREGATION is used.
        :param activations: Default node activation method to be used. If set to None, the current
            default found in cl_core.library.causal_graph.DEFAULT_NODE_ACTIVATION is used.
        :param variable_type:  The variable type that the node represents. The choices are available through the
        `cai_causal_graph.type_definitions.NodeVariableType` enum.
        """
        super().__init__(identifier=identifier, meta=meta, variable_type=variable_type)

        # set aggregation and activation in the meta
        self.meta['aggregation'] = aggregation
        self.meta['activations'] = activations

    @property
    def aggregation(self):
        return self.meta['aggregation']

    @aggregation.setter
    def aggregation(self, aggregation: Optional[_Aggregation]):
        self.meta['aggregation'] = aggregation

    @property
    def activations(self):
        return self.meta['activations']

    @activations.setter
    def activations(self, activations: Optional[Union[_Activation, List[_Activation]]]):
        self.meta['activations'] = activations


class CausalModelEdge(Edge):
    """A utility class that manages the state of an model edge."""

    def __init__(
        self,
        source: CausalModelNode,
        destination: CausalModelNode,
        edge_type: EdgeType = EdgeType.DIRECTED_EDGE,
        meta: Optional[Dict[str, Any]] = None,
        activations: Optional[Union[_Activation, List[_Activation]]] = None,
    ):
        super().__init__(source, destination, edge_type, meta)

        self.meta['activations'] = activations

    @property
    def activations(self):
        return self.meta['activations']

    @activations.setter
    def activations(self, activations: Optional[Union[_Activation, List[_Activation]]]):
        self.meta['activations'] = activations


class CausalModelGraph(CausalGraph):
    """A low-level class that uniquely defines the state of a causal model graph."""

    def __init__(
        self,
        input_list: Optional[List[NodeLike]] = None,
        output_list: Optional[List[NodeLike]] = None,
        fully_connected: bool = True,
        default_node_aggregation: Optional[_Aggregation] = None,
        default_node_activation: Optional[Union[_Activation, List[_Activation]]] = None,
        default_edge_activation: Optional[Union[_Activation, List[_Activation]]] = None,
    ):
        super().__init__(input_list=input_list, output_list=output_list, fully_connected=fully_connected)

        if default_node_aggregation is None:
            default_node_aggregation = DEFAULT_NODE_AGGREGATION

        if default_node_activation is None:
            default_node_activation = DEFAULT_NODE_ACTIVATION

        if default_edge_activation is None:
            default_edge_activation = DEFAULT_EDGE_ACTIVATION

        # resolve the user-specified default value
        self.default_node_aggregation: Optional[_Aggregation] = default_node_aggregation
        self.default_node_activation: Optional[Union[_Activation, List[_Activation]]] = default_node_activation
        self.default_edge_activation: Optional[Union[_Activation, List[_Activation]]] = default_edge_activation

    def add_node(
        self,
        /,
        identifier: Optional[NodeLike] = None,
        *,
        aggregation: Optional[_Aggregation] = None,
        activations: Optional[Union[_Activation, List[_Activation]]] = None,
        variable_type: Union[str, NodeVariableType] = NodeVariableType.UNSPECIFIED,
        meta: Optional[dict] = None,
        node: Optional[CausalModelNode] = None,
        **kwargs,
    ) -> CausalModelNode:

        if node is not None:
            assert (
                identifier is None
                and variable_type == NodeVariableType.UNSPECIFIED
                and meta is None
                and aggregation is None
                and activations is None
                and len(kwargs) == 0
            ), 'If specifying `node` argument, all other arguments should not be specified.'
            identifier = node.identifier
            variable_type = node.variable_type
            meta = copy.deepcopy(node.meta)
        else:
            assert identifier is not None, (
                'You must either specify an `identifier` of a node to add, '
                'or provide a constructed `Node` object using the `node` parameter.'
            )

        identifier = Node.identifier_from(identifier)

        if identifier in self._nodes_by_identifier:
            raise CausalGraphErrors.NodeDuplicatedError(f'Node already exists: {identifier}')

        node = CausalModelNode(
            identifier,
            variable_type=variable_type,
            meta=meta,
            aggregation=self.default_node_aggregation,
            activations=self.default_node_activation,
        )

        # Add aggregation if defined:
        if aggregation is not None:
            self.aggregation = aggregation

        if activations is not None:
            # Add any of the activations
            activations = [activation for activation in to_list(activations) if activation is not None]
            node.activations = activations

        self._nodes_by_identifier[identifier] = node

        return node

    def add_edge(
        self,
        /,
        source: Optional[NodeLike] = None,
        destination: Optional[NodeLike] = None,
        *,
        edge_type: Union[str, NodeVariableType] = NodeVariableType.UNSPECIFIED,
        activations: Optional[Union[_Activation, List[_Activation]]] = None,
        meta: Optional[dict] = None,
        edge: Optional[CausalModelEdge] = None,
        **kwargs,
    ) -> CausalModelEdge:

        if edge is not None:
            assert (
                source is None
                and destination is None
                and edge_type == EdgeType.DIRECTED_EDGE
                and activations is None
                and meta is None
                and len(kwargs) == 0
            ), 'If specifying `edge` argument, all other arguments should not be specified.'
            source, destination = edge.get_edge_pair()
            edge_type = edge.get_edge_type()
            activations = copy.deepcopy(edge.activations)
            meta = copy.deepcopy(edge.meta)
        else:
            assert source is not None and destination is not None, (
                'You must either specify a `source` and `destination` of an edge to add, '
                'or provide a constructed `Edge` object using the `edge` parameter.'
            )

        source_meta = None
        if isinstance(source, HasMetadata):
            source_meta = source.get_metadata()

        destination_meta = None
        if isinstance(destination, HasMetadata):
            destination_meta = destination.get_metadata()

        source = Node.identifier_from(source)
        destination = Node.identifier_from(destination)

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
            self.add_node(source, meta=source_meta)
            source_nodes = self.get_nodes(source)
        if len(destination_nodes) != 1:
            # The node was not explicitly defined by the user, so we will add it implicitly based on the edge info
            self.add_node(destination, meta=destination_meta)
            destination_nodes = self.get_nodes(destination)
        if len(edges) != 0:
            # We don't allow implicit edge override. The user should delete the node or modify it.
            raise CausalGraphErrors.EdgeDuplicatedError(
                f'An edge already exists between {source} and {destination}. '
                f'Please modify or delete this and then create the new edge explicitly.'
            )

        edge = CausalModelEdge(
            source_nodes[0], destination_nodes[0], edge_type=edge_type, activations=self.default_edge_activation
        )
        if activations is not None:
            edge.activations = activations

        # Add any meta
        if meta is not None:
            edge.meta = meta

        self._edges_by_source[source][destination] = edge
        self._edges_by_destination[destination][source] = edge

        # only add edges to inbound / outbound edges of a node if the specified edge type is directed
        if edge_type == EdgeType.DIRECTED_EDGE:
            self._nodes_by_identifier[destination]._add_inbound_edge(edge)
            self._nodes_by_identifier[source]._add_outbound_edge(edge)

        # check that there are no cycles of directed edges
        try:
            self._assert_node_does_not_depend_on_itself(destination)
        except AssertionError:
            self.delete_edge(source, destination)
            raise CausalGraphErrors.CyclicConnectionError(
                f'Adding an edge from {source} to {destination} would create a cyclic connection.'
            )

        return edge
