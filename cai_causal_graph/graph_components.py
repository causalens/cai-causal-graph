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
from typing import Any, Dict, List, Optional, Tuple, Union

from cai_causal_graph.exceptions import CausalGraphErrors
from cai_causal_graph.interfaces import CanDictSerialize, HasIdentifier, HasMetadata
from cai_causal_graph.metadata_handler import MetaField
from cai_causal_graph.type_definitions import TIME_LAG, VARIABLE_NAME, EdgeType, NodeLike, NodeVariableType
from cai_causal_graph.utils import get_name_with_lag, get_variable_name_and_lag

logger = logging.getLogger(__name__)


class Node(HasIdentifier, HasMetadata, CanDictSerialize):
    """A utility class that manages the state of a node."""

    def __init__(
        self,
        identifier: str,
        meta: Optional[Dict[str, Any]] = None,
        variable_type: NodeVariableType = NodeVariableType.UNSPECIFIED,
    ):
        """
        :param identifier: String that uniquely identifies the node within the causal graph.
        :param meta: The metadata of the node. Default is None. If passed, meta is shallow-copied.
        :param variable_type: The variable type that the node represents. The choices are available through the
            `cai_causal_graph.type_definitions.NodeVariableType` enum. Default is `NodeVariableType.UNSPECIFIED`.
        """
        assert isinstance(
            identifier, str
        ), f'Node identifiers must be strings. Got {identifier} which is of type {type(identifier)}.'
        self._identifier = identifier

        self.variable_type: NodeVariableType = variable_type
        super(HasIdentifier, self).__init__(meta=meta)

        self._inbound_edges: List[Edge] = []
        self._outbound_edges: List[Edge] = []

        # Switches to False if the node is deleted
        self._is_valid: bool = True

    def __hash__(self) -> int:
        """Return a hash value of the node identifier."""
        return hash(self.identifier)

    def __eq__(self, other: object, deep: bool = False) -> bool:
        """
        Check if a node is equal to another node.

        When `deep` is `False` (default), this method checks equality between the node identifiers. When `deep` is
        `True`, variable type and metadata is also checked. Inbound and outbound edges are never checked.

        :param other: The other node to compare to.
        :param deep: If `True`, then the variable type and metadata are also checked, in addition to the identifier.
            Default is `False`.
        """
        if not isinstance(other, Node):
            return False
        if deep:
            return (
                self.identifier == other.identifier
                and self.variable_type == other.variable_type
                and self.meta == other.meta
            )
        return self.identifier == other.identifier

    def __ne__(self, other: object) -> bool:
        """Check if the node is not equal to another node."""
        return not (self == other)

    def _assert_is_valid(self):
        """Assert that the node is valid, i.e. that it has not been deleted."""
        if not self._is_valid:
            raise CausalGraphErrors.NodeDoesNotExistError(f'The node {self._identifier} has been deleted.')

    def invalidate(self):
        """Set this node to be invalid after deleting it."""
        self._assert_is_valid()
        self._is_valid = False

    @property
    def node_name(self) -> str:
        """Alias for identifier."""
        return self.identifier

    @property
    def identifier(self) -> str:
        """Return the node identifier."""
        return self._identifier

    def get_identifier(self) -> str:
        """Return the node identifier."""
        return self.identifier

    @property
    def variable_type(self) -> NodeVariableType:
        """Return the variable type of the node."""
        return self._variable_type

    @variable_type.setter
    def variable_type(self, new_type: Union[NodeVariableType, str]):
        """
        Set the variable type of the node.

        :param new_type: New variable type.
        """
        if isinstance(new_type, str):
            new_type = NodeVariableType(new_type)
        if not isinstance(new_type, NodeVariableType):
            raise TypeError(f'Expected NodeVariableType or string type, got object of type {type(new_type)}.')
        self._variable_type = new_type

    @property
    def metadata(self) -> dict:
        """Return the node metadata."""
        return self.meta

    def get_metadata(self) -> dict:
        """Return the node metadata."""
        return self.metadata

    def get_inbound_edges(self) -> List[Edge]:
        """Get all inbound (directed) edges to the node."""
        self._assert_is_valid()
        return self._inbound_edges

    def get_outbound_edges(self) -> List[Edge]:
        """Get all outbound (directed) edges to the node."""
        self._assert_is_valid()
        return self._outbound_edges

    def count_inbound_edges(self) -> int:
        """Count the number of inbound (directed) edges to the node."""
        return len(self._inbound_edges)

    def count_outbound_edges(self) -> int:
        """Count the number of outbound (directed) edges to the node."""
        return len(self._outbound_edges)

    def is_source_node(self) -> bool:
        """Return whether the node is a source node (no incoming edges)."""
        return len(self._inbound_edges) == 0

    def is_sink_node(self) -> bool:
        """Return whether the node is a sink node (no outgoing edges)."""
        return len(self._outbound_edges) == 0

    def _add_inbound_edge(self, edge: Edge):
        """Add a specific inbound (directed) edge to the node."""
        self._assert_is_valid()
        assert edge not in self._inbound_edges, 'Provided edge is already an inbound edge to the node.'
        self._inbound_edges.append(edge)

    def _add_outbound_edge(self, edge: Edge):
        """Add a specific outbound (directed) edge from the node."""
        self._assert_is_valid()
        assert edge not in self._outbound_edges, 'Provided edge is already an outbound edge from the node.'
        self._outbound_edges.append(edge)

    def _delete_inbound_edge(self, edge: Edge):
        """Delete a specific inbound (directed) edge to the node."""
        self._assert_is_valid()
        self._inbound_edges.remove(edge)  # Will raise ValueError if edge is not in the list.

    def _delete_outbound_edge(self, edge: Edge):
        """Delete a specific outbound (directed) edge to the node."""
        self._assert_is_valid()
        self._outbound_edges.remove(edge)  # Will raise ValueError if edge is not in the list.

    @staticmethod
    def identifier_from(node_like: NodeLike) -> str:
        """Return the node identifier from a node-like object instance."""
        if isinstance(node_like, HasIdentifier):
            return str(node_like.get_identifier())
        elif isinstance(node_like, str):
            return node_like
        else:
            raise TypeError(f'The provided node needs to be a string or HasIdentifier subclass. Got {type(node_like)}.')

    def __repr__(self) -> str:
        """Return a string description of the object."""
        type_string = f', type="{self.variable_type}"' if self.variable_type != NodeVariableType.UNSPECIFIED else ''

        return f'{self.__class__.__name__}("{self.identifier}"{type_string})'

    def details(self) -> str:
        """Return a detailed string description of the object."""
        return self.__repr__()

    def to_dict(self, include_meta: bool = True) -> dict:
        """
        Serialize a `cai_causal_graph.graph_components.Node` instance to a dictionary.

        Returned dictionary contains a shallow-copy of the metadata of this node (if `include_meta` is `True`).

        :param include_meta: Whether to include meta information about the node in the dictionary. Default is `True`.
        :return: The dictionary representation of the `cai_causal_graph.graph_components.Node` instance.
        """
        node_dict = {
            'identifier': self.identifier,
            'variable_type': self.variable_type,
            'node_class': self.__class__.__name__,
        }
        if include_meta:
            node_dict['meta'] = self.meta.copy()  # type: ignore

        return node_dict

    @classmethod
    def from_dict(cls, node_dict: dict) -> Node:
        """Return a `cai_causal_graph.graph_components.Node` instance from a dictionary."""
        assert 'identifier' in node_dict, 'The provided dictionary does not contain an identifier.'
        return cls(
            identifier=node_dict['identifier'],
            variable_type=node_dict.get('variable_type', NodeVariableType.UNSPECIFIED),
            meta=node_dict.get('meta', None),
        )


class TimeSeriesNode(Node):
    """
    Time series node.

    A node in a time series causal graph will have additional metadata and attributes that provides the time
    information of the node together with the variable name.

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

        :param identifier: String that uniquely identifies the node within the causal graph. If the `identifier` is
            provided, the `time_lag` and `variable_name` will be extracted from the `identifier`. Default is `None`.
        :param time_lag: The time lag of the node. If `time_lag` is provided, then `variable_name` must be provided
            to set the identifier. If both `time_lag` and `variable_name` are provided, the `identifier` must be `None`.
            Default is `None`.
        :param variable_name: The variable name of the node. If `variable_name` is provided, then `time_lag` must be
            provided to set the identifier. If both `time_lag` and `variable_name` are provided, the `identifier` must
            be `None`. Default is `None`.
        :param meta: The metadata of the node. Default is `None`. If passed, meta is shallow-copied.
        :param variable_type: The variable type that the node represents. The choices are available through the
            `cai_causal_graph.type_definitions.NodeVariableType` enum. Default is `NodeVariableType.UNSPECIFIED`.
        """
        if time_lag is not None and variable_name is not None:
            # if identifier is not None and both time_lag and variable_name are provided,
            # check that the identifier is correct.
            rec_identifier = get_name_with_lag(variable_name, time_lag)
            assert identifier is None or identifier == rec_identifier, (
                'The provided identifier does not match the provided time lag and variable name. Either provide a '
                'correct identifier or do not provide the time lag and variable name. Or provide the correct time lag '
                'and variable name and do not provide the identifier.'
            )
            identifier = get_name_with_lag(variable_name, time_lag)
        elif identifier is not None:
            assert (
                time_lag is None and variable_name is None
            ), 'If `identifier` is provided, `time_lag` and `variable_name` must be `None`.'
            identifier = Node.identifier_from(identifier)
            variable_name, time_lag = get_variable_name_and_lag(identifier)
        else:
            raise ValueError(
                'Either `identifier` or both `time_lag` and `variable_name` must be provided to initialize a time '
                'series node.'
            )

        meta = self._process_meta(meta=meta, kwargs_dict=dict(variable_name=variable_name, time_lag=time_lag))

        # populate the metadata for the node
        super().__init__(identifier=identifier, meta=meta, variable_type=variable_type)

    @classmethod
    def get_metadata_schema(cls) -> List[MetaField]:
        """
        Return the metadata schema of `TimeSeriesNode`.

        See `cai_causal_graph.interfaces.HasMetadata.get_metadata_schema` for more information on the how the
        metadata schema is used.
        """
        return super().get_metadata_schema() + [
            MetaField(metatag=TIME_LAG, property_name='time_lag'),
            MetaField(metatag=VARIABLE_NAME, property_name='variable_name'),
        ]

    @property
    def time_lag(self) -> int:
        """Return the time lag of the node from the metadata."""
        lag = self.meta.get(TIME_LAG)
        if lag is None:
            raise ValueError(f'The time lag for node {self.identifier} is not set.')
        return lag

    @property
    def variable_name(self) -> str:
        """Return the variable name of the node from the metadata."""
        name = self.meta.get(VARIABLE_NAME)
        if name is None:
            raise ValueError(f'The variable name for node {self.identifier} is not set.')
        return name

    def __eq__(self, other: object, deep: bool = False) -> bool:
        """
        Check if a node is equal to another node.

        When `deep` is `False` (default), this method checks equality between the node identifiers, variable names, and
        time lags. When `deep` is `True`, variable type and metadata is also checked. Inbound and outbound edges are
        never checked.

        :param other: The other node to compare with.
        :param deep: If `True`, then the variable type and metadata are also checked, in addition to the identifier,
            variable name, and time lag. Default is `False`.
        """
        if not isinstance(other, TimeSeriesNode):
            return False
        # As TimeSeriesNode is a subclass of Node, we can use the Node.__eq__ method and then check for the properties
        if not super().__eq__(other, deep):
            return False
        return self.variable_name == other.variable_name and self.time_lag == other.time_lag

    def __hash__(self) -> int:
        """Return a hash value of the node identifier."""
        return super().__hash__()

    def to_dict(self, include_meta: bool = True) -> dict:
        """
        Serialize a `cai_causal_graph.graph_components.TimeSeriesNode` instance to a dictionary.

        :param include_meta: Whether to include meta information about the node in the dictionary. Default is `True`.
        :return: The dictionary representation of the `cai_causal_graph.graph_components.TimeSeriesNode` instance.
        """
        dictionary = super().to_dict(include_meta)
        # add the time lag and variable name to the dictionary
        dictionary.update({TIME_LAG: self.time_lag, VARIABLE_NAME: self.variable_name})
        return dictionary


class Edge(HasIdentifier, HasMetadata, CanDictSerialize):
    """A utility class that manages the state of an edge."""

    _NodeClassDict = {'Node': Node, 'TimeSeriesNode': TimeSeriesNode}

    def __init__(
        self,
        source: Node,
        destination: Node,
        edge_type: EdgeType = EdgeType.DIRECTED_EDGE,
        meta: Optional[Dict[str, Any]] = None,
    ):
        """
        :param source: The `cai_causal_graph.graph_components.Node` from which the edge will originate.
        :param destination: The `cai_causal_graph.graph_components.Node` at which the edge will terminate.
        :param edge_type: The type of the edge to be added. Default is
            `cai_causal_graph.type_definitions.EdgeType.DIRECTED_EDGE`. See `cai_causal_graph.type_definitions.EdgeType`
            for the list of possible edge types.
        :param meta: The meta values for the edge. Default is `None`. If passed, meta is shallow-copied.
        """
        self._source = source
        self._destination = destination
        self._edge_type = edge_type

        super(HasIdentifier, self).__init__(meta=meta)

        # Switches to False if the edge is deleted
        self._valid: bool = True

    def __hash__(self) -> int:
        """Return a hash value of the edge identifier."""
        return hash(self.identifier)

    def __eq__(self, other: object, deep: bool = False) -> bool:
        """
        Check if the edge is equal to another edge.

        When `deep` is `False` (default), this method checks equality between the source node identifiers, destination
        node identifiers, and edge types. When `deep` is `True`, the metadata is also checked as well as a deep
        equality check on the nodes.

        For some edge types, particularly
        `cai_causal_graph.type_definitions.EdgeType.UNDIRECTED_EDGE`,
        `cai_causal_graph.type_definitions.EdgeType.BIDIRECTED_EDGE`, and
        `cai_causal_graph.type_definitions.EdgeType.UNKNOWN_EDGE`, there is inherently no direction. Therefore, edges
        of those types with the source and destination nodes switched are considered equal.

        :param other: The other edge to compare with.
        :param deep: If `True`, then a deep equality check is done on the nodes and the metadata is also checked, in
            addition to the edge types. Default is `False`.
        """
        if not isinstance(other, Edge):
            return False

        dont_care_direction = [EdgeType.UNDIRECTED_EDGE, EdgeType.BIDIRECTED_EDGE, EdgeType.UNKNOWN_EDGE]

        if deep:
            # Check nodes for deep equality.
            are_sources_not_equal = not self.source.__eq__(other.source, True)
            are_destinations_not_equal = not self.destination.__eq__(other.destination, True)

            # Some edges inherently have no direction. So allow them to be defined with opposite source/destination
            if self.get_edge_type() in dont_care_direction and self.get_edge_type() == other.get_edge_type():
                # allow source and destination to be flipped between each edge
                if are_sources_not_equal and not self.source.__eq__(other.destination, True):
                    return False
                if are_destinations_not_equal and not self.destination.__eq__(other.source, True):
                    return False
            elif are_sources_not_equal or are_destinations_not_equal:
                # source and destination must be the same between each edge
                return False
            # No else needed as source nodes and destination nodes are deeply equal.

            # check that the metadata is the same
            if self.meta != other.meta:
                return False

        # if the same source and destination, check that the edge type is the same
        if self.get_edge_pair() == other.get_edge_pair():
            return self.get_edge_type() == other.get_edge_type()
        elif self.get_edge_pair() == other.get_edge_pair()[::-1]:
            # Some edges inherently have no direction. So allow them to be defined with opposite source/destination
            return self.get_edge_type() in dont_care_direction and self.get_edge_type() == other.get_edge_type()

        return False

    def __ne__(self, other: object) -> bool:
        """Check if the edge is not equal to another edge."""
        return not (self == other)

    def _assert_valid(self):
        """Assert that the edge is valid, i.e. that has not been deleted."""
        if not self._valid:
            raise CausalGraphErrors.EdgeDoesNotExistError(f'The edge {self.identifier} has been deleted.')

    def invalidate(self):
        """Set this edge to be invalid after deleting it."""
        self._assert_valid()
        self._valid = False

    @property
    def source(self) -> Node:
        """Return the source node."""
        self._assert_valid()
        return self._source

    @property
    def destination(self) -> Node:
        """Return the destination node."""
        self._assert_valid()
        return self._destination

    @property
    def identifier(self) -> str:
        """Return the edge identifier."""
        return str(self.get_edge_pair())

    def get_identifier(self) -> str:
        """Return the edge identifier."""
        return self.identifier

    @property
    def descriptor(self) -> str:
        """Return the edge descriptor."""
        return f'({self._source.identifier} {self._edge_type} {self._destination.identifier})'

    @property
    def metadata(self) -> dict:
        """Return the edge metadata."""
        return self.meta

    def get_metadata(self) -> dict:
        """Return the edge metadata."""
        return self.metadata

    @property
    def edge_type(self) -> EdgeType:
        """Return the edge type."""
        return self._edge_type

    def get_edge_type(self) -> EdgeType:
        """
        Return the edge type.

        Please note that to change the edge type, you must use the
        `cai_causal_graph.causal_graph.CausalGraph.change_edge_type` method defined on the causal
        graph.
        """
        return self._edge_type

    def get_edge_pair(self) -> Tuple[str, str]:
        """Return a tuple of the source node and destination node identifiers."""
        return self._source.identifier, self._destination.identifier

    def __repr__(self) -> str:
        """Return a string description of the object."""
        return (
            f'{self.__class__.__name__}("{self.source.identifier}", "{self.destination.identifier}", '
            f'type={self._edge_type})'
        )

    def details(self) -> str:
        """Return a detailed string description of the object."""
        return self.__repr__()

    @classmethod
    def from_dict(cls, edge_dict: dict) -> Edge:
        """
        Deserialize a dictionary to a `cai_causal_graph.graph_components.Edge` instance.

        :param edge_dict: The dictionary representation of the `cai_causal_graph.graph_components.Edge` instance.
            If the `node class` is not specified, it defaults to `cai_causal_graph.graph_components.Node`.
        :return: The `cai_causal_graph.graph_components.Edge` instance.
        """
        # if node class is not specified, default to Node
        source_node_class = edge_dict['source'].get('node_class', 'Node')
        destination_node_class = edge_dict['destination'].get('node_class', 'Node')

        # if not in the dict, then try with the generic node class and send a debug message
        if source_node_class not in cls._NodeClassDict:
            source_node_class = 'Node'
            logger.debug(
                f'The source node class was not specified in the edge dictionary. '
                f'Using the generic node class {source_node_class}.'
            )

        if destination_node_class not in cls._NodeClassDict:
            destination_node_class = 'Node'
            logger.debug(
                f'The destination node class was not specified in the edge dictionary. '
                f'Using the generic node class {destination_node_class}.'
            )

        SourceNodeCls = cls._NodeClassDict[source_node_class]
        DestinationNodeCls = cls._NodeClassDict[destination_node_class]
        assert issubclass(SourceNodeCls, Node)   # for linting
        assert issubclass(DestinationNodeCls, Node)   # for linting

        source = SourceNodeCls.from_dict(edge_dict['source'])
        destination = DestinationNodeCls.from_dict(edge_dict['destination'])

        edge_type = edge_dict['edge_type']

        meta = edge_dict.get('meta', None)

        return cls(source=source, destination=destination, edge_type=edge_type, meta=meta)

    def to_dict(self, include_meta: bool = True) -> dict:
        """
        Serialize a `cai_causal_graph.graph_components.Edge` instance to a dictionary.

        Returned dictionary contains a shallow-copy of the metadata of this edge (if `include_meta` is `True`).

        :param include_meta: Whether to include meta information about the edge in the dictionary. Default is `True`.
        :return: The dictionary representation of the `cai_causal_graph.graph_components.Edge` instance.
        """
        edge_dict = {
            'source': self._source.to_dict(),
            'destination': self.destination.to_dict(),
            'edge_type': self._edge_type,
        }

        if include_meta:
            edge_dict['meta'] = self.meta.copy()

        return edge_dict


class TimeSeriesEdge(Edge):
    """
    Class defining a time-series edge.

    Time-series edge is equivalent to a `cai_causal_graph.graph_components.Edge` class, except that it only
    ensures that the destination of an edge is at the same or later time than its source.

    This means that it is not possible to construct a directed time-series edge which does not respect time. Moreover,
    if an undirected edge is constructed by passing a source which is at a later time than the destination, the source
    and destination are swapped.
    """

    _NodeClassDict = Edge._NodeClassDict.copy()
    _NodeClassDict.update({TimeSeriesNode.__name__: TimeSeriesNode})

    source: TimeSeriesNode
    destination: TimeSeriesNode

    def __init__(
        self,
        source: Node,
        destination: Node,
        edge_type: EdgeType = EdgeType.DIRECTED_EDGE,
        meta: Optional[Dict[str, Any]] = None,
    ):
        """
        Construct a `TimeSeriesEdge`.

        :param source: The `cai_causal_graph.graph_components.Node` from which the edge will originate.
        :param destination: The `cai_causal_graph.graph_components.Node` at which the edge will terminate.
        :param edge_type: The type of the edge to be added. Default is
            `cai_causal_graph.type_definitions.EdgeType.DIRECTED_EDGE`. See `cai_causal_graph.type_definitions.EdgeType`
            for the list of possible edge types.
        :param meta: The meta values for the edge. Default is `None`. If passed, meta is shallow-copied.
        """
        if not isinstance(source, TimeSeriesNode):
            source = TimeSeriesNode.from_dict(source.to_dict(include_meta=True))

        assert isinstance(source, TimeSeriesNode)   # for lint

        if not isinstance(destination, TimeSeriesNode):
            destination = TimeSeriesNode.from_dict(destination.to_dict(include_meta=True))

        assert isinstance(destination, TimeSeriesNode)   # for lint

        # If edge type is not directed, swap source and destination to respect time
        if edge_type != EdgeType.DIRECTED_EDGE and source.time_lag > destination.time_lag:
            source, destination = destination, source
        elif source.time_lag > destination.time_lag:
            raise ValueError(
                f'Cannot add a directed edge between {source} and {destination} because this does not respect time.'
            )

        super().__init__(source=source, destination=destination, edge_type=edge_type, meta=meta)
