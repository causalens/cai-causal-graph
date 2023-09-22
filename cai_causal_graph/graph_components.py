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
from typing import Any, Dict, List, Optional, Tuple, Union

from cai_causal_graph.exceptions import CausalGraphErrors
from cai_causal_graph.interfaces import CanDictSerialize, HasIdentifier, HasMetadata
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
        :param meta: The metadata of the node. Default is None.
        :param variable_type: The variable type that the node represents. The choices are available through the
            `cai_causal_graph.type_definitions.NodeVariableType` enum. Default is `NodeVariableType.UNSPECIFIED`.
        """
        assert isinstance(
            identifier, str
        ), f'Node identifiers must be strings. Got {identifier} which is of type {type(identifier)}.'
        self._identifier = identifier

        self.variable_type: NodeVariableType = variable_type
        self.meta = dict() if meta is None else meta
        assert isinstance(self.meta, dict) and all(
            isinstance(k, str) for k in self.meta
        ), 'Metadata must be provided as a dictionary with strings as keys.'

        self._inbound_edges: List[Edge] = []
        self._outbound_edges: List[Edge] = []

        # Switches to False if the node is deleted
        self._is_valid: bool = True

    def __hash__(self) -> int:
        """Return a hash value of the node identifier."""
        return hash(self.identifier)

    def __eq__(self, other: object) -> bool:
        """
        Check if a node is equal to another node.

        This method checks for the node identifier, but ignores variable type, inbound/outbound edges, and any metadata.
        """
        if not isinstance(other, Node):
            return False
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

        :param include_meta: Whether to include meta information about the node in the dictionary. Default is `True`.
        :return: The dictionary representation of the `cai_causal_graph.graph_components.Node` instance.
        """
        node_dict = {
            'identifier': self.identifier,
            'variable_type': self.variable_type,
        }
        if include_meta:
            node_dict['meta'] = deepcopy(self.meta)   # type: ignore

        return node_dict

    @classmethod
    def from_dict(cls, node_dict: dict) -> Node:
        """Return a `cai_causal_graph.graph_components.Node` instance from a dictionary."""
        return cls(
            identifier=node_dict['identifier'],
            variable_type=node_dict['variable_type'],
            meta=node_dict.get('meta', {}),
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
        :param meta: The metadata of the node. Default is `None`.
        :param variable_type: The variable type that the node represents. The choices are available through the
            `cai_causal_graph.type_definitions.NodeVariableType` enum. Default is `NodeVariableType.UNSPECIFIED`.
        """
        if time_lag is not None and variable_name is not None:
            # if identifier is None and both time_lag and variable_name are provided, check that the identifier is correct
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

        # populate the metadata for each node
        if meta is not None:
            meta = meta.copy()
            meta_time_lag = meta.get(TIME_LAG)
            meta_variable_name = meta.get(VARIABLE_NAME)
            if meta_time_lag is not None and meta_time_lag != time_lag:
                logger.info(
                    'The current time lag in the meta (%d) for node %s will be overwritten to the newly provided value (%d).',
                    meta_time_lag,
                    identifier,
                    time_lag,
                )
            if meta_variable_name is not None and meta_variable_name != variable_name:
                logger.info(
                    'The current variable name in the meta (%s) for node %s will be overwritten to the newly provided value (%s).',
                    meta_variable_name,
                    identifier,
                    variable_name,
                )
            meta.update({TIME_LAG: time_lag, VARIABLE_NAME: variable_name})
        else:
            meta = {TIME_LAG: time_lag, VARIABLE_NAME: variable_name}

        # populate the metadata for the node
        super().__init__(identifier, meta, variable_type)

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

    def __eq__(self, other: object) -> bool:
        """
        Check if a node is equal to another node.

        This method checks for the node identifier, variable name, and time lag, but ignores variable type,
        inbound/outbound edges, and any other metadata.
        """
        if not isinstance(other, TimeSeriesNode):
            return False
        # As TimeSeriesNode is a subclass of Node, we can use the Node.__eq__ method and then check for the properties
        if not super().__eq__(other):
            return False
        return self.variable_name == other.variable_name and self.time_lag == other.time_lag

    def __hash__(self) -> int:
        """Return a hash value of the node identifier."""
        return super().__hash__()

    def to_dict(self, include_meta: bool = True) -> dict:
        """
        Return a dictionary representation of the time series node.

        :param include_meta: Whether to include the metadata in the dictionary. Default is `True`.
        :return: The dictionary representation of the time series node.
        """
        dictionary = super().to_dict(include_meta)
        # add the time lag and variable name to the dictionary
        dictionary.update({TIME_LAG: self.time_lag, VARIABLE_NAME: self.variable_name})
        return dictionary

    @classmethod
    def from_dict(cls, dictionary: dict) -> TimeSeriesNode:
        """
        Create a time series node from a dictionary.

        :param dictionary: The dictionary from which to create the time series node.
        :return: The time series node created from the dictionary.
        """
        assert 'identifier' in dictionary, 'The dictionary must contain the `identifier` key.'
        assert TIME_LAG in dictionary, f'The dictionary must contain the `{TIME_LAG}` key.'
        assert VARIABLE_NAME in dictionary, f'The dictionary must contain the `{VARIABLE_NAME}` key.'

        return cls(
            identifier=dictionary.get('identifier'),
            time_lag=dictionary.get(TIME_LAG),
            variable_name=dictionary.get(VARIABLE_NAME),
            meta=dictionary.get('meta', None),
            variable_type=dictionary.get('variable_type', NodeVariableType.UNSPECIFIED),
        )


class Edge(HasIdentifier, HasMetadata, CanDictSerialize):
    """A utility class that manages the state of an edge."""

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
        :param meta: The meta values for the node.
        """
        self._source = source
        self._destination = destination
        self._edge_type = edge_type

        self.meta = dict() if meta is None else meta

        # Switches to False if the edge is deleted
        self._valid: bool = True

    def __hash__(self) -> int:
        """Return a hash value of the edge identifier."""
        return hash(self.identifier)

    def __eq__(self, other: object) -> bool:
        """
        Check if the edge is equal to another edge.

        This method checks for the edge source, destination, and type but ignores any metadata.
        """
        if not isinstance(other, Edge):
            return False

        # if the same source and destination, check that the edge type is the same
        if self.get_edge_pair() == other.get_edge_pair():
            return self.get_edge_type() == other.get_edge_type()
        elif self.get_edge_pair() == other.get_edge_pair()[::-1]:
            # Some edges inherently have no direction. So allow them to be defined with opposite source/destination
            return (
                self.get_edge_type() in [EdgeType.UNDIRECTED_EDGE, EdgeType.BIDIRECTED_EDGE, EdgeType.UNKNOWN_EDGE]
                and self.get_edge_type() == other.get_edge_type()
            )

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

    def get_edge_pair(self) -> Tuple[str, str]:
        """Return a tuple of the source node and destination node identifiers."""
        return self._source.identifier, self._destination.identifier

    def get_edge_type(self) -> EdgeType:
        """
        Return the edge type.

        Please note that to change the edge type, you must use the
        `cai_causal_graph.causal_graph.CausalGraph.change_edge_type` method defined on the causal
        graph.
        """
        return self._edge_type

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
        :return: The `cai_causal_graph.graph_components.Edge` instance.
        """
        source = Node.from_dict(edge_dict['source'])
        destination = Node.from_dict(edge_dict['destination'])

        edge_type = edge_dict['edge_type']

        if 'meta' in edge_dict:
            meta = edge_dict['meta']
        else:
            meta = None

        return cls(source, destination, edge_type, meta)

    def to_dict(self, include_meta: bool = True) -> dict:
        """
        Serialize a `cai_causal_graph.graph_components.Edge` instance to a dictionary.

        :param include_meta: Whether to include meta information about the edge in the dictionary. Default is `True`.
        :return: The dictionary representation of the `cai_causal_graph.graph_components.Edge` instance.
        """
        edge_dict = {
            'source': self._source.to_dict(),
            'destination': self.destination.to_dict(),
            'edge_type': self._edge_type,
        }

        if include_meta:
            edge_dict['meta'] = deepcopy(self.meta)

        return edge_dict
