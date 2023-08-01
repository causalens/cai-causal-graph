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

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

from cai_causal_graph.exceptions import CausalGraphErrors
from cai_causal_graph.interfaces import CanDictSerialize, HasIdentifier, HasMetadata
from cai_causal_graph.type_definitions import EDGE_T, NodeLike, NodeVariableType


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
        :param meta: The meta values for the node.
        :param variable_type: The variable type that the node represents. The choices are available through the
            `cai_causal_graph.type_definitions.NodeVariableType` enum.
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

    def __hash__(self):
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

    def __ne__(self, other: Any) -> bool:
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
            raise TypeError(f'Expected NodeVariableType or string, got object of type {type(new_type)}.')
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

        return f'Node("{self.identifier}"{type_string})'

    def details(self) -> str:
        """Return a detailed string description of the object."""
        return self.__repr__()

    def to_dict(self) -> dict:
        """Serialize the Node instance to a dictionary."""
        return {
            'identifier': self.identifier,
            'variable_type': self.variable_type,
            'meta': deepcopy(self.meta),
        }


class Edge(HasIdentifier, HasMetadata, CanDictSerialize):
    """A utility class that manages the state of an edge."""

    def __init__(
        self,
        source: Node,
        destination: Node,
        edge_type: EDGE_T = EDGE_T.DIRECTED_EDGE,
        meta: Optional[Dict[str, Any]] = None,
    ):
        """
        :param source: The `cai_causal_graph.graph_components.Node` from which the edge will originate.
        :param destination: The `cai_causal_graph.graph_components.Node` at which the edge will terminate.
        :param edge_type: The type of the edge to be added. Default is
            `cai_causal_graph.type_definitions.EDGE_T.DIRECTED_EDGE`. See `cai_causal_graph.type_definitions.EDGE_T` for
            the list of possible edge types.
        :param meta: The meta values for the node.
        """
        self._source = source
        self._destination = destination
        self._edge_type = edge_type

        self.meta = dict() if meta is None else meta

        # Switches to False if the edge is deleted
        self._valid: bool = True

    def __hash__(self):
        """Return a hash value of the node identifier."""
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
                self.get_edge_type() in [EDGE_T.UNDIRECTED_EDGE, EDGE_T.BIDIRECTED_EDGE, EDGE_T.UNKNOWN_EDGE]
                and self.get_edge_type() == other.get_edge_type()
            )

        return False

    def __ne__(self, other: Any) -> bool:
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

    def get_edge_type(self) -> EDGE_T:
        """
        Return the edge type.

        Please note that to change the edge type, you must use the `CausalGraph.change_edge_type` method defined
        on the causal graph.
        """
        return self._edge_type

    def __repr__(self) -> str:
        """Return a string description of the object."""
        return f'Edge("{self.source.identifier}", "{self.destination.identifier}", type={self._edge_type})'

    def details(self) -> str:
        """Return a detailed string description of the object."""
        return self.__repr__()

    def to_dict(self) -> dict:
        """Serialize the Edge instance to a dictionary."""
        return {
            'source': self._source.to_dict(),
            'destination': self.destination.to_dict(),
            'edge_type': self._edge_type,
            'meta': deepcopy(self.meta),
        }
