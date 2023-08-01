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
from enum import Enum, IntEnum
from numbers import Real
from typing import Tuple, Union

from cai_causal_graph.exceptions import CausalGraphErrors
from cai_causal_graph.interfaces import HasIdentifier


class EdgeConstraint(IntEnum):
    """Enumeration defining the types of edge constraints for domain knowledge."""

    NONE = 0  # This says no constraint, which is equivalent to saying it is unknown or maybe there is an edge.
    UNDIRECTED_EDGE = 1
    FORWARD_DIRECTED_EDGE = 2
    BACKWARD_DIRECTED_EDGE = 3
    FORBIDDEN_EDGE = 4


class EdgeTypeEnum(str, Enum):
    """Enumeration defining the types of edges the graph can contain."""

    UNDIRECTED_EDGE = '--'
    DIRECTED_EDGE = '->'
    BIDIRECTED_EDGE = '<>'
    UNKNOWN_EDGE = 'oo'
    UNKNOWN_DIRECTED_EDGE = 'o>'
    UNKNOWN_UNDIRECTED_EDGE = 'o-'

    def __str__(self) -> str:
        """Return the string value for the enumeration."""
        return self.value


class NodeVariableType(str, Enum):
    """Enumeration defining the types of variable a node in a graph can represent."""

    UNSPECIFIED = 'unspecified'
    CONTINUOUS = 'continuous'
    BINARY = 'binary'
    MULTICLASS = 'multiclass'
    ORDINAL = 'ordinal'

    def __str__(self) -> str:
        """Return the string value for the enumeration."""
        return self.value


# Simple Variable IO Types
NUMBER_T = Union[int, float, Real]  # see: https://github.com/python/mypy/issues/3186

# Causal Structure Types
EDGE_T = EdgeTypeEnum  # This should be the base one, it will allow for all that inherit from it.
NODE_T = NodeVariableType  # This should be the base one, it will allow for all that inherit from it.
PAIR_T = Tuple[str, str]  # Defines a pair of variables. Can be used to query edges, paths, functions.
NodeLike = Union[str, HasIdentifier]


def validate_pair_type(pair: Union[PAIR_T, Tuple[NodeLike, NodeLike]]):
    """
    Validate an edge pair type by raising a `cai_causal_graph.exceptions.CausalGraphErrors.EdgeInvalidError` if it is
    not an expected type.
    """
    if not (isinstance(pair, tuple) and len(pair) == 2):
        raise CausalGraphErrors.EdgeInvalidError(
            f'Expected a tuple of (source.identifier, destination.identifier). Got {pair}.'
        )
    if isinstance(pair[0], HasIdentifier) and not isinstance(pair[0].identifier, str):
        raise CausalGraphErrors.EdgeInvalidError(f'Source identifier must be a string. Got {pair[0].identifier}.')
    elif not isinstance(pair[0], str):
        raise CausalGraphErrors.EdgeInvalidError(f'Source identifier must be a string. Got {pair[0]}.')
    # No else needed as source is OK.
    if isinstance(pair[1], HasIdentifier) and not isinstance(pair[1].identifier, str):
        raise CausalGraphErrors.EdgeInvalidError(f'Destination identifier must be a string. Got {pair[1].identifier}.')
    elif not isinstance(pair[1], str):
        raise CausalGraphErrors.EdgeInvalidError(f'Destination identifier must be a string. Got {pair[1]}.')
    # No else needed as destination is OK.
    if pair[0] == pair[1]:
        CausalGraphErrors.EdgeInvalidError(f'Source and destination identifiers cannot be the same. Got {pair}.')
