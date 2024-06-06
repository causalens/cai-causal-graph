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


class CausalGraphErrors:
    """All errors related to causal graphs."""

    class CyclicConnectionError(Exception):
        """Error raised when a cyclic connection would be made in a DAG."""

    class GraphConversionError(Exception):
        """Raised when converting graph formats fails."""

    class InvalidAdjacencyMatrixError(Exception):
        """Error raised when trying to instantiate a causal graph from an invalid adjacency matrix."""

    class InvalidNetworkXError(Exception):
        """Error raised when trying to instantiate a causal graph from an invalid networkx object."""

    class NodeExistsError(Exception):
        """Error raised when node already exists in the graph."""

    class NodeDoesNotExistError(Exception):
        """Error raised when node does not exist in the graph."""

    class NodeDuplicatedError(Exception):
        """Error raised when duplicate node is attempted to be added to the graph."""

    class NodeInvalidError(Exception):
        """Error raised when node specifications are invalid."""

    class EdgeExistsError(Exception):
        """Error raised when edge already exists in the graph."""

    class EdgeDoesNotExistError(Exception):
        """Error raised when edge does not exist in the graph."""

    class EdgeInvalidError(Exception):
        """Error raised when edge specifications are invalid."""

    class EdgeDuplicatedError(Exception):
        """Error raised when duplicate edge is attempted to be added to the graph."""


class CausalKnowledgeErrors:
    """All errors related to causal knowledge."""

    class KnowledgeConflictError(Exception):
        """Error raised when there is a conflict in the causal knowledge."""

    class InvalidDomainKnowledgeError(Exception):
        """Raised when the provided domain knowledge is invalid."""

    class InvalidSkeletonError(Exception):
        """Raised when the provided skeleton is invalid."""

    class InvalidTiersError(Exception):
        """Raised when the provided tiers are invalid."""


class MetaDataError(Exception):
    """Error raised by the `cai_causal_graph.interfaces.HasMetadata` relating to metadata."""

    pass
