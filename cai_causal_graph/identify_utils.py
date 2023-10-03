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
from typing import List

from cai_causal_graph import CausalGraph
from cai_causal_graph.exceptions import CausalGraphErrors
from cai_causal_graph.graph_components import Node
from cai_causal_graph.type_definitions import NodeLike


def identify_confounders(graph: CausalGraph, node_1: NodeLike, node_2: NodeLike) -> List[str]:
    """
    Identify all confounders between `node_1` and `node_2` in the provided `graph`.

    A confounder between `node_1` and `node_2` is a node that is a (minimal) ancestor of both the `node_1` and
    `node_2`. Being a _minimal_ ancestor here means that the node is not an ancestor of other confounder nodes, unless
    it has another directed path to either `node_1` or `node_2` that does not go through other confounder nodes.

    Example:
        >>> from typing import List
        >>> from cai_causal_graph import CausalGraph
        >>> from cai_causal_graph.identify_utils import identify_confounders
        >>>
        >>> # define a causal graph
        >>> cg = CausalGraph()
        >>> cg.add_edge('z', 'u')
        >>> cg.add_edge('u', 'x')
        >>> cg.add_edge('u', 'y')
        >>> cg.add_edge('x', 'y')
        >>>
        >>> # compute confounders between node_1 and node_2; output: ['u']
        >>> confounders_list: List[str] = identify_confounders(cg, node_1='x', node_2='y')

    :param graph: The causal graph given by a `cai_causal_graph.causal_graph.CausalGraph` instance. This must be a DAG,
        i.e. it must only contain directed edges and be acyclic, otherwise a `TypeError` is raised.
    :param node_1: The first node or its identifier.
    :param node_2: The second node or its identifier.
    :return: A list of all confounders between `node_1` and `node_2`.
    """

    def _identify_confounders_no_checks_no_descendant_pruning(clean_graph: CausalGraph, n1: str, n2: str) -> List[str]:
        """Private function that does not check if DAG or do descendant pruning."""

        # create a copy of the provided graph and prune the children of node 1 and node 2
        final_graph = clean_graph.copy()
        for child in final_graph.get_children(n1):
            final_graph.remove_edge(source=n1, destination=child)
        for child in final_graph.get_children(n2):
            final_graph.remove_edge(source=n2, destination=child)

        # search the parents of node 1 and check whether any of them is an ancestor to node 2
        confounders = set()
        for parent in final_graph.get_parents(n1):
            # add the parent to the confounding set if a directed path exists
            if parent in final_graph.get_ancestors(n2):
                confounders.add(parent)
            # otherwise, recursively call this function to identify confounders of the parent
            else:
                confounders = confounders.union(
                    set(_identify_confounders_no_checks_no_descendant_pruning(final_graph, parent, n2))
                )

        # do the reverse of the above by searching through the parents of node 2
        confounders_reverse = set()
        for parent in final_graph.get_parents(n2):
            # add the parent to the confounding set if a directed path exists
            if parent in final_graph.get_ancestors(n1):
                confounders_reverse.add(parent)
            # otherwise, recursively call this function to identify confounders of the parent
            else:
                confounders_reverse = confounders_reverse.union(
                    set(_identify_confounders_no_checks_no_descendant_pruning(final_graph, parent, n1))
                )

        # take the intersection of both sets to get the minimal confounder set
        # parents of confounders may be identified as confounders if they have a directed path to the second node
        # only occurs in one configuration, i.e. either forward or reverse direction
        minimal_confounders = confounders.intersection(confounders_reverse)

        return list(minimal_confounders)

    if not graph.is_dag():
        raise TypeError(f'Expected a DAG, but got a mixed causal graph.')

    # Confirm node_1 and node_2 are in the graph.
    if not graph.node_exists(node_1):
        raise CausalGraphErrors.NodeDoesNotExistError(f'Node not found: {node_1}')
    if not graph.node_exists(node_2):
        raise CausalGraphErrors.NodeDoesNotExistError(f'Node not found: {node_2}')

    # Coerce NodeLike to identifier. We already know they are NodeLike as node_exists does this.
    node_1_id = Node.identifier_from(node_1)
    node_2_id = Node.identifier_from(node_2)

    # Ensure node_1 != node_2
    if node_1_id == node_2_id or node_1 == node_2:
        raise ValueError('node_1 and node_2 cannot be equal. Please provide different nodes / node identifiers.')

    # Remove all descendants of node 1 and node 2 as they won't be necessary in the search for confounders.
    # This is to reduce memory overhead as we pass the graph throughout recursive calls.
    pruned_graph = graph.copy()
    all_descendants = pruned_graph.get_descendants(node_1).union(pruned_graph.get_descendants(node_2))
    # Remove node 1 and node 2 identifiers if they are in descendants. Get descendants returns str so must work with id.
    all_descendants -= {node_1_id, node_2_id}
    for descendant in all_descendants:
        pruned_graph.remove_node(descendant)

    return _identify_confounders_no_checks_no_descendant_pruning(pruned_graph, node_1_id, node_2_id)
