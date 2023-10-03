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


def identify_confounders(graph: CausalGraph, node_1: str, node_2: str) -> List[str]:
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
    :param node_1: The first variable.
    :param node_2: The second variable.
    :return: A list of all confounders between `node_1` and `node_2`.
    """
    if not graph.is_dag():
        raise TypeError(f'Expected a DAG, but got a mixed causal graph.')

    # create a copy of the provided graph and prune the children of node_1 and node_2
    pruned_graph = graph.copy()
    for child in pruned_graph.get_children(node_1):
        pruned_graph.remove_edge(source=node_1, destination=child)
    for child in pruned_graph.get_children(node_2):
        pruned_graph.remove_edge(source=node_2, destination=child)

    # search the parents of node_1 and check whether any of them is an ancestor to node_2
    confounders = set()
    for parent in pruned_graph.get_parents(node_1):
        # add the parent to the confounding set if a directed path exists
        if parent in pruned_graph.get_ancestors(node_2):
            confounders.add(parent)
        # otherwise, recursively call this function to identify confounders of the parent
        else:
            confounders = confounders.union(set(identify_confounders(pruned_graph, parent, node_2)))

    # do the reverse of the above by searching through the parents of node_2
    confounders_reverse = set()
    for parent in pruned_graph.get_parents(node_2):
        # add the parent to the confounding set if a directed path exists
        if parent in pruned_graph.get_ancestors(node_1):
            confounders_reverse.add(parent)
        # otherwise, recursively call this function to identify confounders of the parent
        else:
            confounders_reverse = confounders_reverse.union(set(identify_confounders(pruned_graph, parent, node_1)))

    # take the intersection of both sets to get the minimal confounder set
    # parents of confounders may be identified as confounders if they have a directed path to the second node
    # only occurs in one configuration, i.e. either forward or reverse direction
    minimal_confounders = confounders.intersection(confounders_reverse)

    return list(minimal_confounders)
