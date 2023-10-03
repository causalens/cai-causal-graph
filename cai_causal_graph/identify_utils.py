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


def identify_instruments(graph: CausalGraph, source: str, destination: str) -> List[str]:
    """
    Identify all instrumental variables for the causal effect of `source` and `destination` in the provided `graph`.

    An instrumental variable for the causal effect of `source` on `destination` satisfies the following criteria:
        1. There is a causal effect between the `instrument` and the `source`.
        2. The `instrument` has a causal effect on the `destination` _only_ through the `source`.
        3. There is no confounding between the `instrument` and the `destination`.

    Example:
        >>> from typing import List
        >>> from cai_causal_graph import CausalGraph
        >>> from cai_causal_graph.identify_utils import identify_instruments
        >>>
        >>> # define a causal graph
        >>> cg = CausalGraph()
        >>> cg.add_edge('z', 'x')
        >>> cg.add_edge('u', 'x')
        >>> cg.add_edge('u', 'y')
        >>> cg.add_edge('x', 'y')
        >>>
        >>> # find the instruments between 'x' and 'y'; output: ['z']
        >>> instrumental_variables: List[str] = identify_instruments(cg, source='x', destination='y')

    :param graph: The causal graph given by a `cai_causal_graph.causal_graph.CausalGraph` instance. This must be a DAG,
        i.e. it must only contain directed edges and be acyclic, otherwise a `TypeError` is raised.
    :param source: The source variable.
    :param destination: The destination variable.
    :return: A list of instrumental variables for the causal effect of `source` on `destination`.
    """
    if not graph.is_dag():
        raise TypeError(f'Expected a DAG, but got a mixed causal graph.')

    # get all the confounders between the source and destination
    confounders = identify_confounders(graph, source, destination)

    # get all the ancestors of the source that are not in the above confounding set
    candidate_nodes = set(graph.get_ancestors(source)).difference(confounders)

    # reject all candidate nodes that have a directed path to any confounders, and vice versa
    for confounder in confounders:
        confounder_descendants = graph.get_descendants(confounder)
        confounder_ancestors = graph.get_ancestors(confounder)
        for candidate in candidate_nodes.copy():
            if candidate in confounder_descendants or candidate in confounder_ancestors:
                candidate_nodes.remove(candidate)

    # reject all candidate nodes that have a directed path to the destination that does not go through the source
    for candidate in candidate_nodes.copy():
        for causal_path in graph.get_all_causal_paths(candidate, destination):
            if source not in causal_path:
                candidate_nodes.remove(candidate)
                break

    # reject all candidate nodes that have a non-empty confounding set between itself and the destination
    for candidate in candidate_nodes.copy():
        if len(identify_confounders(graph, candidate, destination)) > 0:
            candidate_nodes.remove(candidate)

    return list(candidate_nodes)
