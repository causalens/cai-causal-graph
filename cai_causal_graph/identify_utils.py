"""
Copyright (c) 2023 by Impulse Innovations Ltd. Private and confidential. Part of the causaLens product.
"""
from typing import List

from cai_causal_graph import CausalGraph
from cai_causal_graph.exceptions import CausalGraphErrors


def identify_confounders(graph: CausalGraph, source: str, destination: str) -> List[str]:
    """
    Identify all confounders between the `source` and `destination` in the provided `graph`.

    A confounder between the `source` and `destination` node is a node that is a (minimal) ancestor of both the
    `source` and the `destination` node. Being a _minimal_ ancestor here means that the node is not an ancestor of
    other confounder nodes, unless it has another directed path to the `destination` node that does not go through other
    confounder nodes.

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
        >>> # compute confounders between source and destination; output: ['u']
        >>> confounders_list: List[str] = identify_confounders(cg, source='x', destination='y')

    :param graph: The causal graph given by a `cai_causal_graph.causal_graph.CausalGraph` instance. This must be a DAG,
        i.e. it must only contain directed edges, otherwise an error is raised.
    :param source: The source variable.
    :param destination: The destination variable.
    :return: A list of all confounders between the `source` and `destination`.
    """
    if not graph.is_dag():
        raise TypeError(f'Expected a DAG, but got a mixed causal graph.')

    # create a copy of the provided graph and prune the edge between source and destination
    pruned_graph = graph.copy()
    try:
        pruned_graph.remove_edge(source=source, destination=destination)
    except CausalGraphErrors.EdgeDoesNotExistError:
        pass

    # go through each of the parents of the source node
    confounders = set()
    for parent in pruned_graph.get_parents(source):
        # add the parent to the confounding set if a directed path exists
        if parent in pruned_graph.get_ancestors(destination):
            confounders.add(parent)
        # otherwise, recursively call this function to identify confounders of the parent
        else:
            confounders = confounders.union(set(identify_confounders(pruned_graph, parent, destination)))

    return list(confounders)
