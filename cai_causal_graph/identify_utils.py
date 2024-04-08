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
from itertools import combinations
from typing import List, Optional, Set, Tuple, Union

import networkx

from cai_causal_graph import CausalGraph, Skeleton
from cai_causal_graph.exceptions import CausalGraphErrors
from cai_causal_graph.graph_components import Node
from cai_causal_graph.type_definitions import EdgeType, NodeLike


def _verify_identify_inputs(
    graph: Union[CausalGraph, Skeleton], node_1: NodeLike, node_2: Optional[NodeLike] = None
) -> Tuple[str, str]:
    """
    Verify the inputs to the identify utilities.

    :param graph: The graph given by a `cai_causal_graph.causal_graph.CausalGraph` or
        `cai_causal_graph.causal_graph.Skeleton` instance. If a `cai_causal_graph.causal_graph.CausalGraph` is
        provided, it must be a DAG, i.e. it must only contain directed edges and be acyclic, otherwise a `TypeError`
        is raised.
    :param node_1: The first node or its identifier.
    :param node_2: The second (optional) node or its identifier.
    :return: A tuple of the node identifiers. If node_2 was `None`, then `''` is returned for its identifier.
    """
    if isinstance(graph, CausalGraph) and not graph.is_dag():
        raise TypeError(f'Expected a DAG, but got a mixed causal graph.')

    # Confirm node_1 and node_2 are in the graph.
    if not graph.node_exists(node_1):
        raise CausalGraphErrors.NodeDoesNotExistError(f'Node not found: {node_1}')
    if node_2 is not None and not graph.node_exists(node_2):
        raise CausalGraphErrors.NodeDoesNotExistError(f'Node not found: {node_2}')

    # Coerce NodeLike to identifier. We already know they are NodeLike as node_exists does this.
    node_1_id = Node.identifier_from(node_1)
    node_2_id = Node.identifier_from(node_2) if node_2 is not None else ''

    # Ensure node_1 != node_2
    if node_1_id == node_2_id or node_1 == node_2:
        raise ValueError('node_1 and node_2 cannot be equal. Please provide different nodes / node identifiers.')

    return node_1_id, node_2_id


def identify_confounders(graph: CausalGraph, node_1: NodeLike, node_2: NodeLike) -> List[str]:
    """
    Identify all confounders between `node_1` and `node_2` in the provided `graph`.

    A confounder between `node_1` and `node_2` is a node that is a (minimal) ancestor of both the `node_1` and
    `node_2`. Being a _minimal_ ancestor here means that the node is not an ancestor of other confounder nodes, unless
    it has another directed path to either `node_1` or `node_2` that does not go through other confounder nodes.

    Note that this method returns a full list of all possible confounders. It is up to the user to decide which
    confounder(s) to use for downstream tasks, e.g. causal effect estimation. Note, however, that the list of
    (minimal) confounders returned by this method is a sufficient adjustment set for causal effect estimation, and
    therefore it is advised to use all returned variables when adjusting for confounding.

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
        >>> confounder_variables: List[str] = identify_confounders(cg, node_1='x', node_2='y')

    :param graph: The causal graph given by a `cai_causal_graph.causal_graph.CausalGraph` instance. This must be a DAG,
        i.e. it must only contain directed edges and be acyclic, otherwise a `TypeError` is raised.
    :param node_1: The first node or its identifier.
    :param node_2: The second node or its identifier.
    :return: A list of all confounders between `node_1` and `node_2`.
    """

    def _identify_confounders_no_checks_no_descendant_pruning_networkx(
        clean_graph: networkx.DiGraph, n1: str, n2: str
    ) -> Set[str]:
        """
        Private function that does not check if DAG or do descendant pruning.

        This function searches for confounders using the node `n1` as the starting point. Importantly, this
        does not produce a set of minimal confounders.
        """
        # create a copy of the provided graph and prune the children of node 1 and node 2
        removed_edges = list()
        final_graph = clean_graph
        for child in list(final_graph.successors(n1)):
            removed_edges.append((n1, child))
            final_graph.remove_edge(n1, child)
        for child in list(final_graph.successors(n2)):
            removed_edges.append((n2, child))
            final_graph.remove_edge(n2, child)

        # search the parents of node 1 and check whether any of them is an ancestor to node 2
        confounders = set()
        for parent in list(final_graph.predecessors(n1)):
            # add the parent to the confounding set if a directed path exists
            if parent in networkx.ancestors(final_graph, n2):
                confounders.add(parent)
            # otherwise, recursively call this function to identify confounders of the parent
            else:
                confounders = confounders.union(
                    set(_identify_confounders_no_checks_no_descendant_pruning_networkx(final_graph, parent, n2))
                )

        # add the edges that were removed back
        for edge in removed_edges:
            final_graph.add_edge(*edge)

        return confounders

    # verify inputs and obtain node identifiers
    node_1_id, node_2_id = _verify_identify_inputs(graph, node_1, node_2)

    # convert to networkx for optimized performance
    digraph: networkx.DiGraph = graph.to_networkx()

    # identify confounders starting the search from each node
    confounders = _identify_confounders_no_checks_no_descendant_pruning_networkx(digraph, node_1_id, node_2_id)
    reverse_confounders = _identify_confounders_no_checks_no_descendant_pruning_networkx(digraph, node_2_id, node_1_id)

    # take the intersection of both sets to get the minimal confounder set
    # parents of confounders may be identified as confounders if they have a directed path to the second node
    # only occurs in one configuration, i.e. either forward or reverse direction
    minimal_confounders = confounders.intersection(reverse_confounders)

    return list(minimal_confounders)


def identify_instruments(
    graph: CausalGraph, source: NodeLike, destination: NodeLike, max_num_paths: int = 25
) -> List[str]:
    """
    Identify all instrumental variables for the causal effect of `source` on `destination` in the provided `graph`.

    An instrumental variable for the causal effect of `source` on `destination` satisfies the following criteria:
        1. There is a causal effect between the `instrument` and the `source`.
        2. The `instrument` has a causal effect on the `destination` _only_ through the `source`.
        3. There is no confounding between the `instrument` and the `destination`.

    Note that this method returns a full list of all possible instrumental variables. It may not be necessary to use
    all identified instruments in instrumental variable regression, e.g. for causal effect estimation, and it is up to
    the user to decide which instruments to use (if not all).

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
    :param source: The source node or its identifier.
    :param destination: The destination node or its identifier.
    :param max_num_paths: The maximum number of paths to consider between the source and destination. Default is `25`.
    :return: A list of instrumental variables for the causal effect of `source` on `destination`.
    """
    # verify inputs and obtain node identifiers
    source_id, destination_id = _verify_identify_inputs(graph, source, destination)

    # return an empty list if the destination is an ancestor of the source
    if destination_id in graph.get_ancestors(source_id):
        return []

    # get all the confounders between the source and destination
    confounders = identify_confounders(graph, source_id, destination_id)

    # get all the ancestors of the source that are not in the above confounding set
    candidate_nodes = set(graph.get_ancestors(source_id)).difference(confounders)

    # reject all candidate nodes that have a directed path to any confounders, and vice versa
    for confounder in confounders:
        confounder_descendants = graph.get_descendants(confounder)
        confounder_ancestors = graph.get_ancestors(confounder)
        for candidate in candidate_nodes.copy():
            if candidate in confounder_descendants or candidate in confounder_ancestors:
                candidate_nodes.remove(candidate)

    # reject all candidate nodes that have a directed path to the destination that does not go through the source
    for candidate in candidate_nodes.copy():
        for i, causal_path in enumerate(graph.get_all_causal_paths(candidate, destination_id)):
            if i > max_num_paths:
                raise ValueError(
                    f'The number of paths between the instrument candidate {candidate} and destination {destination} '
                    f'exceeds the maximum number of paths {max_num_paths}.'
                )
            if source_id not in causal_path:
                candidate_nodes.remove(candidate)
                break

    # reject all candidate nodes that have a non-empty confounding set between itself and the destination
    for candidate in candidate_nodes.copy():
        if len(identify_confounders(graph, candidate, destination_id)) > 0:
            candidate_nodes.remove(candidate)

    return list(candidate_nodes)


def identify_mediators(
    graph: CausalGraph, source: NodeLike, destination: NodeLike, max_num_paths: int = 25
) -> List[str]:
    """
    Identify all mediators for the causal effect of `source` on `destination` in the provided `graph`.

    A mediator variable for the causal effect of `source` on `destination` satisfies the following criteria:
        1. There is a causal effect between the `source` and the `mediator`.
        2. There is a causal effect between the `mediator` and the `destination`.
        3. The `mediator` blocks all directed causal paths between the `source` and the `destination`.
        4. There is no directed causal path from any confounder between `source` and `destination` to the `mediator`.

    Note that this method returns a full list of all possible mediator variables. It will be up to the user to decide
    which mediators to use for downstream tasks, e.g. causal effect estimation.

    Example:
        >>> from typing import List
        >>> from cai_causal_graph import CausalGraph
        >>> from cai_causal_graph.identify_utils import identify_mediators
        >>>
        >>> # define a causal graph
        >>> cg = CausalGraph()
        >>> cg.add_edge('x', 'm')
        >>> cg.add_edge('m', 'y')
        >>> cg.add_edge('u', 'x')
        >>> cg.add_edge('u', 'y')
        >>> cg.add_edge('x', 'y')
        >>>
        >>> # find the mediators between 'x' and 'y'; output: ['m']
        >>> mediator_variables: List[str] = identify_mediators(cg, source='x', destination='y')

    :param graph: The causal graph given by a `cai_causal_graph.causal_graph.CausalGraph` instance. This must be a DAG,
        i.e. it must only contain directed edges and be acyclic, otherwise a `TypeError` is raised.
    :param source: The source node or its identifier.
    :param destination: The destination node or its identifier.
    :param max_num_paths: The maximum number of paths to consider between the source and destination. Default is `25`.
    :return: A list of mediator variables for the causal effect of `source` on `destination`.
    """
    # verify inputs and obtain node identifiers
    source_id, destination_id = _verify_identify_inputs(graph, source, destination)

    # return an empty list if the destination is an ancestor of the source
    if destination_id in graph.get_ancestors(source_id):
        return []

    # get all the confounders between the source and destination
    confounders = identify_confounders(graph, source_id, destination_id)

    # query all causal paths between the source and destination (also remove trivial paths)
    causal_paths = []
    for i, path in enumerate(graph.get_all_causal_paths(source_id, destination_id)):
        if i > max_num_paths:
            raise ValueError(
                f'The number of paths between the source {source} and destination {destination} exceeds '
                f'the maximum number of paths {max_num_paths}.'
            )
        elif len(path) > 2:
            causal_paths.append(set(path))
        # No else needed as path is just source -> destination
    if len(causal_paths) == 0:
        return []

    # find the intersection of nodes in all causal paths (since mediators must block _all_ causal paths)
    candidate_nodes = set.intersection(*[path.difference(source_id, destination_id) for path in causal_paths])

    # create a copy of the provided graph and prune edges to the children of the source
    pruned_graph = graph.copy()
    for child in pruned_graph.get_children(source_id):
        pruned_graph.remove_edge(source=source_id, destination=child)

    # reject all candidate nodes that are descendants of any confounders in the pruned graph
    for confounder in confounders:
        for candidate in candidate_nodes.copy():
            if candidate in pruned_graph.get_descendants(confounder):
                candidate_nodes.remove(candidate)
                break

    return list(candidate_nodes)


def identify_markov_boundary(graph: Union[CausalGraph, Skeleton], node: NodeLike) -> List[str]:
    """
    Identify all the Markov boundary for the specified `node in the provided `graph`.

    The Markov boundary is defined as the minimal Markov blanket. The Markov blanket is defined as the set of variables
    such that if you condition on them, it makes your variable of interest (`node` in this case) conditionally
    independent of all other variables. The Markov boundary is minimal meaning that you cannot drop any variables from
    it for the conditional independence condition to still hold.

    For a directed acyclic graph (DAG), provided as a `cai_causal_graph.causal_graph.CausalGraph` instance, the Markov
    boundary of node 'A' is defined as the parents of 'A', the children of 'A', and the other parents of the children
    of 'A'.

    For an undirected graph, provided as a `cai_causal_graph.causal_graph.Skeleton` instance, the Markov boundary of
    node 'A' is simply defined as the neighbors of 'A'.

    See https://en.wikipedia.org/wiki/Markov_blanket for further information.

    Example for `cai_causal_graph.causal_graph.CausalGraph`:
        >>> from typing import List
        >>> from cai_causal_graph import CausalGraph
        >>> from cai_causal_graph.identify_utils import identify_markov_boundary
        >>>
        >>> # define a causal graph
        >>> cg = CausalGraph()
        >>> cg.add_edge('u', 'b')
        >>> cg.add_edge('v', 'c')
        >>> cg.add_edge('b', 'a')  # 'b' is a parent of 'a'
        >>> cg.add_edge('c', 'a')  # 'c' is a parent of 'a'
        >>> cg.add_edge('a', 'd')  # 'd' is a child of 'a'
        >>> cg.add_edge('a', 'e')  # 'e' is a child of 'a'
        >>> cg.add_edge('w', 'f')
        >>> cg.add_edge('f', 'd')  # 'f' is a parent of 'd', which is a child of 'a'
        >>> cg.add_edge('d', 'x')
        >>> cg.add_edge('d', 'y')
        >>> cg.add_edge('g', 'e')  # 'g' is a parent of 'e', which is a child of 'a'
        >>> cg.add_edge('g', 'z')
        >>>
        >>> # compute Markov boundary for node 'a'; output: ['b', 'c', 'd', 'e', 'f', 'g']
        >>> # parents: 'b' and 'c', children: 'd' and 'e', and other parents of children are 'f' and 'g'
        >>> # note the order may not match but the elements will be those six.
        >>> markov_boundary: List[str] = identify_markov_boundary(cg, node='a')

    Example for `cai_causal_graph.causal_graph.Skeleton`:
        >>> from typing import List
        >>> from cai_causal_graph import Skeleton
        >>> from cai_causal_graph.identify_utils import identify_markov_boundary
        >>>
        >>> # use causal graph from above and get is skeleton
        >>> skeleton: Skeleton = cg.skeleton
        >>>
        >>> # compute Markov boundary for node 'a'; output: ['b', 'c', 'd', 'e']
        >>> # as we have no directional information in the undirected skeleton, the neighbors of 'a' are returned.
        >>> # note the order may not match but the elements will be those four.
        >>> markov_boundary: List[str] = identify_markov_boundary(skeleton, node='a')

    :param graph: The graph given by a `cai_causal_graph.causal_graph.CausalGraph` or
        `cai_causal_graph.causal_graph.Skeleton` instance. If a `cai_causal_graph.causal_graph.CausalGraph` is
        provided, it must be a DAG, i.e. it must only contain directed edges and be acyclic, otherwise a `TypeError`
        is raised.
    :param node: The node or its identifier.
    :return: A list of all node identifiers for the nodes in the Markov boundary of `node`.
    """

    # verify inputs and obtain node identifier
    if isinstance(graph, (CausalGraph, Skeleton)):
        node_id, _ = _verify_identify_inputs(graph, node, None)
    else:
        raise TypeError(f'Expected graph to be passed as a CausalGraph or Skeleton object. Got {type(graph)}.')

    if isinstance(graph, CausalGraph):
        child_set = set(graph.get_children(node_id))
        mb = list(
            set(graph.get_parents(node_id))
            | child_set
            | {parent for child in child_set for parent in graph.get_parents(child) if parent != node_id}
        )
    else:
        # We already know it is Skeleton here so no need for elif.
        mb = graph.get_neighbors(node_id)

    return mb


def identify_colliders(graph: CausalGraph, unshielded_only: bool = False) -> List[str]:
    """
    Identify all the collider nodes in the provided `graph`.

    :param graph: The graph given by a `cai_causal_graph.causal_graph.CausalGraph` instance.
    :param unshielded_only: If `True`, only unshielded colliders are returned. If `False`, all colliders are returned.
        Default is `False`. A collider is unshielded if there is no edge between any of its parents.
    :return: A list of all node identifiers for the collider nodes in the graph.
    """
    colliders = set()

    bidirected_edges = [(edge.source.identifier, edge.destination.identifier) for edge in graph.get_bidirected_edges()]

    # go through every node in the graph
    for node in graph.get_node_names():
        # the node z is collider if x <> z <> y, x -> z <- y, x <> z <- y, or x -> z <> y
        neighbors = graph.get_neighbors(node)

        potential_parents = set()

        for neighbor_node in neighbors:
            # we need to check if the directed edge (neigh_node, node) exists
            directed_exists = (
                graph.edge_exists(neighbor_node, node)
                and graph.get_edge(neighbor_node, node).edge_type == EdgeType.DIRECTED_EDGE
            )
            # we need to check if the bi-directed edge (neigh_node, node) or (node, neigh_node) exists
            bidirected_exists = (neighbor_node, node) in bidirected_edges or (node, neighbor_node) in bidirected_edges

            condition = directed_exists or bidirected_exists

            if condition:
                potential_parents.add(neighbor_node)

        if len(potential_parents) >= 2:
            # check if the collider is unshielded. A collider is unshielded if there is no edge between any of its
            # parents
            is_unshielded = True
            if unshielded_only:
                # test all the possible combinations of the potential parents
                for parent_1, parent_2 in combinations(potential_parents, 2):
                    if graph.edge_exists(parent_1, parent_2) or graph.edge_exists(parent_2, parent_1):
                        is_unshielded = False
                        break

            if not unshielded_only or is_unshielded:
                colliders.add(node)

    return list(colliders)
