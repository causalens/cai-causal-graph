# Changelog

## NEXT

- Fixed a bug in the `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph.from_adjacency_matrices`
  method of `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph`, introduced in the previous bug fix
  where the method was not handling properly undirected edges in the adjacency matrices.
- Added the boolean keyword argument `construct_minimal` in
  `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph.from_adjacency_matrices` to allow the user to specify
  whether to construct a minimal graph from adjacency matrices. Default is `True`, making the change backwards
  compatible.
- Removed the `return_minimal` argument from the
  `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph.adjacency_matrices` property of
  `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` as this was never working.

## 0.4.5

- Fixed a bug in `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph.from_adjacency_matrices` for
  `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` where the method was not handling properly
  undirected edges in the adjacency matrices.
- Added `mypy-extensions` dependency as `"^1.0.0"` and moved `mypy` back to dev dependency (setting it as `"^1.8.0"`).

## 0.4.4

- Relaxed `mypy` dependency to `"*"`.

## 0.4.3

- Added the `cai_causal_graph.causal_graph.CausalGraph.replace_edge` method to the
  `cai_causal_graph.causal_graph.CausalGraph` class, which allows you to replace an existing edge with a new one.
- Added the `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph.get_nodes_for_variable_name` method to the
  `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` class.
- Added the `cai_causal_graph.graph_components.Node.is_source_node` and
  `cai_causal_graph.graph_components.Node.is_sink_node` methods to the `cai_causal_graph.graph_components.Node`,
  indicating whether the node is a source node or sink node respectively.
- Changed the following methods in `cai_causal_graph.causal_graph.CausalGraph` from static to class methods:
  - `cai_causal_graph.causal_graph.CausalGraph.from_skeleton`
  - `cai_causal_graph.causal_graph.CausalGraph.from_networkx`
  - `cai_causal_graph.causal_graph.CausalGraph.from_gml_string`
  - Note that `cai_causal_graph.causal_graph.CausalGraph.from_dict` and
    `cai_causal_graph.causal_graph.CausalGraph.from_adjacency_matrix` were already class methods. Now all the `from_`
    methods are consistent. This change is transparent to the user.
  - This allowed us to remove them from `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph`. This change
    is transparent to the user.
- Updated `cai_causal_graph.causal_graph.Skeleton` such that its nodes match the class type of its graph.
  - Added ability to pass `graph_class` to the following methods in `cai_causal_graph.causal_graph.Skeleton` such that
    the node classes will match accordingly when the new `cai_causal_graph.causal_graph.Skeleton` is constructed:
    - `cai_causal_graph.causal_graph.Skeleton.from_dict`
    - `cai_causal_graph.causal_graph.Skeleton.from_adjacency_matrix`
    - `cai_causal_graph.causal_graph.Skeleton.from_networkx`
    - `cai_causal_graph.causal_graph.Skeleton.from_gml_string`
    - Note that `cai_causal_graph.causal_graph.Skeleton.from_dict` was already a class method but the other three
      were also changed to class methods for consistency. This change is transparent to the user.
- Fixed a bug where the `cai_causal_graph.type_definitions.NodeVariableType` was not synced between the
  `cai_causal_graph.causal_graph.CausalGraph` and `cai_causal_graph.causal_graph.Skeleton`.
- Changed `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph.from_adjacency_matrices` from a static method
  to a class method to align with other `from_` methods.
- Updated `numpy` dependency from `"^1.18.0"` to `"^1.20.0"` and allow different `numpy` versions in the
  `poetry.lock` for `python` `3.8` and `3.9` - `3.12`. This is to allow specific versions for `3.8` and the others
  as there is no version of `numpy` that supports them all.
- Upgraded `poetry` version from `1.4.2` to `1.7.1` in the GitHub workflows.
- Moved `mypy` from a dev dependency to a dependency as it is now used in source code.
- Updated actions in GitHub workflows to transition from `Node` `16` to `Node` `20`.

## 0.4.2

- Added the `validate` flag to `cai_causal_graph.causal_graph.CausalGraph.add_edge`,
  `cai_causal_graph.causal_graph.CausalGraph.add_edges_from`,
  `cai_causal_graph.causal_graph.CausalGraph.add_edges_from_paths`,
  `cai_causal_graph.causal_graph.CausalGraph.add_edge_by_pair`,
  `cai_causal_graph.causal_graph.CausalGraph.from_dict`,
  `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph.add_edge`, and
  `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph.add_time_edge` which, if set to `False`, will disable
  validation checks. Currently, this disables cyclicity checks when adding edges. Default is `True`, making the change
  backwards compatible. There is no guarantees about the behavior of the resulting graph if this is disabled
  specifically to introduce cycles. This should only be used to speed up this method in situations where it is known
  the resulting graph is still valid, for example when copying a graph.

## 0.4.0

- Removed `**kwargs` from the `cai_causal_graph.causal_graph.CausalGraph.add_node`,
  `cai_causal_graph.causal_graph.CausalGraph.add_edge`, and `cai_causal_graph.causal_graph.CausalGraph.add_edge_by_pair`
  methods in `cai_causal_graph.causal_graph.CausalGraph` and its subclasses, as none of them use it.
- Improved the `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph.get_minimal_graph` method in
  `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` to avoid a recursion issue.
- Added `cai_causal_graph.causal_graph.CausalGraph.add_edges_from_paths` convenience method to the `cai_causal_graph.causal_graph.CausalGraph` class,
  in order to add edges from paths.
- The `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph.get_minimal_graph` method in
  `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` will now always return a graph of the same class type
  as the current graph instance.
- The returned graph type from the `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph.get_summary_graph` method
  in `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` can now be specified in subclasses of
  `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` using the `_SummaryGraphCls` attribute.
- Fixed a bug where `__getitem__` method of `cai_causal_graph.causal_graph.CausalGraph` would work when specifying invalid
  node or edge query, for example a whole path. Instead, a `TypeError` is now raised.
- Fixed a bug where `cai_causal_graph.causal_graph.CausalGraph.delete_edge` would not support passing source and destination
  as `cai_causal_graph.graph_components.Node`.
- Added support for passing source and destination as `cai_causal_graph.graph_components.Node` to `cai_causal_graph.causal_graph.CausalGraph.remove_edge`.
- The `cai_causal_graph.causal_graph.CausalGraph.from_adjacency_matrix` is now a class method (rather than being a static method)
  which has been generalized to return an instance of the class on which it has been called (e.g. enabling returning instances of
  classes inheriting from `cai_causal_graph.causal_graph.CausalGraph`.

## 0.3.14

- Fixed a bug in `cai_causal_graph.identify_utils.identify_confounders` where an empty confounding set would be
  returned in the edge case where all causal paths from the true confounders to `node_1` were blocked by
  ancestors of `node_2`, or vice versa. This comes at a slight performance cost.

## 0.3.13

- Improved the `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph.get_topological_order` method in
  `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` to improve performance. Added a new keyword
  argument `respect_time_ordering` to allow the user to specify whether the topological order must respect the
  time ordering of the nodes. If `respect_time_ordering=True`, the topological order will respect the time ordering,
  otherwise it may not. For example, if the graph is `'Y lag(n=1)' -> 'Y' <- 'X'`, then `['X', 'Y lag(n=1)', 'Y']` and
  `['Y lag(n=1)', 'X', 'Y']` are both valid topological orders. However, only the second one would respect time
  ordering. If both `return_all` and `respect_time_ordering` are `True`, then only all topological orders
  that respect time are returned, not all valid topological orders. The default is `respect_time_ordering=True`,
  matching previous behavior.

## 0.3.12

- Improved efficiency of `cai_causal_graph.identify_utils.identify_confounders` by performing all operations
  directly using `networkx`, removing the need to copy graphs and improving recursive logic, such that
  only the minimal confounders of the specified nodes are calculated (rather than recursively calculating minimal
  confounders of each parent). This results in significant speedups (in the order of hundreds of times in some cases).

## 0.3.11

- Fixed a bug in the `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph.extend_graph` method in
  `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` where the method was not adding nodes correctly with
  particular graph configurations.

## 0.3.10

- Fixed a bug in the `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph.get_minimal_graph` method in
  `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` where floating nodes were not added correctly.
  This also impacted the `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph.extend_graph` method in
  `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph`; it is also fixed now for floating nodes.

## 0.3.9

- Fixed a bug in the `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph.get_topological_order` method in
  `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph`.

## 0.3.8

- Fixed a bug in `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph.from_causal_graph` for
  `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` where the method was not adding floating nodes
  correctly.
- Added `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph.max_backward_lag` and
  `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph.max_forward_lag` properties to
  `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` to return the absolute maximum backward and
  forward time lag of the graph, respectively.
- Fixed a bug with the property `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph.maxlag` in
  `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` since it could give wrong information if the
  future was included in the graph.

## 0.3.7

- Fixed a bug in `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph.from_adjacency_matrices` for
  `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` where the method was not adding floating nodes to
  the graph. Now any floating nodes at time lag 0 will be added.

## 0.3.6

- Added `__iter__` to `cai_causal_graph.causal_graph.Skeleton`.

## 0.3.5

- Added the `cai_causal_graph.identify_utils.identify_markov_boundary` utility function, which allows you to identify
  the Markov boundary of a node in a `cai_causal_graph.causal_graph.CausalGraph` or in a
  `cai_causal_graph.causal_graph.Skeleton`.
- Added `get_neighbor_nodes` and `get_neighbors` methods to `cai_causal_graph.causal_graph.CausalGraph` and
  `cai_causal_graph.causal_graph.Skeleton`. `get_neighbor_nodes` returns the nodes neighboring the specified node while
  `get_neighbors` returns the identifiers of the neighboring nodes. Note: For a
  `cai_causal_graph.causal_graph.CausalGraph`, it does not matter what the edge type is, as long as there is an edge
  between the specified node and another node, that other node is considered its neighbor.

## 0.3.4

- Extended documentation to provide further information regarding the types of mixed graphs that can be defined in a
  `cai_causal_graph.causal_graph.CausalGraph`.

## 0.3.3

- Fixed a bug in `cai_causal_graph.identify_utils.identify_instruments` and
  `cai_causal_graph.identify_utils.identify_mediators`, where an unclear error was raised if the `source` node was a
  descendant of the `destination` node. Instead, these methods now return an empty list in that case.
- Extended the quickstart documentation to describe how to set the `variable_type` when adding a
  `cai_causal_graph.graph_components.Node` / `cai_causal_graph.graph_components.TimeSeriesNode` to a
  `cai_causal_graph.causal_graph.CausalGraph` / `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph`,
  respectively.

## 0.3.2

- Improved documentation.

## 0.3.1

- Fixed a bug for forward extension in the
  `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph.extend_graph` method in
  `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph`.

## 0.3.0

> **_NOTE:_**  **Backwards compatibility warning!** The definition of the
> `cai_causal_graph.type_definitions.EdgeConstraint` enumeration has changed.

- Updated the `cai_causal_graph.type_definitions.EdgeConstraint` enumeration to simplify the enumeration members and
  exposed it at the root level, so it can be imported as `from cai_causal_graph import EdgeConstraint`.
  `cai_causal_graph.type_definitions.EdgeConstraint` is not used by the `cai-causal-graph` package but any packages
  that rely on its definition, must be updated to reflect the new members.

## 0.2.17

- Fixed the docstrings of `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph.from_adjacency_matrices`,
  such that the examples render properly.

## 0.2.16

- Fixed a bug where `cai_causal_graph.utils.get_variable_name_and_lag` would not match variable names with
  non-alphanumeric characters, and would not match variable names with the string `lag` or `future` in them.

## 0.2.15

- Improved performance of checking for cycles when adding edges by avoiding repeated checks.

## 0.2.14

- Modified `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph.add_edge` in
  `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` to always order source and destination nodes by
  `time_lag` even when the edge is not directed. For example,
  `add_edge('c', 'a (lag=1)', edge_type=EdgeType.UNDIRECTED_EDGE)` will add an undirected edge `a (lag=1) -- c` instead
  of `c -- a (lag=1)`. This is done just for convenience and to avoid confusion.

## 0.2.13

- Improved documentation.

## 0.2.12

- Added `cai_causal_graph.graph_components.Node.node_name` property to `cai_causal_graph.graph_components.Node` as an
  alias of `cai_causal_graph.graph_components.Node.identifier`.
- Added `cai_causal_graph.causal_graph.CausalGraph.get_parent_nodes` and
  `cai_causal_graph.causal_graph.CausalGraph.get_children_nodes` to `cai_causal_graph.causal_graph.CausalGraph`. These
  return a list of the parent and children `cai_causal_graph.graph_components.Node` objects, respectively. This is to
  supplement the `cai_causal_graph.causal_graph.CausalGraph.get_parents` and
  `cai_causal_graph.causal_graph.CausalGraph.get_children` methods, which only return the node identifiers.

## 0.2.11

- Improved the `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph.get_topological_order` method for a
  `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` to better account for time.
- Fixed a bug in `cai_causal_graph.causal_graph.CausalGraph` and
  `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` that prevented
  `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph.get_minimal_graph`,
  `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph.get_summary_graph` and
  `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph.extend_graph` from working properly as it did not
  maintain the correct extra information such as node variable types.

## 0.2.10

- Added the `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph.get_nodes_at_lag` and
  `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph.get_contemporaneous_nodes` methods to the
  `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` class to get the nodes at a given lag and the
  contemporaneous nodes of the provided node, respectively.
- General improvements to several `from_*` methods in the
  `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` class.

## 0.2.9

- Added support for `python` version `3.12`.

## 0.2.8

- Changed the order of the documentation in the sidebar to ensure Quickstart is at the top.

## 0.2.7

- Improved internal logic for how an `cai_causal_graph.graph_components.Edge` is instantiated from a dictionary.

## 0.2.6

- Improved the `cai_causal_graph.causal_graph.CausalGraph.copy` method in `cai_causal_graph.causal_graph.CausalGraph`
  such that it is more general and preserves the subclass type. As such, the `.copy` method was removed from the
  `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` class.
- Extended equality methods for the `cai_causal_graph.causal_graph.Skeleton`,
  `cai_causal_graph.causal_graph.CausalGraph`, and
  `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` classes. A new keyword parameter `deep` has been
  added. If `deep=True`, deep equality checks are also done on all nodes and edges in the graphs. To call you must do
  `graph_1.__eq__(graph_2, deep=True)` as `graph_1 == graph_2` still matches previous behavior.

## 0.2.5

- Added the `cai_causal_graph.identify_utils.identify_confounders` utility function, which allows you to identify
  a list of confounders between two nodes in a `cai_causal_graph.causal_graph.CausalGraph`.
- Added the `cai_causal_graph.identify_utils.identify_instruments` utility function, which allows you to identify
  a list of instrumental variables between two nodes in a `cai_causal_graph.causal_graph.CausalGraph`.
- Added the `cai_causal_graph.identify_utils.identify_mediators` utility function, which allows you to identify
  a list of mediators between two nodes in a `cai_causal_graph.causal_graph.CausalGraph`.

## 0.2.4

- Fixed formatting in the documentation.

## 0.2.3

- Added the deserialization method `from_dict` to the following classes:
  `cai_causal_graph.graph_components.Node`, `cai_causal_graph.graph_components.TimeSeriesNode`, and
  `cai_causal_graph.graph_components.Edge`.
- Added the serialization method `cai_causal_graph.graph_components.TimeSeriesNode.to_dict` to
  `cai_causal_graph.graph_components.TimeSeriesNode`. `cai_causal_graph.graph_components.Node` and
  `cai_causal_graph.graph_components.Edge` already had it.
- Changed behavior of `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph.add_node` method in
  `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` such that when both `identifier` and (`time_lag`,
  `variable_name`) are provided. Now, if all are provided, the method will raise an error only if `identifier` is not
  equal to `get_name_with_lag(time_lag, variable_name)`, that is, the correct name.
- Extended equality methods for the `cai_causal_graph.graph_components.Node`, `cai_causal_graph.graph_components.Edge`
  and `cai_causal_graph.graph_components.TimeSeriesNode` classes. A new keyword parameter `deep` has been added.
  If `deep=True`, additional class attributes are also checked; see the docstrings for additional information. To call
  you must do `node_1.__eq__(node_2, deep=True)` as `node_1 == node_2` still matches previous behavior.
- Added `cai_causal_graph.graph_components.Edge.edge_type` property to the `cai_causal_graph.graph_components.Edge`
  class.

## 0.2.2

- Fixed typo in the quickstart documentation.

## 0.2.1

- Fixed `repr` bug in the `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` class.
- Added `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph.to_numpy_by_lag` method to convert the
  `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` to a dictionary of adjacency matrices where the
  keys are the time lags with the values being the adjacency matrices with respect to the variables.
- Changed the `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph.extend_graph` method in
  `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` to work with a non-negative `backward_steps` and
  `forward_steps` instead of strictly positive.
- Fixed edge type in the `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph.extend_graph` method in
  `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph`.
- Added `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph.add_time_edge` method in
  `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` to add a time edge between two nodes. This method
  allows to specify the time lag for source and destination variables. This avoids having to create the corresponding
  node name manually or using the utility function `cai_causal_graph.utils.get_name_with_lag`.
- Added equality method for the `cai_causal_graph.graph_components.TimeSeriesNode` class.
- Extended unit tests for the `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` and
  `cai_causal_graph.graph_components.TimeSeriesNode` classes.
- Documentation:
  - Added a documentation page for the `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` class.
  - Changed quickstart to start from a `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` instead of a
    `cai_causal_graph.causal_graph.CausalGraph`.

## 0.2.0

- Added the `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` class to represent a time series causal
  graph. `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` is aware of the time relationships between
  the nodes in the graph while `cai_causal_graph.causal_graph.CausalGraph` is not. Moreover, the
  `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` class has three new representations:
  - The minimal graph, which can be obtained via the
    `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph.get_minimal_graph` method, defines the graph with
    the minimal number of nodes and edges that is required to capture all the information encoded in the original
    graph. This is because a time series causal graph may contain a lot of repetitive information. For example, if the
    original graph is `x(t-2) -> x(t-1) -> x(t)`, then the minimal graph would be `x(t-1) -> x(t)`. In other words, it
    is a graph that has no edges whose destination is not time 0.
  - The summary graph, which can be obtained via the
    `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph.get_summary_graph` method, defines the graph
    collapsed in time so there is a single node per variable. For example, if the original graph is
    `z(t-2) -> x(t) <- y(t) <- y(t-1)` then the summary group would be `z -> x <- y`. Note, it is possible to have
    cycles in the summary graph. For example, a graph with edges `y(t-1) -> x(t)` and `x(t-1) -> y(t)` would have a
    summary graph of `x <-> y`.
  - The extended graph, which can be obtained via the
    `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph.extend_graph` method, defines the graph obtained
    by extending backward and forward in time via the arguments `backward_steps` and `forward_steps`, respectively.
    This graph may contain lots of redundant information. For example, if the original graph is `x(t-1) -> x(t)` and
    `backward_steps=2` and `forward_steps=1`, then the extended graph would be `x(t-2) -> x(t-1) -> x(t) -> x(t+1)`.
- The `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph.from_adjacency_matrices` method was added to
  instantiate an instance of `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` from a dictionary of
  adjacency matrices where the keys are the time lags.
- Added the `cai_causal_graph.graph_components.TimeSeriesNode` class to extend the
  `cai_causal_graph.graph_components.Node` class to represent time information on the node.
  The following properties were added:
  - `variable_name`: The variable name of the time series node. For example, if the identifier of the time series node
    is `'X1 lag(n=1)'`, i.e., it is a lagged version of the variable `'X1'`, then `variable_name` would be `'X1'`.
  - `time_lag`: The time lag of the time series node. For example, if the identifier of the time series node
    is `'X1 lag(n=1)'`, i.e., it is a lagged version of the variable `'X1'`, then `time_lag` would be `-1`.
- Renamed `EdgeTypeEnum` to `cai_causal_graph.type_definitions.EdgeType`.

## 0.1.3

- Fixed a syntax error in the docstring for the `cai_causal_graph.causal_graph.CausalGraph.get_bidirected_edges`
  method that was preventing the reference docs from being built.

## 0.1.2

- Improved `README` links so images appear on PyPI.
- Upgraded `poetry` version from `1.2.2` to `1.4.2` in the GitHub workflows.

## 0.1.1

- Added security linting checks of source code using `bandit`.
- Improved documentation packaging and publishing.

## 0.1.0

- Initial release of the `cai-causal-graph` package with the `cai_causal_graph.causal_graph.CausalGraph` class and
  component classes: `cai_causal_graph.causal_graph.Skeleton`, `cai_causal_graph.graph_components.Node`, and
  `cai_causal_graph.graph_components.Edge`.
