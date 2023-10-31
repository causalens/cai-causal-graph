# Changelog

## NEXT

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
    the minimal number of nodes and edges that is required to capture all the information encoded in the original graph.
    This is because a time series causal graph may contain a lot of repetitive information. For example, if the original
    graph is `x(t-2) -> x(t-1) -> x(t)`, then the minimal graph would be `x(t-1) -> x(t)`. In other words, it is a
    graph that has no edges whose destination is not time 0.
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
