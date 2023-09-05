# Changelog

## 0.2.0

- Added the `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` class to represent a time series causal
  graph. `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` is aware of the time relationships between
  the nodes in the graph while `cai_causal_graph.causal_graph.CausalGraph` is not. Moreover, the
  `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` class has three new representations:
  - The minimal graph, which can be obtained via the
    `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph.get_minimal_graph` method, defines the graph with
    the minimal number of nodes and edges that is required to capture all the information encoded in the original graph.
    This is because a time series causal graph may contain a lot repetitive information. For example, if the original
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

- Added `cai_causal_graph.time_series_causal_graph.TimeSeriesNode` class to extend the `cai_causal_graph.graph_components.Node` class to time series representations of causal graphs. The main additions are:
  - `variable_name`: the variable name of the time series node. For example, if the time series node is a lagged version of the variable `x`, then `variable_name` would be `x`.
  - `time_lag` attribute that represents the time lag of the time series node.
- Added `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` class to deal with time series representations of causal graphs. The main differences from the `cai_causal_graph.causal_graph.CausalGraph` class is that the `TimeSeriesCausalGraph` is aware of time. Moreover, the `TimeSeriesCausalGraph` class has three new representations of the causal graph:
  - minimal graph (obtained with the method `get_minimal_graph`) defined as the graph with the minimal number of edges that is equivalent to the original graph. In other words, it is a graph that has no edges whose destination is not time delta 0.
    For example, if the original graph has the following edges: `x(t-1) -> x(t)`, `x(t-2) -> x(t)`, and `x(t-1) -> x(t-2)`, then the minimal graph would have the following edges: `x(t-1) -> x(t)`, `x(t-2) -> x(t)`.
  - summary graph (obtained with the method `get_summary_graph`), obtained by collapsing the graph in time into a single node per variable (column name).
  - extended summary graph (obtained with the method `extend_graph`), obtained by extending backward and forward in time via the arguments `backward_steps` and `forward_steps` respectively.
- A new `from_adjacency_matrices` method to create a `TimeSeriesCausalGraph` from a dictionary of adjacency matrices where the keys are the time lags.
- Added security linting checks of source code using `bandit`.
- Improved documentation packaging and publishing.

## 0.1.0

- Initial release of the `cai-causal-graph` package with the `cai_causal_graph.causal_graph.CausalGraph` class and
  component classes: `cai_causal_graph.causal_graph.Skeleton`, `cai_causal_graph.graph_components.Node`, and
  `cai_causal_graph.graph_components.Edge`.
