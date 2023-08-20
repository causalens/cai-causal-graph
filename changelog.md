# Changelog

## NEXT

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

## 0.1.0

- Initial release of the `cai-causal-graph` package with the `cai_causal_graph.causal_graph.CausalGraph` class and component classes: `cai_causal_graph.causal_graph.Skeleton`, `cai_causal_graph.graph_components.Node`, and `cai_causal_graph.graph_components.Edge`.
