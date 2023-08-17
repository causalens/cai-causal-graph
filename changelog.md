# Changelog

## NEXT

- Added `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` class to deal with time series representations of causal graphs. The main additions with respect to the `cai_causal_graph.causal_graph.CausalGraph` class are:
    - each node is associated with a `variable_name` and a `time_lag` attribute.
    - a new `from_adjacency_matrices` method to create a `TimeSeriesCausalGraph` from a dictionary of adjacency matrices where the keys are the time lags.
- Added security linting checks of source code using `bandit`.

## 0.1.0

- Initial release of the `cai-causal-graph` package with the `cai_causal_graph.causal_graph.CausalGraph` class and component classes: `cai_causal_graph.causal_graph.Skeleton`, `cai_causal_graph.graph_components.Node`, and `cai_causal_graph.graph_components.Edge`.
