# Changelog

## NEXT

- Added `TimeSeriesCausalGraph` class to deal with time series representations of causal graphs. The main additions with respect to the `CausalGraph` class are:
- each node is associated with a `variable_name` and a `time_lag` attributes.
- a new `from_adjacency_matrices` method to create a `TimeSeriesCausalGraph` from a dictionary of adjacency matrices where the keys are the time lags.

## 0.1.0

- Initial release of the `cai-causal-graph` package with the `CausalGraph` class and component classes: `Skeleton`, `Node`, and `Edge`.

