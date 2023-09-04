# Quickstart

The causal graph fundamentally consists of _nodes_, i.e. variables, and _edges_, i.e. their relationships. For instance,
a causal graph containing a directed edge `->` between node `A` and node `B` would imply that variable `A` is a causal
driver of variable `B`, but not the other way around.

You can easily add nodes and edges to the `cai_causal_graph.causal_graph.CausalGraph` class by using the 
`cai_causal_graph.causal_graph.CausalGraph.add_node` / 
`cai_causal_graph.causal_graph.CausalGraph.add_nodes_from` and `cai_causal_graph.causal_graph.CausalGraph.add_edge` 
methods, as shown below.

```python
from cai_causal_graph import CausalGraph

# construct the causal graph object
causal_graph = CausalGraph()

# add a single node to the causal graph
causal_graph.add_node('A')

# add several nodes at once to the causal graph
causal_graph.add_nodes_from(['B', 'C', 'D'])

# add edges to the causal graph
causal_graph.add_edge('A', 'B')  # this adds a directed edge (i.e., an edge from A to B) by default
causal_graph.add_edge('B', 'E')  # if the node does not exist, it gets added automatically
```

Any edge added to causal graph will, by default, be a directed edge. It is, however, possible to specify different
edge types via the `edge_type` argument. For a full list of edge types, see
`cai_causal_graph.type_definitions.EdgeType`. For instance, you can add an undirected edge `A -- C`, as shown below.

```python
from cai_causal_graph import EdgeType

# add an undirected edge between A and C
causal_graph.add_edge('A', 'C', edge_type=EdgeType.UNDIRECTED_EDGE)
```

## Time Series Causal Graphs

The `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` class extends the
`cai_causal_graph.causal_graph.CausalGraph` class to represent time series causal graphs. The main 
differences with respect to the `cai_causal_graph.causal_graph.CausalGraph` class are:
- each node is associated with a `variable_name` and a `time_lag` attribute.
- a new `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph.from_adjacency_matrices` method is provided 
  to create a `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` from a dictionary of adjacency 
  matrices where the keys are the time lags.

`cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` is aware of the time lags of the variables and can 
be extended backwards and forward in time.

If you want to convert a `cai_causal_graph.causal_graph.CausalGraph` to a 
`cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph`, you can use the 
`cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph.from_causal_graph` method as shown below.

```python
from cai_causal_graph import CausalGraph, TimeSeriesCausalGraph

# create the causal graph with lagged nodes to show the difference
causal_graph = CausalGraph()

# add edges to the causal graph
causal_graph.add_edge('X1 lag(n=1)', 'X2')
causal_graph.add_edge('X2 lag(n=1)', 'X2')

# convert the causal graph to a time series causal graph
time_series_causal_graph = TimeSeriesCausalGraph.from_causal_graph(causal_graph)
```

The difference between the two graphs is that the `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` is 
now aware of the time lags of the nodes and understands that `'X1 lag(n=1)'` and `'X1'` refer to the same variable.

Moreover, `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` provides the capability to extend the 
minimal graph backwards and forward in time using the 
`cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph.extend_graph` method. For instance, if you want to 
extend the graph backwards in time to time delta -2 (backwards) you can do the following:

```python
# Using the time causal graph defined above.
time_series_causal_graph.extend_graph(backward_steps=2)

# the graph now contains the following nodes
# X1 lag(n=1), X1 lag(n=2), X2 lag(n=1), X2 lag(n=2), X2

# and the following edges
# X1 lag(n=1) -> X2, X2 lag(n=1) -> X2, X1 lag(n=2) -> X2 lag(n=1), X2 lag(n=2) -> X2 lag(n=1)
```
