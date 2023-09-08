# Time Series Causal Graph
The class `cai_causal_graph.causal_graph.TimeSeriesCausalGraph`, inheriting directly from `cai_causal_graph.causal_graph.CausalGraph`,
extends causal graphs to time series data.

For the theoretical background of causality for time series, please refer to Chapter 10 of _Peters, J., Janzing, D. and Schölkopf, B., 2017. Elements of causal inference: foundations and learning algorithms (p. 288). The MIT Press_. 

The main addition of `cai_causal_graph.causal_graph.TimeSeriesCausalGraph`
lies in the type of nodes it supports.
The new time series node is a new class `cai_causal_graph.causal_graph.TimeSeriesNode`
inheriting from the corresponding `cai_causal_graph.causal_graph.Node` all the original attributes:
- identifier
- variable_type

The node in a time series causal graph, `cai_causal_graph.causal_graph.TimeSeriesCausalGraph`,
will have additional metadata that gives the time information of the node together with the variable name.
The two additional metadata are:
- `cai_causal_graph.type_definitions.TIME_LAG`: the time difference with respect to the reference time 0.
- `cai_causal_graph.type_definitions.VARIABLE_NAME`: the name of the variable (without the lag information).

Therefore, the time series node `cai_causal_graph.causal_graph.TimeSeriesCausalGraph` can be initialized as follows.

```python
from cai_causal_graph import TimeSeriesNode

# Via identifier
ts_node = TimeSeriesNode(identifier='X lag(n=1)')

# Via time lag and variable name
ts_node = TimeSeriesNode(time_lag=1, variable_name='X')
```

## Examples of time series causal graphs
The following examples have been drawn from _Peters, J., Janzing, D. and Schölkopf, B., 2017. Elements of causal inference: foundations and learning algorithms (p. 288). The MIT Press_.

Example of a time series with no instantaneous effects.
![ts_no_instantaneous_effects](images/ts_no_instantaneous_effects.png)

Example of a time series with instantaneous effects.
![ts_instantaneous_effects](images/ts_instantaneous_effects.png)

Summary graph of the full time graphs above.
![summary_graph](images/summary_graph.png)

## Initialization

### Direct initialization
You can initialize the time series causal graph directly.
```python
from cai_causal_graph import EdgeType, TimeSeriesCausalGraph

ts_cg = TimeSeriesCausalGraph()
ts_cg.add_edge('X1 lag(n=1)', 'X1', edge_type=EdgeType.DIRECTED_EDGE)
ts_cg.add_edge('X2 lag(n=1)', 'X2', edge_type=EdgeType.DIRECTED_EDGE)
ts_cg.add_edge('X1', 'X3', edge_type=EdgeType.DIRECTED_EDGE)
```

### From `cai_causal_graph.causal_graph.CausalGraph``
Alternatively, you can initialize a `cai_causal_graph.causal_graph.TimeSeriesCausalGraph` instance
from a `cai_causal_graph.causal_graph.CausalGraph`,
converting a causal graph from a single time step into a time series causal graph.

```python
from cai_causal_graph import CausalGraph, EdgeType, TimeSeriesCausalGraph

cg = CausalGraph()
cg.add_edge('X1 lag(n=1)', 'X1', edge_type=EdgeType.DIRECTED_EDGE)
cg.add_edge('X2 lag(n=1)', 'X2', edge_type=EdgeType.DIRECTED_EDGE)
cg.add_edge('X1', 'X3', edge_type=EdgeType.DIRECTED_EDGE)

ts_cg = TimeSeriesCausalGraph.from_causal_graph(cg)
```

The time series causal graph will have the same nodes and edges as the causal graph,
but will be aware of the time information so 'X1 lag(n=1)' and 'X1' represent the same
variable but at different times.

### From an adjacency matrix
You can initialize a `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` instance from an adjacency matrix
and optionally a list of node names.

```python
import numpy
from cai_causal_graph import TimeSeriesCausalGraph

# The adjacency matrix should be a squared binary numpy array
adjacency_matrix: numpy.ndarray

# Simply via the adjacency matrix
ts_cg = TimeSeriesCausalGraph.from_adjacency_matrix(adjacency=adjacency_matrix)

# Also specifying the node names
ts_cg = TimeSeriesCausalGraph.from_adjacency_matrix(adjacency=adjacency_matrix, node_names=['X1 lag(n=1)', 'X1', 'X2 lag(n=1)', 'X2', 'X3'])
```

### From multiple adjacency matrices
You can initialize a `cai_causal_graph.time_series_causal_graph.TimeSeriesCausalGraph` instance
from a dictionary of adjacency matrices. Keys are the time deltas.
For example, the adjacency matrix with time delta -1 is stored in adjacency_matrices[-1] as would correspond to X-1 -> X,
where X is the set of nodes.

Example:
adjacency_matrices = {
...     -2: numpy.array([[0, 0, 0], [1, 0, 0], [0, 0, 1]]),
...     -1: numpy.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
...     0: numpy.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]]),
... }


```python
from cai_causal_graph import TimeSeriesCausalGraph
adjacency_matrices = {
    -2: numpy.array([[0, 0, 0], [1, 0, 0], [0, 0, 1]]),
    -1: numpy.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
    0: numpy.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]]),
}

# Simply via the adjacency matrices
ts_cg = TimeSeriesCausalGraph.from_adjacency_matrices(adjacency_matrices=adjacency_matrices)

# Also specifying the variable names
ts_cg = TimeSeriesCausalGraph.from_adjacency_matrices(adjacency_matrices=adjacency_matrices, variable_names=['X', 'Y', 'Z'])
```

Let the nodes of this example be X,Y,Z. The corresponding edges are the following:
edges = [
...     (Z-2, Y),
...     (X-1, X),
...     (Y-1, X),
...     (Y-1, Y),
...     (Z-1, Z),
...     (X, Y),
...     (X, Z),
...     (Y, Z),
... ]

## Minimal graph
The minimal graph is the graph with the minimal number of edges that is equivalent to the original graph.
In other words, it is a graph that has no edges whose destination is not time delta 0.


```python
from cai_causal_graph import TimeSeriesCausalGraph

ts_cg: TimeSeriesCausalGraph

# Get the minimal graph
minimal_graph = ts_cg.get_minimal_graph()

# Check whether a graph is in its minimal form
# Returns True if the graph is a minimal graph, False otherwise
is_minimal = ts_cg.is_minimal_graph()
```

## Summary graph
You can collapse the graph in time into a single node per variable (column name).
This can become cyclic and bi-directed as X(t-1) -> Y and Y(t-1) -> X would become X <-> Y.
Note that the summary graph is a `cai_causal_graph.causal_graph.CausalGraph` object.

```python
from cai_causal_graph import TimeSeriesCausalGraph

ts_cg: TimeSeriesCausalGraph

summary_graph = ts_cg.get_summary_graph()
```

## Example
Time series causal graph.
X1(t-1) -> X1(t)
X2(t-1) -> X2(t)
X1(t-1) -> X2(t-1)
X1(t) -> X2(t)
X1(t) -> X1(t+1)
X2(t) -> X2(t+1)
X1(t+1) -> X2(t+1)
![ts_cg](images/ts_cg.png)

Minimal graph.
X1(t-1) -> X1(t)
X2(t-1) -> X2(t)
X1(t) -> X2(t)
![ts_minimal_graph](images/ts_minimal_graph.png)

Summary graph.
X1 -> X2
![ts_summary_graph](images/ts_summary_graph.png)

Extended graph with `backward_steps` = 2 and `forward_steps` = 0.
X1(t-2) -> X1(t-1)
X2(t-2) -> X2(t-1)
X1(t-2) -> X2(t-2)
X1(t-1) -> X1(t)
X2(t-1) -> X2(t)
X1(t-1) -> X2(t-1)
X1(t) -> X2(t)
![ts_cg](images/ts_extended_graph.png)


## Extended graph
You can extend the graph in time by adding nodes for each variable at each time step from `backward_steps` to
`forward_steps`. If a backward step of n is specified, it means that the graph will be extended in order to
include nodes back to time -n and all the nodes connected to them as specified by the minimal graph.
If both `backward_steps` and `forward_steps` are None, the original graph is returned.

```python
from cai_causal_graph import TimeSeriesCausalGraph

ts_cg: TimeSeriesCausalGraph
backward_steps = 2
forward_steps = 3

extended_graph = ts_cg.extend_graph(backward_steps=backward_steps, forward_steps=forward_steps)
```

## Markov Equivalence Classes
Certain causal relationships yield the same conditional independencies and are therefore indistinguishable from each other with only observational data. The set of such causal relationships is called the Markov Equivalence Class (MEC) for a particular set of nodes. Most causal discovery methods that use observational data are only able to return the MEC of a set of nodes and not the full directed acyclic graph (DAG).

One prominent representation of MECs is a Completed Partially Directed Acyclic Graph (CPDAG), which only contains directed (->) and undirected (--) edges. In this case, an undirected edge simply implies that a causal relationship can point either way, i.e. A -- B can be resolved to either A -> B or A <- B

DAGs are directed and contain only directed edges. 

Finally, a Partial Ancestral Graph (PAG) can encode all the information that CPDAGs can, but also provides more detailed information about the relationships between variables, such as including whether a latent confounder is likely to exist or selection bias is likely to be present. 

Specifically, in addition to directed edges (A -> B or A <- B) and undirected edges (A -- B), PAGs represent an equivalence class of MAGs (Maximal Ancestor Graphs) that may also contain bi-directed edges A <-> B, which implies that there is a latent confounder between the respective variables. They introduce the additional "wild-card" or "circle" edges A -o B, which can either be a directed or undirected arrow head, i.e. A -o B can be resolved to A -- B or A -> B.

Similar to a CPDAG, a PAG can represent a number of DAGs. Therefore, PAGs, CPDAGs and DAGs can be thought of in a hierarchical way.

MAGs and PAGs retain the pairwise Markov property of DAGs - every missing edges corresponds to a statement of conditional independence amongst the observed variables. Contrary to DAGs, however, the presence of an edge does not mean necessarily that two nodes are adjacent in the (presumed) true DAG - which may include unobserved variables. Rather, it simply means that a conditional set that separates the two nodes could not be found amongst the observed variables.

`cai_causal_graph.causal_graph.TimeSeriesCausalGraph` also supports Markov equivalence classes, as in the following example with a CPDAG.

```python
from cai_causal_graph import EdgeType, TimeSeriesCausalGraph

ts_cg = TimeSeriesCausalGraph()

ts_cg.add_edge('X1 lag(n=1)', 'X1', edge_type=EdgeType.BIDIRECTED_EDGE)
ts_cg.add_edge('X1 lag(n=1)', 'X2', edge_type=EdgeType.UNKNOWN_UNDIRECTED_EDGE)
ts_cg.add_edge('X1', 'X2', edge_type=EdgeType.DIRECTED_EDGE)
```

## Query the nodes and the variables
You can extract all the variable names from the variable nodes of the time series graph.

```python
from cai_causal_graph import TimeSeriesCausalGraph

ts_cg: TimeSeriesCausalGraph

# Return a list of variable names from a list of node names
node_names = ['X', 'X lag(n=1)', 'Y', 'Z lag(n=2)']
ts_cg.get_variable_names_from_node_names()
>>> ['X', 'Y', 'Z']
```

Conversely, you can extract the time series nodes in the graph from a given list of identifiers.

```python
from cai_causal_graph import TimeSeriesCausalGraph

ts_cg: TimeSeriesCausalGraph

# A single node
variable: TimeSeriesNode = ts_cg.get_nodes(identifier='X lag(n=1)')

# Multiple nodes
variables = ts_cg.get_nodes(identifier=['X', 'X lag(n=1)'])
```

## Add nodes
You can add time series nodes and the corresponding meta data will be automatically populated.

```python
from cai_causal_graph import TimeSeriesCausalGraph, TimeSeriesNode

ts_cg: TimeSeriesCausalGraph

# Via a node object
new_node: TimeSeriesNode
ts_cg.add_node(new_node)
# Via identifier
ts_cg.add_node(identifier='X lag(n=3)')
# Via variable name and time lag
ts_cg.add_node(variable_name='X', time_lag=3)

# Variable type can also be specified
ts_cg.add_node(identifier='X lag(n=3)', variable_type=NodeVariableType.UNSPECIFIED)
```

## Add edges
You can add time series edges and the corresponding meta data will be automatically populated.

```python
from cai_causal_graph import EdgeType, TimeSeriesCausalGraph
from cai_causal_graph.graph_components import Edge

ts_cg: TimeSeriesCausalGraph

# Via a new edge object
new_edge: Edge
ts_cg.add_edge(new_edge)
# Via identifier (the edge type can be specified if desired)
ts_cg.add_edge(source='X lag(n=3)', destination='Y lag(n=3)', edge_type=EdgeType.DIRECTED_EDGE)

# Add edge by pair (the edge type can be specified if desired)
ts_cg.add_edge_by_pair(pair=('X lag(n=2)', 'Y lag(n=2)'), edge_type=EdgeType.DIRECTED_EDGE)

# Add multiple edges by specifying tuples of source and destination node identifiers and with default setup
ts_cg.add_edges_from(pairs=[('X lag(n=2)', 'Y lag(n=2)'), ('X lag(n=3)', 'Y lag(n=3)')])
```

## Replace nodes
Replace a node in the time series graph.

```python
from cai_causal_graph import TimeSeriesCausalGraph

ts_cg: TimeSeriesCausalGraph

# Via identifier
ts_cg.replace_node(node_id='X lag(n=3)', new_node_id='Y lag(n=3)')
# Via variable name and time lag
ts_cg.replace_node(node_id='X lag(n=3)', time_lag=3, variable_name='Y')

# Variable type can also be specified
ts_cg.replace_node(node_id='X lag(n=3)', new_node_id='Y lag(n=3)', variable_type=NodeVariableType.UNSPECIFIED)
```

## Delete nodes and edges
You can delete nodes and edges from the time series graph.

```python
from cai_causal_graph import EdgeType, TimeSeriesCausalGraph

ts_cg: TimeSeriesCausalGraph

# Delete node
ts_cg.delete_node(identifier='X lag(n=3)')

# Delete edge (the edge type can be specified if desired)
ts_cg.delete_edge(source='X lag(n=3)', destination='Z lag(n=3)', edge_type=EdgeType.DIRECTED_EDGE)

# Delete edge from pair (the edge type can be specified if desired)
ts_cg.remove_edge_by_pair(pair=('X lag(n=3)', 'Z lag(n=3)'), edge_type=EdgeType.DIRECTED_EDGE)
```

## Main properties
### Adjacency_matrices
Return the adjacency matrix dictionary of the minimal causal graph.
The keys are the time deltas and the values are the adjacency matrices.

```python
from cai_causal_graph import TimeSeriesCausalGraph

ts_cg: TimeSeriesCausalGraph

adjacency_matrices = ts_cg.adjacency_matrices
```

### maxlag
Return the autoregressive order of the time series graph, i.e. the maximum lag of the nodes in the minimal graph.

```python
from cai_causal_graph import TimeSeriesCausalGraph

ts_cg: TimeSeriesCausalGraph

maxlag = ts_cg.maxlag
```

### Variables
Return the list of variable identifiers in the time series graph.
Variables differ from nodes in that they do not contain the lag.
For example, if the graph contains the node "X1 lag(n=2)", the variable is "X1".

```python
from cai_causal_graph import TimeSeriesCausalGraph

ts_cg: TimeSeriesCausalGraph

variables = ts_cg.variables
```
### Stationarity

It may be useful to check whether the time series graph is stationary, i.e. its edges are not
dependent on the corresponding time lags. In other words, the graph is the same for all time lags.
Stationarity is a useful concept for future prediction: if the graph is stationary, it can be used to predict the future
for any time lag.
Conversely, if the graph is not stationary, it can only be used to predict the future for the specified time lags.
Thus, in a stationary graph, if the edge `X lag(n=1) -> Y lag(n=1)` exists, also the edge `X lag(n=2) -> Y lag(n=2)` 
must be present if the nodes `X lag(n=2)` and `Y lag(n=2)` are also present.

You can check whether the graph is stationary by calling the `is_stationary_gragh` method.

```python

from cai_causal_graph import TimeSeriesCausalGraph

ts_cg: TimeSeriesCausalGraph

is_stationary = ts_cg.is_stationary_graph()
```

Moreover, you can make it stationary by calling the `get_stationary_graph` method as follows:

```python

from cai_causal_graph import TimeSeriesCausalGraph

ts_cg: TimeSeriesCausalGraph

stationary_graph = ts_cg.get_stationary_graph()
```

It is important to note that the minimal graph is not necessarily stationary. This does not mean that the underlying process is not stationary. 
It only means that the graph is missing at least one of the corresponding edges in time. Please refer to the image example for a visualization of a minimal graph that is not stationary.

## Other methods
For all the other base properties and methods (e.g. how to query nodes and edges), please refer to the documentation of `cai_causal_graph.causal_graph.CausalGraph`,
from which `cai_causal_graph.causal_graph.TimeSeriesCausalGraph` inherits all the functionalities.