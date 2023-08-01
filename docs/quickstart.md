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
`cai_causal_graph.type_definitions.EDGE_T`. For instance, you can add an undirected edge `A -- C`, as shown below.

```python
from cai_causal_graph import EDGE_T

# add an undirected edge between A and C
causal_graph.add_edge('A', 'C', edge_type=EDGE_T.UNDIRECTED_EDGE)
```
