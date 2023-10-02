# Causal Graph

Causal graphs represent the underlying data-generation process of a given data set. They comprise a collection of
causal relationships between variables in the data set, which dictate how a given variable causally affects another
variable. As a result, causal graphs are one of the most fundamental building blocks in causality.

Causal graphs consist of _nodes_, i.e. variables, and _edges_, i.e. their causal relationships. A directed edge `->`
between two nodes `A` and `B` would imply that `A` is a causal driver of `B`, but not the other way around. A causal
graph is essentially a collection of such nodes and edges that aim to fully specify the data-generating process of
corresponding data.

There are several types of causal graphs, many of which can be captured by the 
`cai_causal_graph.causal_graph.CausalGraph` class of the `cai-causal-graph` package. One of the most prominent types is 
a _Directed Acyclic Graph_ (DAG), which only contains directed edges. Many downstream causality tasks require the use of 
a DAG, such as causal modeling, counterfactual explanations, causal effect estimation, among others. Sometimes, however, 
you may need to work with different causal graph types that can contain different edge types. See the section on Markov 
Equivalence Classes (MECs) at the end of this documentation page for more detailed information about additional causal 
graph types.

## Constructing a Causal Graph

### Adding Nodes and Edges

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

### Edge Types

Any edge added to causal graph will, by default, be a directed edge. It is, however, possible to specify different
edge types via the `edge_type` argument. For instance, you can add an undirected edge `A -- C`, as shown below, which 
can be resolved to either `A -> C` or `A <- C` in a later downstream task.

```python
from cai_causal_graph import EdgeType

# add an undirected edge between A and C
causal_graph.add_edge('A', 'C', edge_type=EdgeType.UNDIRECTED_EDGE)
```

These are the different edge types that are supported by the `cai_causal_graph.causal_graph.CausalGraph` class (these 
can be accessed via the `cai_causal_graph.type_definitions.EdgeType` enumeration):

- `EdgeType.DIRECTED_EDGE` (`->`)
- `EdgeType.UNDIRECTED_EDGE` (`--`)
- `EdgeType.BIDIRECTED_EDGE` (`<>`)
- `EdgeType.UNKNOWN_EDGE` (`oo`)
- `EdgeType.UNKNOWN_DIRECTED_EDGE` (`o>`)
- `EdgeType.UNKNOWN_UNDIRECTED_EDGE` (`o-`)

See the section on Markov Equivalence Classes at the end of this documentation page for more information on edge types.

## Interacting with a Causal Graph

### Accessing Nodes

The `cai_causal_graph.causal_graph.CausalGraph` class stores nodes via `cai_causal_graph.graph_components.Node` objects. 
It is possible to obtain a list of these `cai_causal_graph.graph_components.Node` objects by calling the 
`cai_causal_graph.causal_graph.CausalGraph.nodes` property.

```python
from typing import List

from cai_causal_graph.graph_components import Node

# query a list of nodes
list_of_nodes: List[Node] = causal_graph.nodes
```

If, instead of obtaining a list of `cai_causal_graph.graph_components.Node` objects, you wish to obtain a list of 
string node identifiers, you can call the `cai_causal_graph.causal_graph.CausalGraph.get_node_names` method:

```python
# obtain a list of node identifiers
node_names: List[str] = causal_graph.get_node_names()
```

It is also possible to query a specific node using its string identifier by means of the
`cai_causal_graph.causal_graph.CausalGraph.get_node` and `cai_causal_graph.causal_graph.CausalGraph.get_nodes` methods. 
The former method returns a single node, while the latter method returns a list of nodes. More concretely, the
`cai_causal_graph.causal_graph.CausalGraph.get_nodes` method accepts a single node identifier (yielding a list 
containing only one `cai_causal_graph.graph_components.Node` object), a list of node identifiers (yielding a list 
containing the corresponding `cai_causal_graph.graph_components.Node` objects), or the default `None` (yielding a 
list of all nodes).

```python
# query a specific node
node_object: Node = causal_graph.get_node(identifier='node_1')

# query a specific node with get_nodes
node_objects: List[Node] = causal_graph.get_nodes(identifier='node_1')

# query a list of nodes with get_nodes
node_objects: List[Node] = causal_graph.get_nodes(identifier=['node_1', 'node_2'])

# query all nodes with get_nodes; equivalent to the nodes property
node_objects: List[Node] = causal_graph.get_nodes()
```

Some nodes can be classified as _inputs_ or _outputs_ which means that they either have no incoming edges or no
outgoing edges, respectively. Inputs can be thought of as source nodes and outputs can be thought of as sink nodes. A
list of such nodes within a causal graph can be obtained via the 
`cai_causal_graph.causal_graph.CausalGraph.get_inputs` and 
`cai_causal_graph.causal_graph.CausalGraph.get_outputs` methods:

```python
# get a list of inputs; these have no incoming edges
list_of_inputs: List[Node] = causal_graph.get_inputs()

# get a list of outputs; these have no outgoing edges
list_of_outputs: List[Node] = causal_graph.get_outputs()
```

You can check for the existence of a node using the `cai_causal_graph.causal_graph.CausalGraph.node_exists` method:

```python
# returns True if the node exists and False otherwise
causal_graph.node_exists(identifier='node_1')
```

### Accessing Edges

The `cai_causal_graph.causal_graph.CausalGraph` class stores edges via `cai_causal_graph.graph_components.Edge` objects. 
It is possible to obtain a list of these `cai_causal_graph.graph_components.Edge` objects by calling the 
`cai_causal_graph.causal_graph.CausalGraph.edges` property.

```python
from typing import List

from cai_causal_graph.graph_components import Edge

# query a list of edges
list_of_edges: List[Edge] = causal_graph.edges
```

It is also possible to query a specific edge using the string identifier of its source node and/or its destination node 
by means of the `cai_causal_graph.causal_graph.CausalGraph.get_edge` and 
`cai_causal_graph.causal_graph.CausalGraph.get_edges` methods. The former method returns a single edge, while the latter 
method returns a list of edges. If the `source` argument of 
`cai_causal_graph.causal_graph.CausalGraph.get_edges` is not provided, i.e. it is `None`, then a list of edges 
connecting to the `destination` node are returned, and vice versa. If neither `source` nor `destination` are 
provided, a list of all edges are returned.

```python
# query a specific edge
edge_object: Edge = causal_graph.get_edge(source='node_1', destination='node_2')

# query all edges originating from node_1
node_1_edges: List[Edge] = causal_graph.get_edges(source='node_1')

# query all edges terminating at node_2
node_2_edges: List[Edge] = causal_graph.get_edges(destination='node_1')

# query all edges with get_edges; equivalent to the edges property
edge_objects: List[Edge] = causal_graph.get_edges()
```

Alternatively, you can also provide a tuple of node identifiers via the 
`cai_causal_graph.causal_graph.CausalGraph.get_edge_by_pair` method to query an edge:

```python
# query a specific edge
edge_object: Edge = causal_graph.get_edge_by_pair(pair=('node_1', 'node_2'))
```

You can check for the existence of an edge using the `cai_causal_graph.causal_graph.CausalGraph.edge_exists` method:

```python
# returns True if the edge exists and False otherwise
causal_graph.node_exists(source='node_1', destination='node_2')
```

Importantly, each of the above queries can also include the `edge_type` of the relevant edge. By default, the
`edge_type` argument of the above methods is `None`, which means the edge is queried no matter its type. However, in
some settings you may wish to further specify the `edge_type` (see the Edge Types section above for more information
on the available types as defined by the `cai_causal_graph.type_definitions.EdgeType` enumeration). If the edge does not 
exist with that type (note that it may exist with a different type), then an 
`cai_causal_graph.exceptions.CausalGraphErrors.EdgeDoesNotExistError` is raised.

```python
from cai_causal_graph import EdgeType

# query for the edge knowing that it is undirected
edge_object: Edge = causal_graph.get_edge(
    source='node_1', destination='node_2', edge_type=EdgeType.UNDIRECTED_EDGE
)
```

Lastly, while the `cai_causal_graph.causal_graph.CausalGraph.get_edges` method returns all edges no matter the type, 
it is possible to only obtain a list of edges that only have a certain type. The following methods are available to 
do this:

- `cai_causal_graph.causal_graph.CausalGraph.get_directed_edges`
- `cai_causal_graph.causal_graph.CausalGraph.get_undirected_edges`
- `cai_causal_graph.causal_graph.CausalGraph.get_bidirected_edges`
- `cai_causal_graph.causal_graph.CausalGraph.get_unknown_edges`
- `cai_causal_graph.causal_graph.CausalGraph.get_unknown_directed_edges`
- `cai_causal_graph.causal_graph.CausalGraph.get_unknown_undirected_edges`

### Manipulating Nodes and Edges

You can delete a node from the `cai_causal_graph.causal_graph.CausalGraph` object by calling the 
`cai_causal_graph.causal_graph.CausalGraph.delete_node` method. This also removes any edges that were previously 
connecting that node to any other nodes. There is also the `cai_causal_graph.causal_graph.CausalGraph.remove_node` 
method, which does exactly the same thing and only exists because the words "delete" and "remove" are often used 
interchangeably.

```python
# delete the node and all incoming / outgoing edges; does the same as remove_node
causal_graph.delete_node(identifier='node_1')
```

Sometimes you way wish to replace a node, or simply rename it to something else. This can be done using the
`cai_causal_graph.causal_graph.CausalGraph.replace_node` method:

```python
# replace a node with a new one
causal_graph.replace_node(node_id='node_1', new_node_id='node_new')
```

Similar to deleting a node, you can delete an edge using the `cai_causal_graph.causal_graph.CausalGraph.delete_edge` 
method (which is the same as the `cai_causal_graph.causal_graph.CausalGraph.remove_edge` method).

```python
# delete the edge; does the same as remove_edge
causal_graph.delete_edge(source='node_1', destination='node_2')
```

## Working with other graph formats

### Skeleton

The `cai_causal_graph.causal_graph.CausalGraph` class has a `cai_causal_graph.causal_graph.CausalGraph.skeleton` 
property that returns the skeleton of the underlying causal graph (as a `cai_causal_graph.causal_graph.Skeleton` 
object), which contains the same nodes and edges but only has undirected edges. For instance, the skeleton of 
`A -> B <> C` would be `A -- B -- C`.

```python
from cai_causal_graph import Skeleton

# query the skeleton of the causal graph
skeleton_object: Skeleton = causal_graph.skeleton
```

### Adjacency Matrix

It is also possible to query the adjacency matrix $A$ of the underlying causal graph. This is a $p \times p$ matrix,
where $p$ is the number of nodes, containing elements $A_{ij} = 1$ if there is an edge from node $i$ to node $j$. If
there is an undirected edge between nodes $i$ and $j$, then $A_{ij} = A_{ji} = 1$. A fully undirected causal graph will
therefore have a symmetric adjacency matrix. You can query the adjacency matrix using the 
`cai_causal_graph.causal_graph.CausalGraph.adjacency_matrix` property, as shown below. Note that adjacency matrices are 
only defined for causal graphs with directed or undirected edges; if the `cai_causal_graph.causal_graph.CausalGraph` 
object contains any other edge types, this will raise an error.

```python
import numpy

# obtain the adjacency matrix
adjacency: numpy.ndarray = causal_graph.adjacency_matrix
```

The `cai_causal_graph.causal_graph.CausalGraph.to_numpy` method is equivalent to querying the adjacency matrix, but 
also returns a list of node names:

```python
from typing import List, Tuple

import numpy

# obtain the adjacency matrix
adjacency, node_names: Tuple[numpy.ndarray, List[str]] = causal_graph.to_numpy()
```

Naturally, you can also construct a `causal_graph.causal_graph.CausalGraph` class from an adjacency matrix, by means 
of the `causal_graph.causal_graph.CausalGraph` `.from_adjacency_matrix` method. Note that this method allows you to 
pass a list of node names that can be used to construct node identifiers. If the `node_names` argument is not 
provided, the default node names will instead be `"node_x"`, where x is between 1 and $p$ (the number of nodes).

```python
from cai_causal_graph import CausalGraph

# construct a causal graph from an adjacency matrix
causal_graph: CausalGraph = CausalGraph.from_adjacency_matrix(adjacency, node_names)
```

Open-source packages often rely on `networkx` for their causal graph objects. Specifically, the `networkx.DiGraph` class
which can represent DAGs, i.e. an acyclic graph with only directed edges. It is straightforward to transform a
`cai_causal_graph.causal_graph.CausalGraph` instance to a `networkx.DiGraph` (or a `networkx.Graph`) instance, and vice 
versa, as shown below.

```python
import networkx

# CausalGraph to networkx.DiGraph or networkx.Graph depending on edge types in the CausalGraph instance
networkx_digraph: networkx.DiGraph = causal_graph.to_networkx()  # if graph is fully directed
networkx_graph: networkx.Graph = causal_graph.to_networkx()  # if graph is fully undirected

# networkx.DiGraph to CausalGraph
causal_graph: CausalGraph = CausalGraph.from_networkx(networkx_digraph)

# networkx.Graph to CausalGraph
causal_graph: CausalGraph = CausalGraph.from_networkx(networkx_graph)
```

Open-source packages also utilize a Graph Modelling Language (GML) string to represent a mixed graph. It is possible to 
convert to and from GML strings as shown below.

```python
# CausalGraph to GML string
gml_graph: str = causal_graph.to_gml_string()

# GML string to CausalGraph
causal_graph: CausalGraph = CausalGraph.from_gml_string(gml_graph)
```

Lastly, the `cai_causal_graph.causal_graph.CausalGraph` object is serializable and can therefore be converted to / from a 
dictionary:

```python
# CausalGraph to dictionary
causal_graph_dict: dict = causal_graph.to_dict()

# Dictionary to CausalGraph
causal_graph: CausalGraph = CausalGraph.from_dict(causal_graph_dict)
```

## Identify Useful Subsets of a Causal Graph

The `cai-causal-graph` package allows for identifying useful subsets of a causal graph. These can be helpful in
interpreting a causal graph, but may also be used directly in some applications such as causal effect estimation. One
example of such a subset is the set of confounders between two variables.

### Identifying Confounders

The `cai-causal-graph` package implements the `cai_causal_graph.identify_utils.identify_confounders` utility function,
which allows you to identify the set of confounders between two variables.

Confounders are defined to be nodes in the causal graph that are (minimal) ancestors of both the source and destination
nodes. Note that, in this package, any parents of confounders (that do not have other direct causal paths to the
destination) are not returned as confounders themselves, even though they may be confounders in other definitions.
Hence, only **minimal** confounders are returned.

```python
from typing import List
from cai_causal_graph import CausalGraph
from cai_causal_graph.identify_utils import identify_confounders

# define a causal graph
cg = CausalGraph()
cg.add_edge('z', 'u')
cg.add_edge('u', 'x')
cg.add_edge('u', 'y')
cg.add_edge('x', 'y')

# compute confounders between source and destination; output: ['u']
confounders_list: List[str] = identify_confounders(cg, source='x', destination='y')
```
