# Utilities

## Identify Useful Subsets of a Causal Graph

The `cai-causal-graph` package allows for identifying useful subsets of a causal graph. These can be helpful in
interpreting a causal graph, but may also be used directly in some applications such as causal effect estimation. One
example of such a subset is the set of confounders between two variables.

### Identifying Confounders

The `cai-causal-graph` package implements the `cai_causal_graph.identify_utils.identify_confounders` utility function,
which allows you to identify the set of confounders between two variables in a directed acyclic graph (DAG).

Confounders are defined to be nodes in the causal graph that are (minimal) ancestors of both the source and destination
nodes. Note that, in this package, any parents of confounders (that do not have other direct causal paths to the
destination) are not returned as confounders themselves, even though they may be confounders in other definitions.
Hence, only **minimal** confounders are returned.

```python
from typing import List
from cai_causal_graph import CausalGraph
from cai_causal_graph.identify_utils import identify_confounders

# define a causal graph that is a DAG
cg = CausalGraph()
cg.add_edge('z', 'u')
cg.add_edge('u', 'x')
cg.add_edge('u', 'y')
cg.add_edge('x', 'y')

# compute confounders between source and destination; output: ['u']
confounder_variables: List[str] = identify_confounders(cg, node_1='x', node_2='y')
```

### Identifying Instruments

The `cai-causal-graph` package implements the `cai_causal_graph.identify_utils.identify_instruments` utility function,
which allows you to identify a list of potential instrumental variables for the causal effect of one node on another
node in a directed acyclic graph (DAG).

An instrumental variable for the causal effect of `source` on `destination` satisfies the following criteria:
    1. There is a causal effect between the `instrument` and the `source`.
    2. The `instrument` has a causal effect on the `destination` _only_ through the `source`.
    3. There is no confounding between the `instrument` and the `destination`.

```python
from typing import List
from cai_causal_graph import CausalGraph
from cai_causal_graph.identify_utils import identify_instruments

# define a causal graph that is a DAG
cg = CausalGraph()
cg.add_edge('z', 'x')
cg.add_edge('u', 'x')
cg.add_edge('u', 'y')
cg.add_edge('x', 'y')

# find the instruments between 'x' and 'y'; output: ['z']
instrumental_variables: List[str] = identify_instruments(cg, source='x', destination='y')
```

### Identifying Mediators

The `cai-causal-graph` package implements the `cai_causal_graph.identify_utils.identify_mediators` utility function,
which allows you to identify a list of potential mediator variables for the causal effect of one node on another node
in a directed acyclic graph (DAG).

    A mediator variable for the causal effect of `source` on `destination` satisfies the following criteria:
        1. There is a causal effect between the `source` and the `mediator`.
        2. There is a causal effect between the `mediator` and the `destination`.
        3. The `mediator` blocks all directed causal paths between the `source` and the `destination`.
        4. There is no directed causal path from any confounder between `source` and `destination` to the `mediator`.

```python
from typing import List
from cai_causal_graph import CausalGraph
from cai_causal_graph.identify_utils import identify_mediators

# define a causal graph
cg = CausalGraph()
cg.add_edge('x', 'm')
cg.add_edge('m', 'y')
cg.add_edge('u', 'x')
cg.add_edge('u', 'y')
cg.add_edge('x', 'y')

# find the mediators between 'x' and 'y'; output: ['m']
mediator_variables: List[str] = identify_mediators(cg, source='x', destination='y')
```

### Identifying Markov Boundary

The `cai-causal-graph` package implements the `cai_causal_graph.identify_utils.identify_markov_boundary` utility 
function, which allows you to identify the Markov boundary for a variable in a directed acyclic graph (DAG) or for a 
variable in an undirected graph.

The Markov boundary is defined as the minimal Markov blanket. The Markov blanket is defined as the set of variables
such that if you condition on them, it makes your variable of interest (`node` in this case) conditionally independent 
of all other variables. The Markov boundary is minimal meaning that you cannot drop any variables from it for the 
conditional independence condition to still hold.

For a DAG, provided as a `cai_causal_graph.causal_graph.CausalGraph` instance, the Markov boundary of a node is defined 
as its parents, its children, and the other parents of its children.

For an undirected graph, provided as a `cai_causal_graph.causal_graph.Skeleton` instance, the Markov boundary of a 
node is simply defined as its neighbors.

See https://en.wikipedia.org/wiki/Markov_blanket for further information. The code example below uses the graph 
from this site.

```python
from typing import List
from cai_causal_graph import CausalGraph, Skeleton
from cai_causal_graph.identify_utils import identify_markov_boundary

# define a causal graph
cg = CausalGraph()
cg.add_edge('u', 'b')
cg.add_edge('v', 'c')
cg.add_edge('b', 'a')  # 'b' is a parent of 'a'
cg.add_edge('c', 'a')  # 'c' is a parent of 'a'
cg.add_edge('a', 'd')  # 'd' is a child of 'a'
cg.add_edge('a', 'e')  # 'e' is a child of 'a'
cg.add_edge('w', 'f')
cg.add_edge('f', 'd')  # 'f' is a parent of 'd', which is a child of 'a'
cg.add_edge('d', 'x')
cg.add_edge('d', 'y')
cg.add_edge('g', 'e')  # 'g' is a parent of 'e', which is a child of 'a'
cg.add_edge('g', 'z')

# compute Markov boundary for node 'a'; output: ['b', 'c', 'd', 'e', 'f', 'g']
# parents: 'b' and 'c', children: 'd' and 'e', and other parents of children are 'f' and 'g'
# note the order may not match but the elements will be those six.
markov_boundary: List[str] = identify_markov_boundary(cg, node='a')

# use causal graph from above and get is skeleton
skeleton: Skeleton = cg.skeleton

# compute Markov boundary for node 'a'; output: ['b', 'c', 'd', 'e']
# as we have no directional information in the undirected skeleton, the neighbors of 'a' are returned.
# note the order may not match but the elements will be those four.
markov_boundary: List[str] = identify_markov_boundary(skeleton, node='a')
```

### Identifying Colliders

The `cai-causal-graph` package implements the `cai_causal_graph.identify_utils.identify_colliders` utility function,
which allows you to identify the list of potential collider nodes in the causal graph.

A collider is a node in the causal graph that is a child in a V-structure. A V-structure is a structure in the graph
where two nodes have a common child, and there is no direct edge between the two nodes.

```python
from typing import List
from cai_causal_graph import CausalGraph
from cai_causal_graph.identify_utils import identify_colliders

# define a causal graph
cg = CausalGraph()
cg.add_edge('x', 'm')
cg.add_edge('m', 'y')
cg.add_edge('x', 'y')

# find the colliders in the graph; output: ['y']
collider_variables: List[str] = identify_colliders(cg, unshielded_only=False)

# find the unshielded colliders in the graph; output: []
collider_variables: List[str] = identify_colliders(cg, unshielded_only=True)
```

The `unshielded_only` parameter is used to specify whether to return only unshielded colliders. If `unshielded_only` 
is set to `True`, only unshielded colliders are returned. If `unshielded_only` is set to `False`, all colliders are 
returned.
