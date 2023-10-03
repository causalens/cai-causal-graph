# Causal Graph Utilities

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
confounders_list: List[str] = identify_confounders(cg, node_1='x', node_2='y')
```
