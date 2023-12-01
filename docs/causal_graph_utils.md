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

TODO
