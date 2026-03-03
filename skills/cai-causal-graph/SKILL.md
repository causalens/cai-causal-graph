---
name: cai-causal-graph
description: |
  Define, manipulate, and serialize causal graph structures (DAG, CPDAG, MAG, PAG) using
  cai-causal-graph. Provides CausalGraph, Node, Edge, Skeleton classes with variable type
  support (CONTINUOUS, BINARY, MULTICLASS, ORDINAL). Use when user imports from
  cai_causal_graph, works with causal graphs, needs graph conversion (NetworkX, adjacency
  matrix, dict), or mentions DAG/CPDAG/PAG/MAG representations. Do NOT trigger for graph
  neural networks or knowledge graphs unrelated to causality.
license: Apache-2.0
metadata:
  author: causalens
  version: "0.5.14"
  sdk_schema_version: "0.1.0"
  sdk_dependencies: "cai-causal-graph>=0.5.0"
allowed-tools: Bash(python:*) Read Write Edit
---

# CAI Causal Graph

Python library for defining, manipulating, and serializing causal graph structures (DAG, CPDAG, MAG, PAG).

## Quick Reference

| Class | Module | Purpose |
|---|---|---|
| `CausalGraph` | `cai_causal_graph.causal_graph` | Core graph: nodes + typed edges, DAG validation |
| `Skeleton` | `cai_causal_graph.causal_graph` | Undirected view of a `CausalGraph` (no edge direction) |
| `TimeSeriesCausalGraph` | `cai_causal_graph.time_series_causal_graph` | Subclass of `CausalGraph` for lagged time-series nodes |
| `Node` | `cai_causal_graph.graph_components` | Single node with `identifier` and `variable_type` |
| `Edge` | `cai_causal_graph.graph_components` | Single edge with `source`, `destination`, `edge_type` |
| `EdgeType` | `cai_causal_graph.type_definitions` | Enum of all valid edge types |
| `NodeVariableType` | `cai_causal_graph.type_definitions` | Enum of variable types for nodes |
| `EdgeConstraint` | `cai_causal_graph.type_definitions` | Enum for domain-knowledge constraints |

All public classes are re-exported from `cai_causal_graph` directly:

```python
from cai_causal_graph import CausalGraph, TimeSeriesCausalGraph, EdgeType, NodeVariableType, Skeleton
```

## Graph Types

Choose the graph representation based on what the edges mean:

| Need | Graph Type | Edge constraint |
|---|---|---|
| Fully directed, no cycles | DAG | All edges `->` |
| Partially directed (discovery output) | CPDAG | Mix of `->` and `--` |
| Latent confounders allowed | MAG | Has `<>` bidirected edges |
| Maximal uncertainty from FCI | PAG | Has `o>`, `o-`, `oo` edges |

`CausalGraph` enforces acyclicity only over directed edges. It does not enforce which combination of edge types is
"valid" for a given graph type — that is the caller's responsibility.

## Edge Semantics

`EdgeType` values and their string representations:

| `EdgeType` member | String | Description |
|---|---|---|
| `DIRECTED_EDGE` | `->` | Causal direction from source to destination |
| `UNDIRECTED_EDGE` | `--` | Undirected; semantics depend on graph type (see gotchas) |
| `BIDIRECTED_EDGE` | `<>` | Latent common cause (MAG/PAG) |
| `UNKNOWN_DIRECTED_EDGE` | `o>` | PAG: circle mark on tail side |
| `UNKNOWN_UNDIRECTED_EDGE` | `o-` | PAG: circle mark, undirected |
| `UNKNOWN_EDGE` | `oo` | PAG: both endpoints ambiguous |

Critical edge semantics differences by graph type:

| Edge | In CPDAG | In MAG | In PAG |
|---|---|---|---|
| `--` (undirected) | Orientation unknown — can resolve to `->` or `<-` | Implies selection bias | Not used |
| `<>` (bidirected) | Not used | Latent common confounder | Latent common confounder |
| `o>`, `o-`, `oo` | Not used | Not used | Ambiguous PAG-specific marks |

## Node Variable Types

`NodeVariableType` values:

| Member | String value | Meaning |
|---|---|---|
| `UNSPECIFIED` | `'unspecified'` | Default — type not declared |
| `CONTINUOUS` | `'continuous'` | Real-valued variable |
| `BINARY` | `'binary'` | Two-valued variable |
| `MULTICLASS` | `'multiclass'` | Categorical with 3+ classes |
| `ORDINAL` | `'ordinal'` | Ordered categorical |

`variable_type` defaults to `UNSPECIFIED` and is mutable post-creation:

```python
from cai_causal_graph import CausalGraph, NodeVariableType

cg = CausalGraph()
cg.add_node('x', variable_type=NodeVariableType.CONTINUOUS)
node = cg.get_node('x')
node.variable_type = NodeVariableType.BINARY  # mutable
```

`UNSPECIFIED` is dangerous — downstream tools in `cai-metrics` or `cai-modeling` may silently treat it as
continuous.

## Core Usage

### Building a graph

```python
from cai_causal_graph import CausalGraph, EdgeType, NodeVariableType

cg = CausalGraph()

# Nodes are auto-created when adding edges
cg.add_edge('x', 'y')                                          # default: DIRECTED_EDGE
cg.add_edge('z', 'y', edge_type=EdgeType.DIRECTED_EDGE)
cg.add_edge('a', 'b', edge_type=EdgeType.UNDIRECTED_EDGE)      # CPDAG undirected

# Or add nodes explicitly first
cg.add_node('w', variable_type=NodeVariableType.BINARY)
cg.add_edges_from([('w', 'x'), ('w', 'z')])

print(cg.is_dag())           # True if all edges are directed and acyclic
print(cg.get_node_names())
print(cg.get_edge_pairs())
```

### Inspecting nodes and edges

```python
node = cg.get_node('x')
print(node.identifier, node.variable_type)
print(node.is_source_node(), node.is_sink_node())
print(node.get_inbound_edges(), node.get_outbound_edges())

edge = cg.get_edge('x', 'y')
print(edge.source.identifier, edge.destination.identifier)
print(edge.get_edge_type())   # EdgeType.DIRECTED_EDGE
print(edge.descriptor)        # '(x -> y)'
```

### Serialization

```python
# dict round-trip (deepcopies metadata)
d = cg.to_dict()
cg2 = CausalGraph.from_dict(d)

# NetworkX (directed edges only -> DiGraph; undirected only -> Graph; mixed raises GraphConversionError)
nx_graph = cg.to_networkx()
cg3 = CausalGraph.from_networkx(nx_graph)

# Numpy adjacency matrix (directed/undirected only; bidirected/unknown raises TypeError)
adj, node_names = cg.to_numpy()
cg4 = CausalGraph.from_adjacency_matrix(adj, node_names=node_names)
```

### Skeleton

```python
skeleton = cg.skeleton          # property; returns Skeleton instance
print(skeleton.nodes)           # nodes without edge direction info
print(skeleton.edges)           # all edges as UNDIRECTED_EDGE
adj = skeleton.adjacency_matrix # symmetric binary matrix
```

### Time series graph

```python
from cai_causal_graph import TimeSeriesCausalGraph, EdgeType

ts = TimeSeriesCausalGraph()
ts.add_edge('X1 lag(n=1)', 'X1', edge_type=EdgeType.DIRECTED_EDGE)
ts.add_edge('X2 lag(n=1)', 'X2', edge_type=EdgeType.DIRECTED_EDGE)

node = ts.get_node('X1 lag(n=1)')
print(node.variable_name)   # 'X1'
print(node.time_lag)        # -1

summary = ts.get_summary_graph()    # CausalGraph collapsing lags
minimal = ts.get_minimal_graph()    # TimeSeriesCausalGraph with minimum lag structure
lag_0_nodes = ts.get_nodes_at_lag(0)
```

### Domain knowledge constraints

```python
from cai_causal_graph.type_definitions import EdgeConstraint

# EdgeConstraint is used by discovery algorithms, not enforced by CausalGraph itself
EdgeConstraint.HARD_DIRECTED_EDGE    # force edge direction
EdgeConstraint.HARD_UNDIRECTED_EDGE  # force undirected
EdgeConstraint.SOFT_DIRECTED_EDGE    # prefer direction
EdgeConstraint.FORBIDDEN_EDGE        # forbid the edge
```

## Assumptions and Limitations

- Self-loops are not allowed — adding `source == destination` raises `CyclicConnectionError`.
- Nodes are auto-created when adding edges if they do not already exist.
- `add_edge` raises `CyclicConnectionError` if directed edges would form a cycle; the graph is rolled back.
- `to_numpy()` / `from_adjacency_matrix()` only support `DIRECTED_EDGE` and `UNDIRECTED_EDGE` — bidirected or
  unknown edge types raise `TypeError`.
- `to_networkx()` only produces `DiGraph` (all directed) or `Graph` (all undirected) — mixed edge types raise
  `GraphConversionError`.
- Node and edge metadata is shallow-copied when passed to constructors; `from_dict` deepcopies.
- `replace_node()` copies edge `meta` shallowly from originals but does not preserve edge object identity.
- `networkx` version is bounded to `>=3.0.0, <3.3.0` due to a bug in `get_all_simple_paths` in 3.3.

## Gotchas

- `--` (undirected) in a CPDAG means orientation is unknown and can be resolved either way. The same symbol in a
  MAG implies selection bias — a completely different semantic.
- Deleting a node invalidates the `Node` object — any subsequent property access on that object raises
  `NodeDoesNotExistError`.
- `node.variable_type` is mutable; changing it after creation is intentional and supported.
- `Edge.__eq__` treats `UNDIRECTED_EDGE`, `BIDIRECTED_EDGE`, and `UNKNOWN_EDGE` as directionless — `(a -- b)` and
  `(b -- a)` are equal.
- `CausalGraph.is_dag()` returns `False` immediately if any non-directed edge is present (does not check only
  directed subgraph).
- `Skeleton.edges` returns all edges as `UNDIRECTED_EDGE` regardless of original type.

## Error Types

All errors are nested inside `CausalGraphErrors`:

| Exception | When raised |
|---|---|
| `CyclicConnectionError` | Self-loop or directed cycle |
| `NodeDoesNotExistError` | Accessing deleted or missing node |
| `NodeDuplicatedError` | Adding a node that already exists |
| `EdgeDoesNotExistError` | Accessing missing or deleted edge |
| `EdgeExistsError` | Adding an edge that already exists |
| `ReverseEdgeExistsError` | Adding edge when reverse already exists |
| `GraphConversionError` | `to_networkx()` / `to_gml_string()` with mixed edges |
| `InvalidAdjacencyMatrixError` | Non-square or non-binary matrix in `from_adjacency_matrix()` |

```python
from cai_causal_graph.exceptions import CausalGraphErrors

try:
    cg.get_node('missing')
except CausalGraphErrors.NodeDoesNotExistError:
    ...
```

## Cross-References

- For scoring / evaluating causal graphs against ground truth → see `cai-metrics` skill.
- For discovering causal graphs from data → see `cai-causal-discovery` skill.
- For causal modeling and inference using graphs → see `cai-modeling` skill.
