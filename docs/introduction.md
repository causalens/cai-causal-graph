# Introduction

[Causal graphs](https://youtu.be/gxA2YxkrKDg) represent the flow of information in the underlying data generating process 
of a given data set. They comprise a collection of causal relationships between variables in the data set, which dictate 
how a given variable causally affects other variables. It is important to note that the causal graph does not define how 
a specific node is functionally related to its parents; that information is encoded in a structural causal model. The 
causal graph simply shows how information flows from one node (i.e., a variable in a data set) to another. As a result, 
causal graphs are a fundamental element of [Causal AI](https://causalens.com/resources/knowledge-hub/what-is-causalai/). 
The majority of Causal AI tasks, such as [causal modeling](https://causalens.com/causalnet-state-of-the-art-structural-causal-modeling/), 
[counterfactual reasoning](https://youtu.be/NwF_gjvcKfE), [causal effect estimation](https://youtu.be/1ZR44wH9QCU), 
[root cause analysis](https://causalens.com/root-cause-analysis/), 
[algorithmic recourse](https://causalens.com/algorithmic-recourse/), and 
[causal fairness](https://causalens.com/causal-fairness/), rely on an accurate and comprehensive causal graph that 
correctly reflects the underlying data generating process.

Causal graphs can be discovered from observational data, which is the goal of 
[causal discovery](https://causalens.com/resources/knowledge-hub/discovering-causal-relationships/). An essential part of 
this discovery process is providing prior knowledge about specific causal relationships, usually formed by experts, which 
substantially reduces computational time and improves accuracy of causal discovery. There are many types of prior 
knowledge that can be provided, such as forbidden edges, directed edges, tiers of causal relationships, among others.
This type of [human-guided causal discovery](https://causalens.com/human-guided-causal-discovery/) is a key 
component in [decisionOS](https://causalens.com/decision-os/).

The `cai-causal-graph` package provides a user-friendly implementation of a causal graph class 
(`cai_causal_graph.causal_graph.CausalGraph`) that allows you to easily define mixed graphs that can represent various 
types of causal graphs. See the [Types of Causal Graphs](#types-of-causal-graphs) section below for information
on different types of causal graphs.

You can find a [quickstart](quickstart.md) to see how to easily build a basic graph, with further details provided in 
the [Causal Graph](causal_graph.md) documentation page. For a full list of all the classes and methods, please see
the provided reference docs. For example, these are the reference docs for the `cai_causal_graph.causal_graph.CausalGraph` 
class.

# Types of Causal Graphs

The most common type of mixed graph used to represent a causal graph is a _Directed Acyclic Graph_ (_DAG_), which has
ony directed edges and the no cycles. However, discovering a single _DAG_ for a given data set is difficult. Certain 
causal relationships are indistinguishable from each other with only observational data, because they encode the
same conditional independencies. The set of such causal relationships is called the _Markov equivalence class_ (_MEC_)
for a particular set of nodes.

For instance, if you identify the graphical structure `X -> Y -> Z`, then corresponding data would show that `X` 
**is** independent of `Z` given `Y`. However, the graphical structures `X <- Y <- Z` and `X <- Y -> Z` would lead to 
the exact same conditional independence test result as above. Only if the graphical structure found was a collider 
connection `X -> Y <- Z` would you be able to identify the structure from observational data,
because the data would tell you that `X` and `Z` are independent, but become dependent given `Y`. The _MEC_ of 
`(X, Y, Z)` is typically represented with undirected edges, i.e., `X -- Y -- Z`.

One prominent representation of _MECs_ is a _Completed Partially Directed Acyclic Graph_ (_CPDAG_), which only contains
directed (`->`) and undirected (`--`) edges. In this case, an undirected edge simply implies that a causal relationship
can point either way, i.e., `A -- B` can be resolved to either `A -> B` or `A <- B`. In the above example, the _CPDAG_
_MEC_ of the two _DAGs_ is therefore `A -- B -- C`.

A _Maximal Ancestral Graph_ (_MAG_) can encode all the information that _CPDAGs_ can, but also provides
more detailed information about the relationships between variables, such as including whether a latent confounder is
likely to exist or selection bias is likely to be present. Specifically, _MAGs_ may also contain bi-directed edges
`A <> B`, which implies that there is a latent confounder between the respective variables. A _Partial Ancestral Graph_ (_PAG_) 
describes an equivalence class of _MAGs_. _PAGs_ may also contain "wild-card" or "circle" edges `A -o B`, which can 
either be a directed or undirected arrow head, i.e. `A -o B` can be  resolved to `A -- B` or `A -> B`. The `o` end is r
eferenced to as "unknown" in this package. See `cai_causal_graph.type_definitions.EDGE_T` for all the supported edge 
types. Similar to a _CPDAG_, a _PAG_ can represent  a number of _DAGs_. Therefore, _PAGs_, _CPDAGs_ and _DAGs_ can be 
thought of in a hierarchical way.

![DAG CPDAG PAG](assets/DAG_CPDAG_PAG.png)

Note that the `cai_causal_graph.causal_graph.CausalGraph` class can contain all the aforementioned edge types, and 
can therefore represent _DAGs_, _CPDAGs_, _MAGs_, and _PAGs_.
