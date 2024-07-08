"""
Copyright 2023 Impulse Innovations Limited

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

__all__ = [
    '__version__',
    # Causal Graph and components
    'CausalGraph',
    'Skeleton',
    'TimeSeriesCausalGraph',
    # Causal types
    'EdgeConstraint',
    'EdgeType',
    'NodeVariableType',
    'EDGE_T',  # Keep for backwards compatibility.
    'NODE_T',  # Keep for backwards compatibility.
    # Time series specific tags
    'TIME_LAG',
    'VARIABLE_NAME',
]

__version__ = '0.5.2'

from cai_causal_graph.causal_graph import CausalGraph, Skeleton
from cai_causal_graph.time_series_causal_graph import TimeSeriesCausalGraph
from cai_causal_graph.type_definitions import (
    EDGE_T,
    NODE_T,
    TIME_LAG,
    VARIABLE_NAME,
    EdgeConstraint,
    EdgeType,
    NodeVariableType,
)
