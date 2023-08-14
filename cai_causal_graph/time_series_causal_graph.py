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

from __future__ import annotations

import re
from typing import Dict, List, Optional, Union

import numpy

from cai_causal_graph import CausalGraph
from cai_causal_graph.type_definitions import NodeLike
from cai_causal_graph.type_definitions import EDGE_T
from cai_causal_graph.graph_components import get_variable_name_and_lag

def extract_names_and_lags(node_names: List[NodeLike]) -> List[Dict[str, int]]:
    """
    Extract the names and lags from a list of node names.
    This is useful for converting a list of node names into a list of node names and lags.

    Example:
        ['X', 'Y', 'Z lag(n=2)'] -> [{'X': 0}, {'Y': 0}, {'Z': 2}]

    :param node_names: List of node names.
    :return: List of dictionaries with node names and lags sorted by lag and the maximum lag.
    """
    names_and_lags = []
    for node_name in node_names:
        variable_name, lag = get_variable_name_and_lag(node_name)
        names_and_lags.append({variable_name: lag})
    sorted_names_and_lags = sorted(names_and_lags, key=lambda x: max(x.values()), reverse=True)
    first_dict_value = next(iter(sorted_names_and_lags[0].values()))
    return sorted_names_and_lags, first_dict_value

class TimeSeriesCausalGraph(CausalGraph):
    """
    A causal graph for time series data.
    
    #TODO: add more documentation

    The node in a time series causal graph will have additional metadata that 
    gives the time information of the node. The two additional metadata are:
    - 'timedelta': the time difference with respect the reference time 0
    - 'variable_name': the name of the variable (without the lag information)
    """

    def __init__(
        self,
        input_list: Optional[List[NodeLike]] = None,
        output_list: Optional[List[NodeLike]] = None,
        fully_connected: bool = True,
        store_graphs: bool = False,
    ):
        """Initialize the time series causal graph."""
        super().__init__(input_list, output_list, fully_connected)
        
        self._store_graphs = store_graphs
        # autoregressive order of the graph (max lag)
        self._order: Optional[int] = None
        # list of variables in the graph, i.e. discarding the lags (X1(t-1) and X1 are the same variable)
        self._variables: Optional[List[NodeLike]] = None
        self._summary_graph: Optional[CausalGraph] = None
        self._minimal_graph: Optional[TimeSeriesCausalGraph] = None
        self._meta_time_name = 'timedelta'
        self._meta_variable_name = 'variable_name'

    def __eq__(self, other: object) -> bool:
        """Return True if the graphs are equal."""
        if not isinstance(other, TimeSeriesCausalGraph):
            return False
        
        # now check if the graphs are equal. Since TimeSeriesCausalGraph is a subclass of CausalGraph,
        # we can use the CausalGraph.__eq__ method and then check for the metadata
        are_equal = super().__eq__(other)
        # no need to check the metadata if the graphs are not equal
        if not are_equal:
            return False

        # now check the metadata timedelta in the nodes
        for node in self.get_nodes():
            other_node = other.get_node(node.identifier)
            # check the variable name
            node_metadata_variable = node.get_metadata()[self._meta_variable_name]
            other_node_metadata_td = other_node.get_metadata()[self._meta_variable_name]

            if node_metadata_variable != other_node_metadata_td:
                return False
            
            # check the time delta
            node_metadata_td = node.get_metadata()[self._meta_time_name]
            other_node_metadata_td = other_node.get_metadata()[self._meta_time_name]

            if node_metadata_td != other_node_metadata_td:
                return False

        return True

    def get_minimum_graph(self) -> TimeSeriesCausalGraph:
        """
        Return a minimal graph.

        The minimal graph is the graph with the minimum number of edges that is equivalent to the original graph.
        In other words, it is a graph that has no edges whose destination is not time delta 0.
        """
        if self.is_minimal_graph():
            return self
        # we need to remove the edges that are not in the minimal graph

        return self

    def is_minimal_graph(self) -> bool:
        """Return True if the graph is minimal."""
        return self == self.get_minimum_graph()

    def get_summary_graph(self) -> CausalGraph:
        """
        Return a summary graph.

        Collapse graph in time into a single node per variable (column name).
        This can become cyclic and bi-directed as X-1 -> Y and Y-1 -> X would become X <-> Y.

        There are several cases to consider. Assume the edge in consideration is called B.
        - if the edge is not in the summary graph, add it
        - if there's an edge A is in the summary graph, then:
            - if A and B have the same direction, keep the direction
            - if A and B have different directions, make it bi-directed
            - if one of the two is already bi-directed, keep it bi-directed

        :return: The summary graph as a CausalGraph object.
        """
        if self._summary_graph is None:
            summary_graph = CausalGraph()
            # now check as described above (assume egdes are already directed)
            edges = self.get_edges()
            for edge in edges:
                # first we need to extract the variable names from the nodes as the summary graph
                # will have the variable names as nodes
                source_node = edge.source
                destination_node = edge.destination
                source_variable_name = source_node.get_metadata()[self._meta_variable_name]
                destination_variable_name = destination_node.get_metadata()[self._meta_variable_name]

                if source_variable_name != destination_variable_name:
                    ltr = summary_graph.is_edge_by_pair((source_variable_name, destination_variable_name))
                    rtl = summary_graph.is_edge_by_pair((destination_variable_name, source_variable_name))

                    # if edge is not in the summary graph, add it
                    if not ltr and not rtl:
                        summary_graph.add_edge_by_pair((source_variable_name, destination_variable_name))
                    elif rtl:
                        # if edge is already in the summary graph but in the opposite direction, make it bi-directed
                        # remove the edge and add it again as bi-directed
                        summary_graph.remove_edge_by_pair((source_variable_name, destination_variable_name))
                        summary_graph.add_edge((source_variable_name, destination_variable_name), directed=False) 
            self._summary_graph = summary_graph 
        return self._summary_graph

    def extend_graph(
        self, backward_steps: Optional[int] = None, forward_steps: Optional[int] = None
    ) -> TimeSeriesCausalGraph:
        """
        Return an extended graph.
        Extend the graph in time by adding nodes for each variable at each time step from backward_steps to forward_steps.

        If both backward_steps and forward_steps are None, return the original graph.

        :param backward_steps: Number of steps to extend the graph backwards in time. If None, do not extend backwards.
        :param forward_steps: Number of steps to extend the graph forwards in time. If None, do not extend forwards.
        :return: Extended graph with nodes for each variable at each time step from backward_steps to forward_steps.
        """
        # check steps are valid (positive integers) if not None
        assert backward_steps is None or backward_steps > 0
        assert forward_steps is None or forward_steps > 0

        # since both of these are relative to time zero we need to validate that backwards is >= maxlag (order) in minimal graph
        return self

    @staticmethod
    def get_variable_names_from_node_names(node_names: List[str]):
        """
        Return a list of variable names from a list of node names.
        This is useful for converting a list of node names into a list of variable names.
        Variables are the elementary (unique) units of a time series causal graph.

        Example:
            ['X', 'X lag(n=1)', 'Y', 'Z lag(n=2)'] -> ['X', 'Y', 'Z']

        :param node_names: Can be a list of node names or a list of dictionaries with node names and lags.
        :return: List of variable names.
        """
        assert isinstance(node_names, list)
        
        for node_name in node_names:
            vname, _ = get_variable_name_and_lag(node_name)
            if vname not in node_names:
                node_names.append(vname)

        return sorted(node_names)
        
        

    @classmethod
    def from_causal_graph(cls, causal_graph: CausalGraph, store_graphs: bool=False) -> TimeSeriesCausalGraph:
        """
        Return a time series causal graph from a causal graph.
        This is useful for converting a causal graph from a single time step into a time series causal graph.
        """
        # create an instance without calling __init__
        obj = cls.__new__(cls)

        # copy all attributes from causal_graph to the new object
        for name, value in vars(causal_graph).items():
            setattr(obj, name, value)

        # set all the attributes related to time series

        # create a temporary TimeSeriesCausalGraph object to get default values for new attributes
        temp_obj = cls(None, None, None, None)  # Assuming __init__ takes three arguments
        
        # Copy all attributes from the temporary object to the new object
        for name, value in vars(temp_obj).items():
            if not hasattr(obj, name):
                setattr(obj, name, value)

        # set the store_graphs attribute
        obj._store_graphs = store_graphs

        variables = []
        maxlag = 0
        # now for each node in tsgraph set the metadata
        for node in obj.get_nodes():
            # get the variable name and lag from the node name
            variable_name, lag = get_variable_name_and_lag(node.identifier)
            maxlag = max(maxlag, lag)
            # set the variable name and lag in the metadata
            node.variable_name = variable_name
            node.lag = lag
            variables.append(variable_name) if variable_name not in variables else variables
        
        obj._order = maxlag
        obj._variables = sorted(variables)
        return obj

    def from_adjacency_matrices(
        self,
        adjacency_matrices: List[Dict[int, numpy.ndarray]],
        node_names: Optional[List[Union[NodeLike, int]]] = None,
    ) -> TimeSeriesCausalGraph:
        """
        Return a time series causal graph from a dictionary of adjacency matrices. Keys are the time delta.
        This is useful for converting a list of adjacency matrices into a time series causal graph.
        """
        return self

    def get_full_autoregressive_adjacency_matrix(self) -> numpy.ndarray:
        """
        Return a full autoregressive adjacency matrix of the graph with inter-slice edges, i.e. X-1 -> X.

        This is useful for converting a time series causal graph into a single adjacency matrix.
        Return sparse matrix of shape (order d \times d) where d is the number of nodes.
        """
        pass

    def get_full_instance_adjacency_matrix(self) -> numpy.ndarray:
        """
        Return a full instance adjacency matrix of the graph with intra-slice edges, i.e. X -> Y.

        This is useful for converting a time series causal graph into a single adjacency matrix.
        Return sparse matrix of shape (d \times d) where d is the number of nodes.
        """
        pass

    @property
    def order(self) -> int:
        """Return the order of the graph."""
        return self._order

    @property
    def variables(self) -> List[str]:
        """Return the variables in the graph."""
        return self._variables    
