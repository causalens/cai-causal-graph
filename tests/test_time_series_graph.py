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

import unittest

from cai_causal_graph import CausalGraph
from cai_causal_graph.time_series_causal_graph import extract_names_and_lags
from cai_causal_graph import TimeSeriesCausalGraph
from cai_causal_graph import EDGE_T, NODE_T


class TestCausalGraphEdgeTypes(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up a time series causal graph tests.

        Create a causal graph `graph_1` with 4 nodes with lagged variables
        autoregressive edges:
            - X1(t-1) -> X1(t)
            - X2(t-1) -> X2(t)
            - X3(t-1) -> X3(t)
            - X1(t-1) -> X2(t)
        instantaneous edges:
            - X1(t) -> X3(t)
        """

        self.nodes = ['X1', 'X1 lag(n=1)', 'X2 lag(n=1)', 'X3 lag(n=1)', 'X2', 'X3']

        # define a DAG
        self.dag = CausalGraph()
        self.dag.add_nodes_from(self.nodes)
        # auto-regressive edges
        self.dag.add_edge('X1 lag(n=1)','X1', edge_type=EDGE_T.DIRECTED_EDGE)
        self.dag.add_edge('X2 lag(n=1)','X2', edge_type=EDGE_T.DIRECTED_EDGE)
        self.dag.add_edge('X3 lag(n=1)','X3', edge_type=EDGE_T.DIRECTED_EDGE)
        self.dag.add_edge('X1 lag(n=1)','X2', edge_type=EDGE_T.DIRECTED_EDGE)
        # instantaneous edges
        self.dag.add_edge('X1','X3', edge_type=EDGE_T.DIRECTED_EDGE)
        self.tsdag = TimeSeriesCausalGraph.from_causal_graph(self.dag) 
        
        # create a more complex DAG as follows
        self.nodes_1 = ['X1', 'X1 lag(n=1)', 'X1 lag(n=2)', 'X2', 'X2 lag(n=1)', 'X2 lag(n=2)', 'X3', 'X3 lag(n=1)', 'X3 lag(n=2)']
        self.dag_1 = CausalGraph()
        self.dag_1.add_nodes_from(self.nodes_1)
        # auto-regressive edges
        # X1
        self.dag_1.add_edge('X1 lag(n=1)','X1', edge_type=EDGE_T.DIRECTED_EDGE)
        self.dag_1.add_edge('X1 lag(n=2)','X1 lag(n=1)', edge_type=EDGE_T.DIRECTED_EDGE)
        # X2
        self.dag_1.add_edge('X2 lag(n=1)','X2', edge_type=EDGE_T.DIRECTED_EDGE)
        self.dag_1.add_edge('X2 lag(n=2)','X2 lag(n=1)', edge_type=EDGE_T.DIRECTED_EDGE)
        # X3
        self.dag_1.add_edge('X3 lag(n=1)','X3', edge_type=EDGE_T.DIRECTED_EDGE)
        self.dag_1.add_edge('X3 lag(n=2)','X3 lag(n=1)', edge_type=EDGE_T.DIRECTED_EDGE)

        # X3 (t-1) -> X1 (t)
        self.dag_1.add_edge('X3 lag(n=1)','X1', edge_type=EDGE_T.DIRECTED_EDGE)
        # X3 (t-2) -> X2 (t)
        self.dag_1.add_edge('X3 lag(n=2)','X2', edge_type=EDGE_T.DIRECTED_EDGE)

        # instantaneous edges (X3 -> X1)
        self.dag_1.add_edge('X3','X1', edge_type=EDGE_T.DIRECTED_EDGE)
        self.tsdag_1 = TimeSeriesCausalGraph.from_causal_graph(self.dag_1)
        

    def test_extract_names_and_lags(self):
        nodes, maxlag = extract_names_and_lags(self.nodes)
        self.assertEqual(nodes, [{'X1': 1}, {'X2': 1}, {'X3': 1}, {'X1': 0}, {'X2': 0}, {'X3': 0}])
        assert maxlag == 1

    def test_from_causal_graph(self):
        # dag
        self.assertEqual(self.tsdag.identifier, '<X1 lag(n=1)>_<X2 lag(n=1)>_<X3 lag(n=1)>_<X1>_<X2>_<X3>')
        self.assertNotEqual(self.tsdag, self.dag)

        # dag_1
        self.assertEqual(self.tsdag_1.identifier, '<X1 lag(n=1)>_<X2 lag(n=1)>_<X3 lag(n=1)>_<X1 lag(n=2)>_<X2 lag(n=2)>_<X3 lag(n=2)>_<X1>_<X2>_<X3>')
        self.assertNotEqual(self.tsdag_1, self.dag_1)


    def test_summary_graph(self):
        summary_graph = self.tsdag.get_summary_graph()
        # the graph should be  X3 <- X1 -> X2
        self.assertEqual(summary_graph.identifier, '<X1>_<X3>_<X2>')

    def test_variable_names(self):
        variables = self.tsdag.variables
        self.assertEqual(variables, ['X1', 'X2', 'X3'])
