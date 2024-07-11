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
from unittest import TestCase

from cai_causal_graph.utils import get_variable_name_and_lag


class TestTimeSeriesGraphUtils(TestCase):
    def test_get_variable_name_and_lag(self):
        # 'Normal' name
        self.assertTupleEqual(get_variable_name_and_lag('x1'), ('x1', 0))
        self.assertTupleEqual(get_variable_name_and_lag('x1 lag(n=1)'), ('x1', -1))
        self.assertTupleEqual(get_variable_name_and_lag('x1 future(n=1)'), ('x1', 1))

        # With spaces
        self.assertTupleEqual(get_variable_name_and_lag('x1 standardized()'), ('x1 standardized()', 0))
        self.assertTupleEqual(get_variable_name_and_lag('x1 standardized() lag(n=1)'), ('x1 standardized()', -1))
        self.assertTupleEqual(get_variable_name_and_lag('x1 standardized() future(n=1)'), ('x1 standardized()', 1))

        # With special characters
        self.assertTupleEqual(
            get_variable_name_and_lag('1@£$%^&*()-+=*/_\\"\'<>?*~.'), ('1@£$%^&*()-+=*/_\\"\'<>?*~.', 0)
        )
        self.assertTupleEqual(
            get_variable_name_and_lag('1@£$%^&*()-+=*/_\\"\'<>?*~. lag(n=2)'), ('1@£$%^&*()-+=*/_\\"\'<>?*~.', -2)
        )
        self.assertTupleEqual(
            get_variable_name_and_lag('1@£$%^&*()-+=*/_\\"\'<>?*~. future(n=2)'), ('1@£$%^&*()-+=*/_\\"\'<>?*~.', 2)
        )

        # Has something that looks like lag in the name
        self.assertTupleEqual(get_variable_name_and_lag('x1 lag(n=not_a_lag)'), ('x1 lag(n=not_a_lag)', 0))
        self.assertTupleEqual(get_variable_name_and_lag('x1 lag(n=not_a_lag) lag(n=3)'), ('x1 lag(n=not_a_lag)', -3))
        self.assertTupleEqual(get_variable_name_and_lag('x1 lag(n=not_a_lag) future(n=3)'), ('x1 lag(n=not_a_lag)', 3))

        # Has lag and future in the name
        self.assertTupleEqual(get_variable_name_and_lag('lag future'), ('lag future', 0))
        self.assertTupleEqual(get_variable_name_and_lag('lag future lag(n=1)'), ('lag future', -1))
        self.assertTupleEqual(get_variable_name_and_lag('lag future future(n=1)'), ('lag future', 1))

        # Has only whitespace
        self.assertTupleEqual(get_variable_name_and_lag('  '), ('  ', 0))
        self.assertTupleEqual(get_variable_name_and_lag('   lag(n=1)'), ('  ', -1))
        self.assertTupleEqual(get_variable_name_and_lag('   future(n=1)'), ('  ', 1))

        # Incomplete match of lag does not count as a lag
        self.assertTupleEqual(get_variable_name_and_lag('x1 lag(n=1'), ('x1 lag(n=1', 0))
        self.assertTupleEqual(get_variable_name_and_lag('x1 lag(n=1 lag(n=1)'), ('x1 lag(n=1', -1))
        self.assertTupleEqual(get_variable_name_and_lag('x1 lag(n=1 future(n=1)'), ('x1 lag(n=1', 1))

        # New lines match
        self.assertTupleEqual(get_variable_name_and_lag('x\n1'), ('x\n1', 0))
        self.assertTupleEqual(get_variable_name_and_lag('x\n1 lag(n=1)'), ('x\n1', -1))
        self.assertTupleEqual(get_variable_name_and_lag('x\n1 future(n=1)'), ('x\n1', 1))

        # New line at the end of a variable name matches (edge case regression test)
        self.assertTupleEqual(get_variable_name_and_lag('x1\n'), ('x1\n', 0))
        self.assertTupleEqual(get_variable_name_and_lag('x1\n lag(n=1)'), ('x1\n', -1))
        self.assertTupleEqual(get_variable_name_and_lag('x1\n future(n=1)'), ('x1\n', 1))

        # Multiple new line at the end of a variable name matches (edge case regression test)
        self.assertTupleEqual(get_variable_name_and_lag('x1\n\n\n'), ('x1\n\n\n', 0))
        self.assertTupleEqual(get_variable_name_and_lag('x1\n\n\n lag(n=1)'), ('x1\n\n\n', -1))
        self.assertTupleEqual(get_variable_name_and_lag('x1\n\n\n future(n=1)'), ('x1\n\n\n', 1))

    def test_get_variable_name_and_lag_raises(self):
        self.assertRaises(ValueError, get_variable_name_and_lag, 'x1 lag(n=1) lag(n=2)')
        self.assertRaises(ValueError, get_variable_name_and_lag, 'x1 future(n=1) future(n=2)')

        self.assertRaises(ValueError, get_variable_name_and_lag, 'x1 lag(n=1) future(n=2)')
        self.assertRaises(ValueError, get_variable_name_and_lag, 'x1 future(n=1) lag(n=2)')

        self.assertRaises(ValueError, get_variable_name_and_lag, 'x1 lag(n=1) something in the middle future(n=2)')
        self.assertRaises(ValueError, get_variable_name_and_lag, 'x1 future(n=1) something in the middle lag(n=2)')
