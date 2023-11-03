"""
Copyright (c) 2023 by Impulse Innovations Ltd. Private and confidential. Part of the causaLens product.
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
            get_variable_name_and_lag('1@£$%^&*()-+=*/_\\"\'<>?*~.'), ('1@£$%^&*()-+=*/_\\"\'<>?*~', 0)
        )
        self.assertTupleEqual(
            get_variable_name_and_lag('1@£$%^&*()-+=*/_\\"\'<>?*~. lag(n=2)'), ('1@£$%^&*()-+=*/_\\"\'<>?*~', -2)
        )
        self.assertTupleEqual(
            get_variable_name_and_lag('1@£$%^&*()-+=*/_\\"\'<>?*~. future(n=2)'), ('1@£$%^&*()-+=*/_\\"\'<>?*~', 2)
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

    def test_get_variable_name_and_lag_raises(self):
        self.assertRaises(ValueError, get_variable_name_and_lag, 'x1 lag(n=1) lag(n=2)')
        self.assertRaises(ValueError, get_variable_name_and_lag, 'x1 future(n=1) future(n=2)')

        self.assertRaises(ValueError, get_variable_name_and_lag, 'x1 lag(n=1) future(n=2)')
        self.assertRaises(ValueError, get_variable_name_and_lag, 'x1 future(n=1) lag(n=2)')

        self.assertRaises(ValueError, get_variable_name_and_lag, 'x1 lag(n=1) something in the middle future(n=2)')
        self.assertRaises(ValueError, get_variable_name_and_lag, 'x1 future(n=1) something in the middle lag(n=2)')
