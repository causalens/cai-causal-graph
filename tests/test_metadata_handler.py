"""
Copyright (c) 2024 by Impulse Innovations Ltd. Private and confidential. Part of the causaLens product.
"""
from unittest import TestCase

from cai_causal_graph.metadata_handler import MetaField


class TestMetaField(TestCase):
    def test_constructor_without_parameter_name(self):
        field = MetaField(metatag='a', property_name='b')

        self.assertEqual(field.parameter_name, field.property_name)
        self.assertEqual(field.parameter_name, 'b')

        print()
