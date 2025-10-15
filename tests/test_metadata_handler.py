"""
Copyright (c) 2024 by Impulse Innovations Ltd. Private and confidential. Part of the causaLens product.
"""

from typing import List, Optional
from unittest import TestCase

from cai_causal_graph.exceptions import MetaDataError
from cai_causal_graph.interfaces import HasMetadata
from cai_causal_graph.metadata_handler import MetaField


class ManualHasMeta(HasMetadata):
    def __init__(self, meta: Optional[dict] = None, **kwargs):
        super().__init__(meta=self._process_meta(meta=meta, kwargs_dict=kwargs, raise_if_unknown_tags=True))

    @classmethod
    def get_metadata_schema(cls) -> List[MetaField]:
        return super().get_metadata_schema() + [
            MetaField(metatag='model', property_name='model'),
            MetaField(metatag='foo_tag', property_name='_foo', parameter_name='foo', default_value='bar'),
        ]


class ChildManualHasMeta(ManualHasMeta):
    @classmethod
    def get_metadata_schema(cls) -> List[MetaField]:
        return super().get_metadata_schema() + [
            MetaField(metatag='hello', property_name='_hello', parameter_name='hello'),
        ]


class GrandChildWithDuplicateTag(ChildManualHasMeta):
    @classmethod
    def get_metadata_schema(cls) -> List[MetaField]:
        return super().get_metadata_schema() + [
            MetaField(metatag='hello', property_name='__hello', parameter_name='_hello'),
        ]


class GrandChildWithDuplicateProperty(ChildManualHasMeta):
    @classmethod
    def get_metadata_schema(cls) -> List[MetaField]:
        return super().get_metadata_schema() + [
            MetaField(metatag='_hello', property_name='_hello', parameter_name='_hello'),
        ]


class GrandChildWithDuplicateParameter(ChildManualHasMeta):
    @classmethod
    def get_metadata_schema(cls) -> List[MetaField]:
        return super().get_metadata_schema() + [
            MetaField(metatag='_hello', property_name='hello', parameter_name='hello'),
        ]


class TestMetaField(TestCase):
    def test_constructor_without_parameter_name(self):
        field = MetaField(metatag='a', property_name='b')

        self.assertEqual(field.parameter_name, field.property_name)
        self.assertEqual(field.parameter_name, 'b')


class TestHasMetaManual(TestCase):
    def test_handle_only_kwargs(self):
        o = ManualHasMeta(model=10, foo='barfoo')

        self.assertEqual(o.meta['model'], 10)
        self.assertEqual(o.meta['foo_tag'], 'barfoo')
        self.assertEqual(len(o.meta), 2)

    def test_handle_meta_and_kwargs(self):
        meta = {'hello': 'bye'}

        o = ManualHasMeta(
            meta=meta,
            model=10,
            foo='barfoo',
        )

        # test for shallow copy of meta
        meta['bye'] = 'hello'

        self.assertEqual(o.meta['model'], 10)
        self.assertEqual(o.meta['foo_tag'], 'barfoo')
        self.assertEqual(o.meta['hello'], 'bye')
        self.assertEqual(len(o.meta), 3)

    def test_handle_only_meta(self):
        meta = dict(
            hello='bye',
            model=10,
        )

        o = ManualHasMeta(
            meta=meta,
            foo='barfoo',
        )

        # test for shallow copy of meta
        meta['bye'] = 'hello'

        self.assertEqual(o.meta['model'], 10)
        self.assertEqual(o.meta['foo_tag'], 'barfoo')
        self.assertEqual(o.meta['hello'], 'bye')
        self.assertEqual(len(o.meta), 3)

    def test_handle_meta_overlapping_with_kwargs(self):
        meta = dict(
            hello='bye',
            model=10,
        )

        o = ManualHasMeta(
            meta=meta,
            model=11,
            foo='barfoo',
        )

        self.assertEqual(o.meta['model'], 11)

        # passing model which equals to the one in meta, should pass
        o = ManualHasMeta(
            meta=meta,
            model=10,
            foo='barfoo',
        )

        # test for shallow copy of meta
        meta['bye'] = 'hello'

        self.assertEqual(o.meta['model'], 10)
        self.assertEqual(o.meta['foo_tag'], 'barfoo')
        self.assertEqual(o.meta['hello'], 'bye')
        self.assertEqual(len(o.meta), 3)

    def test_handle_empty(self):
        o = ManualHasMeta()
        # should add the default value in
        self.assertEqual(len(o.meta), 1)
        self.assertDictEqual(o.meta, {'foo_tag': 'bar'})

    def test_unknown_kwarg_raises(self):
        with self.assertRaises(MetaDataError):
            ManualHasMeta(unknown=2)

    def test_default(self):
        meta = {'hello': 'bye'}

        o = ManualHasMeta(
            meta=meta,
            model=10,
        )

        # test for shallow copy of meta
        meta['bye'] = 'hello'

        self.assertEqual(o.meta['model'], 10)
        # test default value was set
        self.assertEqual(o.meta['foo_tag'], 'bar')
        self.assertEqual(o.meta['hello'], 'bye')
        self.assertEqual(len(o.meta), 3)

    def test_duplicate_tags(self):
        with self.assertRaises(MetaDataError):
            GrandChildWithDuplicateTag()

    def test_duplicate_property_names(self):
        with self.assertRaises(MetaDataError):
            GrandChildWithDuplicateProperty()

    def test_duplicate_parameters(self):
        with self.assertRaises(MetaDataError):
            GrandChildWithDuplicateParameter()

    def test_on_child_kwargs_only_parent(self):
        o = ChildManualHasMeta(model=10, foo='barfoo')

        self.assertEqual(o.meta['model'], 10)
        self.assertEqual(o.meta['foo_tag'], 'barfoo')
        self.assertEqual(len(o.meta), 2)

    def test_on_child_kwargs_only_child(self):
        o = ChildManualHasMeta(hello=3)

        self.assertEqual(len(o.meta), 2)
        self.assertEqual(o.meta['hello'], 3)

    def test_on_child_mixed_kwargs(self):
        o = ChildManualHasMeta(model=10, foo='barfoo', hello=3)

        self.assertEqual(o.meta['model'], 10)
        self.assertEqual(o.meta['foo_tag'], 'barfoo')
        self.assertEqual(o.meta['hello'], 3)
        self.assertEqual(len(o.meta), 3)

    def test_on_child_kwargs_and_meta(self):
        meta = dict(hello='bye', model=10, cool=True)

        o = ChildManualHasMeta(
            meta=meta,
            foo='barfoo',
        )

        # test for shallow copy of meta
        meta['bye'] = 'hello'

        self.assertEqual(o.meta['model'], 10)
        self.assertEqual(o.meta['foo_tag'], 'barfoo')
        self.assertEqual(o.meta['hello'], 'bye')
        self.assertEqual(o.meta['cool'], True)
        self.assertEqual(len(o.meta), 4)
