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

import abc
import logging
from copy import copy
from typing import List, Optional

from cai_causal_graph.exceptions import MetaDataError
from cai_causal_graph.metadata_handler import MetaField

logger = logging.getLogger(__name__)


class CanDictSerialize(abc.ABC):
    """An interface for an object that can be represented as a dictionary."""

    @abc.abstractmethod
    def to_dict(self) -> dict:
        """Return a dictionary representation of the object."""
        pass


class CanDictDeserialize(abc.ABC):
    """An interface for an object that can be instantiated from a dictionary."""

    @classmethod
    @abc.abstractmethod
    def from_dict(cls, d: dict) -> CanDictDeserialize:
        """Return an instance of itself, constructed from the provided dictionary."""
        pass


class HasIdentifier(abc.ABC):
    """An interface for an object that has an identifier."""

    @property
    @abc.abstractmethod
    def identifier(self) -> str:
        """A unique identifier property."""
        pass

    def get_identifier(self) -> str:
        """Return the identifier."""
        return self.identifier


class HasMetadata:
    """
    An interface for an object that has metadata.

    Metadata is stored in a dictionary which can be passed at construction. A common design pattern is then for
    a class to have multiple properties which provide user-friendly access to specific metadata fields. These fields
    can be commonly set at the construction of an object.

    Moreover, the original class may be extended by adding more properties. This interface has been designed to
    facilitate this by ensuring that extending these classes is simple and consistent.

    This class has two main functionalities. Firstly, it defines an interface for saving and accessing metadata, which
    is a simple dictionary with all string keys.

    Secondly, it defines a metadata schema, which can be used to easily combine metadata with parameters passed
    explicitly. This in particular, accommodates the following usecase. Consider a class `ClsWithMeta` which
    extends `HasMetadata`. While `ClsWithMeta` provides a way to store arbitrary metadata, it also provides
    a simple way to store and access `'foo'` property, which is stored in the metadata.

    >>> from typing import List, Optional
    >>>
    >>> from cai_causal_graph.interfaces import HasMetadata
    >>> from cai_causal_graph.metadata_handler import MetaField
    >>>
    >>>
    >>> class ClsWithMeta(HasMetadata):
    >>>     def __init__(self, foo: Optional[int] = None, meta: Optional[dict] = None):
    >>>         meta = self._process_meta(meta=meta, kwargs_dict=dict(foo=foo))
    >>>         super().__init__(meta=meta)
    >>>
    >>>     @property
    >>>     def foo(self) -> Optional[int]:
    >>>         return self.meta.get('foo', None)
    >>>
    >>>     @foo.setter
    >>>     def foo(self, foo: Optional[int]):
    >>>         self.meta['foo'] = foo
    >>>
    >>>     @classmethod
    >>>     def get_metadata_schema(cls) -> List[MetaField]:
    >>>         return super().get_metadata_schema() + [MetaField(metatag='foo', property_name='foo')]

    Notice, that `ClsWithMeta` defines a metadata schema using the `get_metadata_schema` method, where it adds the
    `'foo'` metadata to the schema of its parent class. This in turn, enables easily extending the class and supports
    multiple inheritence, where the inheritence tree can get complex.

    The metadata schema can then be used to process metadata and combine it with explicit parameters. This is
    implemented by the `cai_causal_graph.interfaces.HasMetadata._process_meta` method, and works by checking the
    metadata schema and adding the matching parameters to the metadata. If a value is already set in the metadata, but
    is also defined in the parameter, the parameter takes precendence, meaning the value in the metadata is overwritten.

    While in the example above, the metadata schema is simple, it is also possible to define fields with tags not
    matching the property name, as well as not matching the provided parameter name. Moreover, it is possible to define
    tags with default values.
    """

    def __init__(self, meta: Optional[dict] = None):
        """
        Construct a `HasMetadata` instance with the provided meta.

        :param meta: Optional metadata dictionary. If provided, the dictionary is shallow-copied. If `None`, then
            an empty dictionary is created.
        """
        self.meta = meta.copy() if meta is not None else dict()
        assert isinstance(self.meta, dict) and all(
            isinstance(k, str) for k in self.meta
        ), 'Metadata must be provided as a dictionary with strings as keys.'

    @property
    def metadata(self) -> Optional[dict]:
        """A metadata property."""
        return self.meta

    def get_metadata(self) -> Optional[dict]:
        """Return metadata that is a dictionary."""
        return self.metadata

    @classmethod
    def get_metadata_schema(cls) -> List[MetaField]:
        """
        Get the schema for metadata of this class.

        To support complex class inheritance structures, it is recommended to return a sum of metadata of a parent
        class and this class, i.e. `return super().get_metadata_schema() + [...]`.
        """
        return []

    @classmethod
    def _process_meta(
        cls, meta: Optional[dict], kwargs_dict: dict, raise_if_unknown_tags: bool = False
    ) -> Optional[dict]:
        """
        Combine metadata with explicit keyword arguments.

        The metadata is combined with keyword arguments defined in the `kwargs_dict`. This is done using the
        following rules.

        First, the metadata schema of this class is constructed and validated.

        Secondly, a `requested_meta` dictionary is constructed by parsing the `kwargs_dict`. Any key-value pair
        is popped from `kwargs_dict` where a key appears as a `parameter_name` of a field in the metadata schema. Then
        this key is updated to the corresponding metadata tag of that field and added to the `requested_meta`.

        For any field in the schema which defines a default value, that default value is added to the `requested_meta`
        under the key matching the metadata tag of that field, if this metadata tag is not already present in the
        `meta`.

        Finally, provided `meta` dictionary is updated with the `requested_meta` dictionary. If any value in `meta`
        is overwritten, then a debug message is logged. If `requeste_meta` dictionary is empty and `meta` is `None`,
        then `None` is returned.

        :param meta: Optional dictionary containing metadata.
        :param kwargs_dict: A dictionary containing keyword arguments.
        :param raise_if_unknown_tags: Raise if `kwargs_dict` contains a key which does not correspond to any
            parameter names defined in the metadata schema.
        :return: An optional dictionary containing resolved metadata.
        """
        schema = cls.get_metadata_schema()
        cls._validate_schema(schema=schema)

        meta_dict = meta if meta is not None else dict()

        requested_meta = {
            field.metatag: kwargs_dict.pop(field.parameter_name, meta_dict.get(field.metatag, field.default_value))
            for field in schema
        }

        if raise_if_unknown_tags and len(kwargs_dict) > 0:
            raise MetaDataError(f'Unknown keyword arguments {kwargs_dict}. Metadata schema is {schema}.')

        return cls._update_metadata(meta=copy(meta), **requested_meta)

    @classmethod
    def _validate_schema(cls, schema: List[MetaField]):
        """
        Validate metadata schema of this class.

        This ensures that all metatags, property names and parameter names are unique for this class. If any
        duplicates are found, then a `cai_causal_graph.metadata_handler.MetaDataError` is raised.
        """
        tags, properties, parameters = list(
            zip(*[[field.metatag, field.property_name, field.parameter_name] for field in schema])
        )

        for l, name in [(tags, 'metatags'), (properties, 'property names'), (parameters, 'parameter names')]:
            if len(set(l)) != len(l):
                raise MetaDataError(f'Found multiple meta fields with identical {name} in schema: {schema}.')

    @classmethod
    def _update_metadata(cls, meta: Optional[dict], **kwargs) -> Optional[dict]:
        """
        Update metadata dictionary with the provided keyword arguments.

        This method logs (on debug level) any instances where meta is overwritten.
        """
        if meta is None:
            meta = dict()

        for k, v in kwargs.items():
            if v is not None:
                if (existing_v := meta.get(k, None)) is not None:
                    logger.debug(
                        f'{k} (set to {existing_v}) in the metadata of {cls.__name__} is '
                        f'overwritten by the newly provided value {v}.'
                    )
                meta[k] = v

        if len(meta) == 0:
            return None

        return meta
