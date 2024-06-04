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

from cai_causal_graph.metadata_handler import HasMeta, MetaDataError, MetaField

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


class HasMetadata(HasMeta):
    """An interface for an object that has metadata."""

    def __init__(self, *args, meta: Optional[dict] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta = meta.copy() if meta is not None else dict()
        assert isinstance(self.meta, dict) and all(
            isinstance(k, str) for k in self.meta
        ), 'Metadata must be provided as a dictionary with strings as keys.'

    @classmethod
    def get_metadata_schema(cls) -> List[MetaField]:
        return []

    @classmethod
    def _get_field_for_property_name(cls, property_name: str):
        schema = cls.get_metadata_schema()
        cls._validate_schema(schema=schema)

        matched_fields = list(filter(lambda field_: field_.property_name == property_name, schema))

        if len(matched_fields) == 0:
            raise KeyError(f'No metadata field with property name matching {property_name} in the schema {schema}.')
        elif len(matched_fields) > 1:
            # This should never be raised in reality since schema is validated.
            raise MetaDataError(f'Found multiple fields {matched_fields} with identical property name.')
        else:
            return matched_fields[0]

    @classmethod
    def _process_meta(
        cls, meta: Optional[dict], kwargs_dict: dict, raise_if_unknown_tags: bool = False
    ) -> Optional[dict]:
        schema = cls.get_metadata_schema()
        cls._validate_schema(schema=schema)

        requested_meta = {field.metatag: kwargs_dict.pop(field.parameter_name, field.default_value) for field in schema}

        if raise_if_unknown_tags and len(kwargs_dict) > 0:
            raise MetaDataError(f'Unknown keyword arguments {kwargs_dict}. Metadata schema is {schema}.')

        return cls._update_metadata(meta=copy(meta), **requested_meta)

    @classmethod
    def _validate_schema(cls, schema: List[MetaField]):
        tags, properties, parameters = list(
            zip(*[[field.metatag, field.property_name, field.parameter_name] for field in schema])
        )

        for l, name in [(tags, 'metatags'), (properties, 'property names'), (parameters, 'parameter names')]:
            if len(set(l)) != len(l):
                raise MetaDataError(f'Found multiple meta fields with identical {name} in schema: {schema}.')

    @classmethod
    def _update_metadata(cls, meta: Optional[dict], **kwargs) -> Optional[dict]:
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

    @property
    def metadata(self) -> Optional[dict]:
        """A metadata property."""
        return self.meta

    def get_metadata(self) -> Optional[dict]:
        """Return metadata that is a dictionary."""
        return self.metadata
