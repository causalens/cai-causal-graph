"""
Copyright (c) 2024 by Impulse Innovations Ltd. Private and confidential. Part of the causaLens product.
"""
import abc
import logging
from copy import copy
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)


class MetaDataError(Exception):
    ...


@dataclass
class MetaField:
    metatag: str
    property_name: str
    parameter_name: Optional[str] = None
    default_value: Optional[Any] = None

    def __post_init__(self):
        if self.parameter_name is None:
            self.parameter_name = self.property_name


class HasMeta(abc.ABC):
    def __init__(self, meta: Optional[dict]):
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


def access_meta(f: callable) -> property:
    @wraps(f)
    def getter(self) -> Any:
        if not isinstance(self, HasMeta):
            raise TypeError(f'Cannot automatically access meta of {self} since it does not extend {HasMeta} class.')
        field: MetaField = self._get_field_for_property_name(f.__name__)

        if field.default_value is None:
            return self.meta.get(field.metatag, None)
        else:
            return self.meta[field.metatag]

    def setter(self, val: Any):
        if not isinstance(self, HasMeta):
            raise TypeError(f'Cannot automatically access meta of {self} since it does not extend {HasMeta} class.')
        field: MetaField = self._get_field_for_property_name(f.__name__)

        self.meta[field.metatag] = val

    return property(getter, setter)
