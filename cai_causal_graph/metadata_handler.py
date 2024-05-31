"""
Copyright (c) 2024 by Impulse Innovations Ltd. Private and confidential. Part of the causaLens product.
"""
import abc
from copy import copy
from dataclasses import dataclass
from typing import Any, List, Optional


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
    def __init__(self, meta: dict):
        self.meta = meta

    @classmethod
    def get_metadata_schema(cls) -> List[MetaField]:
        return []

    def _process_meta(self, meta: Optional[dict], **kwargs) -> Optional[dict]:
        schema = self.get_metadata_schema()
        self._validate_schema(schema=schema)

        requested_meta = {field.metatag: kwargs.get(field.parameter_name, field.default_value) for field in schema}

        return self._update_metadata(meta=copy(meta), **requested_meta)

    def _validate_schema(self, schema: List[MetaField]):
        return
        tags, properties, parameters = list(
            zip(*[[field.metatag, field.property_name, field.parameter_name] for field in schema])
        )

        for l, name in [(tags, 'metatags'), (properties, 'property names'), (parameters, 'parameter names')]:
            if len(set(l)) != len(l):
                raise MetaDataError(f'Found multiple meta fields with identical {name} in schema: {schema}.')

    def _update_metadata(self, meta: Optional[dict], **kwargs) -> dict:
        if meta is None:
            meta = dict()

        for k, v in kwargs.items():
            if v is not None:
                if (existing_v := meta.get(k, None)) is not None:
                    if existing_v != v:
                        raise MetaDataError(
                            f'Cannot set {k} in metadata of {self}, because it already exists in provided '
                            f'metadata as {existing_v}.'
                        )
                meta[k] = v

        return meta
