"""
Copyright (c) 2024 by Impulse Innovations Ltd. Private and confidential. Part of the causaLens product.
"""
import abc
from dataclasses import dataclass
from typing import Any, List, Optional


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
    @abc.abstractmethod
    def get_metadata_schema(cls) -> List[MetaField]:
        pass

    def _process_meta(self, meta: Optional[dict], **kwargs) -> Optional[dict]:
        schema = self.get_metadata_schema()

        requested_meta = {field.metatag: kwargs.get(field.parameter_name, field.default_value) for field in schema}

        return self._update_metadata(meta=meta, **requested_meta)

    @staticmethod
    def _update_metadata(meta: Optional[dict], **kwargs) -> dict:
        if meta is None:
            return kwargs
        else:
            for k, v in kwargs.items():
                if v is not None:
                    if (existing_v := meta.get(k, None)) is not None:
                        if existing_v != v:
                            raise ValueError(
                                f'Cannot set {k}, because it already exists in provided metadata as {existing_v}.'
                            )
                    meta[k] = v

            return meta
