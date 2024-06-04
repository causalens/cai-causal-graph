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
