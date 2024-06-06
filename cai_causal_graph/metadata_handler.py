"""
Copyright (c) 2024 by Impulse Innovations Ltd. Private and confidential. Part of the causaLens product.
"""
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class MetaField:
    """
    A class describing a single field in metadata of the `cai_causal_graph.interfaces.HasMetadata` class.

    :param metatag: A string defining the key used for storing the value of this field in the metadata.
    :param property_name: A string defining the name of the property used to access this metadata field.
    :param parameter_name: A string defining the name of the parameter which is used to set this metadata field. If
        not defined, this will default to the `property_name`.
    :param default_value: Default value for this field. If `None`, then no default value is specified.
    """

    metatag: str
    property_name: str
    parameter_name: Optional[str] = None
    default_value: Optional[Any] = None

    def __post_init__(self):
        """
        Custom constructor for the dataclass.

        Sets the `parameter_name` to `property_name` if it is not defined.
        """
        if self.parameter_name is None:
            self.parameter_name = self.property_name
