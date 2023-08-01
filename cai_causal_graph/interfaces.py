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
from typing import Optional


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


class HasMetadata(abc.ABC):
    """An interface for an object that has metadata."""

    @property
    def metadata(self) -> Optional[dict]:
        """A metadata property."""
        return None

    def get_metadata(self) -> Optional[dict]:
        """Return metadata that is a dictionary."""
        return self.metadata
