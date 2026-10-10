"""Attribute-style view over a frozen Gym recipe, used at the RSL seam."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


class ConfigNode(SimpleNamespace):
    def get(self, name: str, default: Any = None) -> Any:
        return getattr(self, name, default)

    @classmethod
    def from_value(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return cls(**{key: cls.from_value(item) for key, item in value.items()})
        if isinstance(value, list):
            return [cls.from_value(item) for item in value]
        return value
