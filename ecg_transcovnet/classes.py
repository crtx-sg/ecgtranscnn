"""Class-head specification: the ordered class names a model predicts.

The simulator path uses all 16 ``Condition`` enum members.  Real-data
packages (``ecgpkg``) ship a data-driven subset in ``package.json`` and
checkpoints record the head they were trained with in ``class_names``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .simulator.conditions import Condition

_ENUM_NAMES = frozenset(c.name for c in Condition)
_NAME_BY_VALUE = {c.value: c.name for c in Condition}

NOT_IN_HEAD = "n/a (not in head)"


def decode_str(value: Any) -> str:
    """Return *value* as ``str``, decoding HDF5 bytes attributes."""
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    return str(value)


def canonical_condition_name(raw: Any) -> str | None:
    """Map a stored ``condition`` to its ``Condition`` enum name.

    Simulator files store the enum *value* (``"V"``); ecg_sigma stores the
    enum *name* (``"PVC"``).  Returns ``None`` for anything else
    (e.g. ``"OTHER"``, ``"PACED"``).
    """
    if raw is None:
        return None
    text = decode_str(raw).strip()
    if text in _ENUM_NAMES:
        return text
    return _NAME_BY_VALUE.get(text)


@dataclass(frozen=True)
class ClassSpec:
    """Ordered class names of a model head."""

    names: tuple[str, ...]

    def __post_init__(self) -> None:
        names = tuple(self.names)
        object.__setattr__(self, "names", names)
        if not names:
            raise ValueError("ClassSpec needs at least one class")
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate class names in {names}")
        unknown = [n for n in names if n not in _ENUM_NAMES]
        if unknown:
            raise ValueError(f"class names are not Condition enum names: {unknown}")

    # -- construction ------------------------------------------------------

    @classmethod
    def default(cls) -> "ClassSpec":
        """All 16 ``Condition`` members in enum order (simulator head)."""
        return cls(tuple(c.name for c in Condition))

    @classmethod
    def from_names(cls, names: Iterable[str]) -> "ClassSpec":
        return cls(tuple(names))

    @classmethod
    def from_package(cls, path: str | Path) -> "ClassSpec":
        """Load from a package directory or its ``package.json``."""
        path = Path(path)
        if path.is_dir():
            path = path / "package.json"
        with open(path, encoding="utf-8") as f:
            return cls(tuple(json.load(f)["classes"]))

    @classmethod
    def from_checkpoint(cls, ckpt: Mapping[str, Any]) -> "ClassSpec":
        """Head of a checkpoint: ``class_names`` if saved, else the enum.

        Falls back to the 16-class enum only when the stored object queries
        agree with that count.
        """
        names = ckpt.get("class_names")
        if names:
            return cls(tuple(names))
        state = ckpt.get("model_state_dict", {})
        queries = state.get("object_queries")
        default = cls.default()
        if queries is not None and queries.shape[1] != len(default):
            raise ValueError(
                f"checkpoint has {queries.shape[1]} object queries but no class_names"
            )
        return default

    # -- lookup ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.names)

    def __iter__(self):
        return iter(self.names)

    def __contains__(self, name: object) -> bool:
        return name in self.names

    def index(self, name: str) -> int:
        return self.names.index(name)

    def get_index(self, name: str | None) -> int | None:
        if name is None or name not in self.names:
            return None
        return self.names.index(name)

    def resolve(self, raw: Any) -> tuple[str, int | None]:
        """Resolve a stored condition to ``(display_name, head_index)``.

        ``display_name`` is the enum name when resolvable, otherwise the raw
        text; ``head_index`` is ``None`` when the condition is not in the head.
        """
        name = canonical_condition_name(raw)
        display = name if name is not None else decode_str(raw)
        return display, self.get_index(name)
