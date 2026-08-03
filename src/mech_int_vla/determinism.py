"""Stable, cross-process deterministic seed and assignment helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


def _json_default(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: getattr(value, field.name) for field in fields(value)}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot canonically hash {type(value).__name__}")


def hash_seed(namespace: str, *parts: object) -> int:
    """Derive a PyTorch-compatible nonnegative seed using canonical SHA-256."""

    payload = json.dumps(
        [namespace, *parts],
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        default=_json_default,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def hash_order(values: Iterable[T], namespace: str, *parts: object) -> tuple[T, ...]:
    """Return a stable pseudorandom ordering without relying on Python's hash()."""

    return tuple(
        sorted(
            values, key=lambda value: (hash_seed(namespace, *parts, value), str(value))
        )
    )
