"""Readers shared by scripts for simulator and ecg_sigma HDF5 files.

Simulator files store ``/metadata`` fields as datasets; ecg_sigma stores
them as attributes with UTF-8 bytes strings.  Both are accepted.
"""

from __future__ import annotations

from typing import Any, Sequence

import h5py
import numpy as np

from .classes import decode_str


def _decode(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, (bytes, np.bytes_)):
        return decode_str(value)
    return value


def read_metadata_field(hf: h5py.File, key: str, default: Any = None) -> Any:
    """Read ``/metadata/<key>`` from an attribute or a dataset."""
    meta = hf.get("metadata")
    if meta is None:
        return default
    if key in meta.attrs:
        return _decode(meta.attrs[key])
    if key in meta and isinstance(meta[key], h5py.Dataset):
        return _decode(meta[key][()])
    return default


def event_keys(hf: h5py.File) -> list[str]:
    return sorted(k for k in hf.keys() if k.startswith("event_"))


def read_ecg_leads(ecg_grp: h5py.Group, leads: Sequence[str]) -> np.ndarray:
    """Stack *leads* into ``(len(leads), T)`` float32; missing leads are zeros."""
    arrays: dict[str, np.ndarray] = {}
    length = None
    for lead in leads:
        if lead in ecg_grp and isinstance(ecg_grp[lead], h5py.Dataset):
            arrays[lead] = np.asarray(ecg_grp[lead][:], dtype=np.float32)
            length = len(arrays[lead]) if length is None else length
    if length is None:
        raise KeyError(f"none of the leads {list(leads)} found in {ecg_grp.name}")
    out = np.zeros((len(leads), length), dtype=np.float32)
    for i, lead in enumerate(leads):
        if lead in arrays:
            out[i, : len(arrays[lead])] = arrays[lead][:length]
    return out
