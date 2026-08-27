"""
KDA state surgery for KimiDynamicCache.

The whole transplant reduces to moving tensors between four plain Python lists:

    cache.recurrent_states[i]   # the KDA S matrices  <- what we carry over
    cache.conv_states[i]        # tuple (q, k, v), short-conv, kernel 4
    cache.key_cache[i]          # MLA KV
    cache.value_cache[i]

`KimiDynamicCache.get_seq_length()` reads `key_cache` only and returns 0 when it is None,
so a cache holding transplanted S with a flushed KV honestly reports length 0 and run B
takes the normal position-0 prefill path. That is what makes the S-only transplant work
without patching the model.
"""

from __future__ import annotations

import torch

COMPONENTS = ("recurrent_states", "conv_states", "key_cache", "value_cache")


def _clone(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().clone()
    if isinstance(x, (tuple, list)):
        return type(x)(_clone(v) for v in x)
    raise TypeError(f"unexpected cache entry type: {type(x)}")


def snapshot(cache) -> dict:
    """Deep-copy every cache component. Detached clones, device preserved."""
    return {name: [_clone(x) for x in getattr(cache, name)] for name in COMPONENTS}


def state_nbytes(snap: dict) -> int:
    """Bytes held by recurrent_states only -- ~20 MiB at bf16, 40 MiB as actually stored."""
    total = 0
    for s in snap["recurrent_states"]:
        if isinstance(s, torch.Tensor):
            total += s.numel() * s.element_size()
    return total


def restore(cache_cls, config, snap: dict, keep=("recurrent_states",), device=None):
    """
    Build a fresh cache and load only the components named in `keep`.

    keep=COMPONENTS                -> full restore, used by the exactness test
    keep=("recurrent_states",)     -> the S-only transplant: KV and conv left None
    """
    unknown = set(keep) - set(COMPONENTS)
    if unknown:
        raise ValueError(f"unknown cache components: {sorted(unknown)}")

    cache = cache_cls(config)
    for name in keep:
        src = snap[name]
        dst = getattr(cache, name)
        if len(src) != len(dst):
            raise ValueError(
                f"{name}: snapshot has {len(src)} layers, fresh cache has {len(dst)}"
            )
        for i, x in enumerate(src):
            x = _clone(x)
            if device is not None and x is not None:
                x = _to_device(x, device)
            dst[i] = x
    return cache


def _to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return type(x)(_to_device(v, device) for v in x)


def describe(cache_or_snap, label="") -> str:
    """One-line-per-component summary. Handy when a test fails and you need to see why."""
    snap = cache_or_snap if isinstance(cache_or_snap, dict) else snapshot(cache_or_snap)
    lines = [f"--- {label} ---"] if label else []
    for name in COMPONENTS:
        entries = snap[name]
        filled = [i for i, x in enumerate(entries) if x is not None]
        shapes = []
        for i in filled[:3]:
            x = entries[i]
            shapes.append(
                tuple(x.shape) if isinstance(x, torch.Tensor)
                else [tuple(t.shape) for t in x]
            )
        lines.append(f"{name:17s} filled={len(filled):2d}/{len(entries):2d} layers={filled} e.g.={shapes}")
    return "\n".join(lines)
