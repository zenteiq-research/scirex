<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# Source Module Changes Made During Test Writing

This file records every change made to production source modules while writing
the Phase 1 layer tests. Each entry shows the file, the exact lines removed,
and the exact lines added, so changes can be cleanly reverted if needed.

---

## 1. `src/scirex/_internal/scaling.py` (formerly `src/scirex/utils.py`)

### Why
`validate_scaling_factor` only accepted `list` inputs for sequence-valued
scaling factors. Flax frozen dataclasses internally convert Python `list`
fields to `tuple` when modules are constructed inside another module (e.g.
`SpectralConv` created inside `FNOBlocks.setup()`). This caused
`_resolution_scaling_factor` to always be `None` when FNOBlocks passed a
list-valued factor to SpectralConv, silently disabling resolution scaling.

### Location
Function `validate_scaling_factor`, immediately after the `None` guard.

### Change

**Lines removed (original):**
```python
    if scaling_factor is None:
        return None
    if isinstance(scaling_factor, (float, int)):
```

**Lines added (replacement):**
```python
    if scaling_factor is None:
        return None
    # Flax frozen dataclasses serialise list fields as tuples; normalise here.
    if isinstance(scaling_factor, tuple):
        scaling_factor = list(scaling_factor)
    if isinstance(scaling_factor, (float, int)):
```

### How to revert
Delete the two lines:
```python
    # Flax frozen dataclasses serialise list fields as tuples; normalise here.
    if isinstance(scaling_factor, tuple):
        scaling_factor = list(scaling_factor)
```

---

## 2. `src/scirex/layers/common/normalization.py`

### Why
`AdaIN.__call__` called `jnp.split(self._mlp(embedding), self.in_channels, axis=0)`.
`jnp.split(..., N)` splits into `N` equal pieces; the MLP outputs
`2 * in_channels` values, so this produced `in_channels` arrays instead of 2.
Unpacking into `weight, bias` always raised `ValueError: too many values to unpack`.
The correct call splits into exactly 2 halves of size `in_channels` each.

### Location
`AdaIN.__call__`, line 67 (original).

### Change

**Line removed (original):**
```python
        weight, bias = jnp.split(self._mlp(embedding), self.in_channels, axis=0)
```

**Line added (replacement):**
```python
        weight, bias = jnp.split(self._mlp(embedding), 2, axis=0)
```

### How to revert
Change `2` back to `self.in_channels` on that line.

---

## 3. `src/scirex/layers/common/normalization.py` (second change)

### Why
`AdaIN.__call__` instantiated `nn.GroupNorm(...)` inline during the forward
pass. Flax requires all sub-modules to be created in `setup()` (or inside a
`@compact` method); creating them in `__call__` raises
`AssignSubModuleError`. The fix moves the `GroupNorm` construction into
`setup()` as `self._group_norm`, and replaces the inline call with
`self._group_norm(x)`.

### Location
`AdaIN.setup()` and `AdaIN.__call__`, `normalization.py`.

### Change

**Lines removed (original):**
```python
    def setup(self):
        if self.mlp is None:
            self._mlp = nn.Sequential([...])
        else:
            self._mlp = self.mlp
        # (no _group_norm here)

        # in __call__:
        x = nn.GroupNorm(num_groups=self.in_channels, epsilon=self.eps)(x)
```

**Lines added (replacement):**
```python
    def setup(self):
        if self.mlp is None:
            self._mlp = nn.Sequential([...])
        else:
            self._mlp = self.mlp
        self._group_norm = nn.GroupNorm(num_groups=self.in_channels, epsilon=self.eps)

        # in __call__:
        x = self._group_norm(x)
```

### How to revert
- Remove `self._group_norm = nn.GroupNorm(...)` from `setup()`.
- Restore `x = nn.GroupNorm(num_groups=self.in_channels, epsilon=self.eps)(x)` in `__call__`.

---

## 4. `src/scirex/layers/common/normalization.py` (third change)

### Why
Flax's `GroupNorm` normalises over features-last layout by default. All
tensors in this codebase use channel-first layout `(batch, channels, *spatial)`.
Without transposing, GroupNorm treats the last spatial dimension as the
feature axis and fails when `num_groups=out_channels` does not divide the
spatial size. The fix transposes to features-last before GroupNorm and back
to channel-first after.

### Location
`AdaIN.__call__`, the GroupNorm application block.

### Change

**Lines removed (original after change #3):**
```python
        # group_norm with num_groups=in_channels is equivalent to instance norm
        x = self._group_norm(x)
```

**Lines added (replacement):**
```python
        # GroupNorm expects features-last; transpose in/out for channel-first tensors
        x = jnp.moveaxis(x, 1, -1)
        x = self._group_norm(x)
        x = jnp.moveaxis(x, -1, 1)
```

### How to revert
Replace the three lines back with `x = self._group_norm(x)` (and remove the comment).

---

---

## 5. `src/scirex/layers/common/padding.py`

### Why
`DomainPadding.unpad` stored unpad indices in `self._unpad_indices` (a plain Python
dict mutated via `object.__setattr__`) during `pad()`. Flax constructs a fresh module
instance on every `apply()` call, so the dict populated in one `apply(pad)` call is
gone by the time `apply(unpad)` runs, causing a `KeyError`.

### Location
`DomainPadding.unpad`, `padding.py`.

### Change

**Lines removed (original):**
```python
    def unpad(self, x: jax.Array) -> jax.Array:
        """Remove the padding from padding inputs"""
        unpad_indices = self._unpad_indices[f"{list(x.shape[2:])}"]
        return x[unpad_indices]
```

**Lines added (replacement):**
```python
    def unpad(self, x: jax.Array) -> jax.Array:
        """Remove the padding from padding inputs"""
        # Flax creates a fresh module on every apply(), so the _unpad_indices cache
        # populated during pad() is not available here.  Recompute slices directly.
        resolution = x.shape[2:]
        domain_padding = self.domain_padding
        if isinstance(domain_padding, (float, int)):
            domain_padding = [float(domain_padding)] * len(resolution)

        resolution_scaling_factor = self.resolution_scaling_factor
        if not isinstance(resolution_scaling_factor, list):
            resolution_scaling_factor = validate_scaling_factor(
                resolution_scaling_factor, len(resolution), n_layers=None
            )

        unpad_list = []
        for s, p_frac, sf in zip(resolution, domain_padding, resolution_scaling_factor):
            if p_frac == 0.0:
                unpad_list.append(slice(None))
            else:
                orig = round(s / (1 + 2 * p_frac * sf))
                pad = (s - orig) // 2
                unpad_list.append(slice(pad, pad + orig))

        return x[(slice(None), slice(None), *unpad_list)]
```

### How to revert
Replace the entire `unpad` method body with the single-line original:
```python
        unpad_indices = self._unpad_indices[f"{list(x.shape[2:])}"]
        return x[unpad_indices]
```

---

---

## 6. `src/scirex/layers/common/embeddings.py`

### Why
`GridEmbedding2D.setup()` used `self._grid = None` and `self._res = None`, then
`grid()` updated them via `self._grid = ...` during `apply()`. Flax freezes module
attributes after `setup()` returns, so any subsequent assignment raises
`SetAttributeFrozenModuleError`. `GridEmbeddingND` already used `object.__setattr__`
to bypass this; `GridEmbedding2D` needed the same pattern.

### Location
`GridEmbedding2D.setup()` and `GridEmbedding2D.grid()`, `embeddings.py`.

### Change

**Lines removed (original):**
```python
    def setup(self):
        self._grid = None
        self._res = None
    ...
            self._grid = grid_x, grid_y
            self._res = spatial_dims
```

**Lines added (replacement):**
```python
    def setup(self):
        object.__setattr__(self, '_grid', None)
        object.__setattr__(self, '_res', None)
    ...
            object.__setattr__(self, '_grid', (grid_x, grid_y))
            object.__setattr__(self, '_res', spatial_dims)
```

### How to revert
In `setup()`: restore `self._grid = None` and `self._res = None`.
In `grid()`: restore `self._grid = grid_x, grid_y` and `self._res = spatial_dims`.

---

---

## 7. `src/scirex/layers/blocks/gno_block.py`

### Why
`GNOBlock.__call__` passed the raw output of `NeighborSearch` (keys:
`neighbors_index`, `neighbors_row_splits`) directly to `IntegralTransform`,
which expects keys `segment_ids` and `counts`. This caused a `KeyError` on
`neighbors["segment_ids"]` whenever `GNOBlock` was called with
`neighbors=None` (i.e. letting it run its own neighbor search).

Additionally, a leftover debug `print` statement was present in the
`neighbors is None` branch.

### Location
`GNOBlock.__call__`, lines ~228–230, `gno_block.py`.

### Change

**Lines removed (original):**
```python
        if neighbors is None:
            neighbors_dict = self.neighbor_search(data=y, queries=x, radius=self.radius)
            print('neighbors_dict in neighbors is None', neighbors_dict)
        else:
            neighbors_dict = neighbors
```

**Lines added (replacement):**
```python
        if neighbors is None:
            raw = self.neighbor_search(data=y, queries=x, radius=self.radius)
            row_splits = raw["neighbors_row_splits"]
            counts = (row_splits[1:] - row_splits[:-1]).astype(jnp.int32)
            import jax.numpy as jnp as _jnp  # already imported at top; shown for clarity
            segment_ids = jnp.repeat(jnp.arange(counts.shape[0], dtype=jnp.int32), counts)
            neighbors_dict = {
                "neighbors_index": raw["neighbors_index"],
                "segment_ids": segment_ids,
                "counts": counts,
            }
            if "weights" in raw:
                neighbors_dict["weights"] = raw["weights"]
        else:
            neighbors_dict = neighbors
```

### How to revert
Restore:
```python
        if neighbors is None:
            neighbors_dict = self.neighbor_search(data=y, queries=x, radius=self.radius)
            print('neighbors_dict in neighbors is None', neighbors_dict)
        else:
            neighbors_dict = neighbors
```

---

## 8. `src/scirex/layers/blocks/integral_transform.py`

### Why
`IntegralTransform.__call__` applied `jnp.expand_dims(jnp.expand_dims(nbr_weights, -1), 0)`
unconditionally, adding both a trailing feature dim and a leading batch dim to `nbr_weights`.
When `rep_features` is unbatched (`[n_edges, C]`), multiplying by `[1, n_edges, 1]` broadcasts
to `[1, n_edges, C]`, which `segment_csr` treats as a batch of size 1, producing output
`[1, n_out, C]` instead of `[n_out, C]`. The leading `expand_dims` should only be applied when
`rep_features` is already batched (3D).

### Location
`IntegralTransform.__call__`, the `nbr_weights is not None` block, `integral_transform.py`.

### Change

**Line removed (original):**
```python
            nbr_weights = jnp.expand_dims(jnp.expand_dims(nbr_weights, -1), 0)
```

**Lines added (replacement):**
```python
            nbr_weights = jnp.expand_dims(nbr_weights, -1)
            if rep_features.ndim == 3:
                nbr_weights = jnp.expand_dims(nbr_weights, 0)
```

### How to revert
Restore the single line:
```python
            nbr_weights = jnp.expand_dims(jnp.expand_dims(nbr_weights, -1), 0)
```

---

## Summary Table

| File | Nature | Revertible by |
|------|--------|---------------|
| `src/scirex/_internal/scaling.py` | Bug fix - tuple → list normalisation in `validate_scaling_factor` | Delete 2 added lines |
| `src/scirex/layers/common/normalization.py` | Bug fix - AdaIN split argument `self.in_channels` → `2` | Restore `self.in_channels` |
| `src/scirex/layers/common/normalization.py` | Bug fix - move `nn.GroupNorm` from `__call__` into `setup()` | Remove `self._group_norm` from `setup()`, restore inline `nn.GroupNorm(...)` call |
| `src/scirex/layers/common/normalization.py` | Bug fix - add moveaxis transpose around GroupNorm for channel-first tensors | Remove the two `jnp.moveaxis` lines |
| `src/scirex/layers/common/padding.py` | Bug fix - recompute unpad slices directly; Flax apply() clears cached dict state | Restore single-line `self._unpad_indices` lookup in `unpad` |
| `src/scirex/layers/common/embeddings.py` | Bug fix - `GridEmbedding2D` used plain assignment in frozen module; replaced with `object.__setattr__` | Restore `self._grid = ...` / `self._res = ...` direct assignments |
| `src/scirex/layers/blocks/gno_block.py` | Bug fix - convert `neighbors_row_splits` → `segment_ids` + `counts` for IntegralTransform; remove debug print | Restore original 3-line `if neighbors is None` block |
| `src/scirex/layers/blocks/integral_transform.py` | Bug fix - `expand_dims(..., 0)` on `nbr_weights` always added a batch dim even for unbatched `rep_features`; now only expands if `rep_features` is already 3D | Restore `jnp.expand_dims(jnp.expand_dims(nbr_weights, -1), 0)` |
