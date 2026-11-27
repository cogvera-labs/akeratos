# MLX notes

Akeratos is built on [MLX](https://github.com/ml-explore/mlx), Apple's array framework for machine learning. This page records the MLX versions Akeratos uses, the MLX problems it works around, where it relies on MLX's low-level interfaces, and where it stands on MLX's CUDA build.

## Versions

- [`pyproject.toml`](https://github.com/cogvera-labs/akeratos/blob/main/pyproject.toml) accepts `mlx>=0.32.4,<0.33` on macOS, and the same range of `mlx[cpu]` on Linux, where the tests run on MLX's CPU backend.
- The lock file pins MLX 0.32.4, which `uv sync --frozen` installs. Each training run records its MLX version in its `run.json`.
- Billet and its twin trained on MLX 0.32.3.

## Problems worked around

| Problem | Upstream | Status on 2026-11-27 | What Akeratos does |
|---|---|---|---|
| On the CPU, in MLX 0.32.3, a reduction over a whole array of more than 2^31 elements returns the identity, such as a sum of 0 | [mlx#4549](https://github.com/ml-explore/mlx/pull/4549) | Fixed in 0.32.4 | Keeps every table shard below 2^31 elements |
| On the CPU and on Metal, a reduction whose reduced axes together hold more than 2^31 elements returns the identity, in 0.32.4 too | [mlx#4888](https://github.com/ml-explore/mlx/pull/4888) | Fixed in 0.32.5 | The same limit |
| `mx.fast.cross_entropy` returns wrong gradients for large non-contiguous inputs on Metal | [mlx#4686](https://github.com/ml-explore/mlx/issues/4686) | Fixed in 0.32.5 by [mlx#4702](https://github.com/ml-explore/mlx/pull/4702) | Makes the logits contiguous before the loss |

The shard limit, 2^31 − 2^20 elements, is `SHARD_LIMIT` in [`akeratos/table/config.py`](https://github.com/cogvera-labs/akeratos/blob/main/akeratos/table/config.py). The cross-entropy guard is `mx.contiguous` in [`akeratos/train/loop.py`](https://github.com/cogvera-labs/akeratos/blob/main/akeratos/train/loop.py) and in `speedrun/train.py`.

## Low-level interfaces

- **Custom Metal kernels.** MLX 0.32.3 computes no fused attention gradient on the GPU: while gradients are traced, its fused attention falls back to plain operations. Akeratos's own fused attention is three Metal kernels, compiled with `mx.fast.metal_kernel` and wrapped in one `mx.custom_function`, in [`akeratos/flash.py`](https://github.com/cogvera-labs/akeratos/blob/main/akeratos/flash.py). `bake bench` and the speedrun use them where they fit, on the GPU in bfloat16. Billet and its twin ran without them. Elsewhere the same formulas run as ordinary MLX operations. MLX 0.32.4 adds a gradient for its own fused attention ([mlx#4563](https://github.com/ml-explore/mlx/pull/4563)).
- **`mx.metal`** is not used.
- **Memory settings.** [`akeratos/macops.py`](https://github.com/cogvera-labs/akeratos/blob/main/akeratos/macops.py) caps MLX's buffer cache with `mx.set_cache_limit`, at 3.81 GiB for Billet and its twin, and never sets the wired limit. The memory counters are read in `akeratos/_compat/mlx_seam.py`.
- **TF32.** The commands set `MLX_ENABLE_TF32=0` before MLX loads, because MLX reads it once per process.

## MLX's CUDA build

MLX also ships a CUDA build for Linux with NVIDIA GPUs ([MLX's install notes](https://ml-explore.github.io/mlx/build/html/install.html)). Akeratos targets Apple silicon and has no CUDA backend. No test or training run has used the CUDA build. `pyproject.toml` has a `cuda13` extra that installs it on Linux; nothing uses it.
