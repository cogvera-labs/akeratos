# Bench rows

What the n-gram table costs per training step on each Mac, for [the open problem on that cost](../../OPEN_PROBLEMS.md#the-tables-cost-on-each-mac): one row per hardware class and MLX version, each measured by `uv run bake bench` and kept as a JSON file in this folder, `<class>-mlx<version>.json`. The bench section of [the Mac speedrun page](../../docs/speedrun.md) says how to make a row and send it. `uv run python speedrun/records.py --write` generates this page from the rows.

Every row runs shape version 1: the `core-10m` model, 2 sequences of 2,048 tokens per step and a table of 1,000,000,000 numbers, 10 warm-up steps and then 300 timed steps with the table off and again with it on, seed 0, and MLX's buffer cache capped at 2 GiB. Step times are medians over the timed steps. The table's bandwidth is an estimate: the table rows it updated per step times the bytes of each, over the step time.

| Chip | GPU cores | Memory | macOS | Form | MLX | Step off (ms) | Step on (ms) | On/off | Table GB/s (estimate) | Peak on (GiB) | Person |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Apple M4 Max | 40 | 48 GB | 26.6.2 | laptop | 0.32.4 | 98.1 | 115.5 | 1.178 | 0.0323 | 6.24 | [BhodiSea](https://github.com/BhodiSea) |
| Apple M4 Max | 40 | 128 GB | 26.6.2 | desktop | 0.32.4 | 97.4 | 111.9 | 1.148 | 0.0334 | 6.24 | [brendanboyle87](https://github.com/brendanboyle87) |
| Apple M6 | 12 | 16 GB | 27.2.1 | desktop | 0.32.4 | 168.9 | 186.6 | 1.105 | 0.02 | 6.24 | [BhodiSea](https://github.com/BhodiSea) |
