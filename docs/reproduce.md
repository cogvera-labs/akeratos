# Reproduce Billet and its twin

These commands rebuild Billet's training data, train Billet and its twin, and score them.

## What you need

- A Mac with Apple silicon and macOS 14.0 or later, the oldest macOS that [MLX supports](https://ml-explore.github.io/mlx/build/html/install.html). Every training run has been on one MacBook Pro with an M4 Max (40 GPU cores, 48 GB).
- About 57 GB of free disk to build the data, plus the checkpoints. The data ends as a 28.5 GB download and a 16.4 GB store.
- [uv](https://docs.astral.sh/uv/) 0.13.6 or newer.

## Training time and memory

| Run | What it trains | Tokens | Steps' time on the M4 Max | Peak memory |
|---|---|---|---|---|
| Billet | the 162M core with a table of 6.0B numbers | 3.2B | 107.6 h | 24.1 GiB |
| Its twin | the same core with the table off | 3.2B | 111.1 h | 12.4 GiB |
| The starter | the 10M model with a table of 1.0B numbers | 75M | 41 min | 6.7 GiB |

Peak memory is MLX's peak reading from the training log, in GiB (2^30 bytes). macOS, your other programs and MLX's buffer cache need memory beside it; `bake train` caps the cache at 3.81 GiB, and 2 GiB for the starter. [Billet and its twin](billet.md) has more on both runs.

## Your Mac

The other commands, on the same Mac:

| Command | Time | Peak memory |
|---|---|---|
| `bake bench` | 74 s | 6.2 GiB |
| `bake speedrun` | 300 s of steps | 7.6 GiB |
| `bake generate` on Billet | | 13.0 GiB |
| `bake evaluate` on Billet's release | 36 min | |
| `bake evaluate` on the twin's release | 18 min | |

8 GB Macs are untested: [a bench row](https://github.com/cogvera-labs/akeratos/issues/new?template=bench.yml) from yours is wanted.

## 1. Set up

```sh
curl -LsSf https://astral.sh/uv/0.13.6/install.sh | sh
git clone https://github.com/cogvera-labs/akeratos.git
cd akeratos
uv sync --frozen
```

Run every command from the repository root. `uv run bake bench` checks the setup without any data: it trains a 10M model on synthetic token ids ([the Mac speedrun](speedrun.md#bake-bench)).

## A first run

Start with the starter: the 10M model with a table of 1.0B numbers, on 75M tokens of [the speedrun's](speedrun.md) training slice. It goes through the same trainer as Billet and needs no data build.

```sh
uv run --frozen --no-sync hf download Cogvera-Labs/Akeratos-Speedrun-5min --repo-type dataset --revision 4fadb133baf45efe8dd764c2a172ff92e5ee2482 --local-dir ~/mf-speedrun-5min
uv run --frozen --no-sync bake train starter --data ~/mf-speedrun-5min/train --out ~/mf-runs/starter
uv run --frozen --no-sync bake evaluate --ckpt ~/mf-runs/starter --data ~/mf-speedrun-5min/heldout --compute-dtype float32 --records ~/mf-runs/starter/records --out ~/mf-runs/starter/heldout.json
```

The original run last printed a training loss of 4.07 and scores 1.2449 bits per byte on the 7,314 held-out documents Billet is scored on. `--seed N` starts any of the three configurations from another seed.

To read what it writes, export the last checkpoint, with Billet's tokenizer and notice, and prompt it:

```sh
uv run --frozen --no-sync hf download Cogvera-Labs/Akeratos-Billet tokenizer.json NOTICE --local-dir ~/mf-runs/starter-files
uv run --frozen --no-sync bake export ~/mf-runs/starter/ckpt-00009156 --out ~/mf-runs/starter-export --tokenizer ~/mf-runs/starter-files/tokenizer.json --licence LICENSE --notice ~/mf-runs/starter-files/NOTICE
uv run --frozen --no-sync bake generate ~/mf-runs/starter-export --prompt "The capital of France is"
```

The original run's model answers ` the capital of the world.`, again and again.

## 2. Build the training data

The data is [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), tokenised with SmolLM2's tokenizer. [`scripts/data/prepare_store.py`](https://github.com/cogvera-labs/akeratos/blob/main/scripts/data/prepare_store.py) builds the store. It pins both inputs, checks every downloaded file against its published hash, never runs downloaded code and never sends a Hugging Face token. It runs in an environment of its own, built from a hash-locked requirements file:

```sh
uv venv --managed-python --python 3.12 ~/.venvs/mf-data
uv pip install --python ~/.venvs/mf-data/bin/python --require-hashes -r scripts/data/requirements.txt
```

Keep `--managed-python`: the script refuses a Python that loads extra modules at start-up, as Homebrew's Python does. Then run its four steps:

```sh
~/.venvs/mf-data/bin/python -I -B scripts/data/prepare_store.py check --sample sample/10BT --files 14 --out ~/mf-data --cap-tokens 7800000000
~/.venvs/mf-data/bin/python -I -B scripts/data/prepare_store.py fetch --sample sample/10BT --files 14 --out ~/mf-data --cap-tokens 7800000000 --confirm-pins 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 93efa2f097d58c2a74874c7e644dbc9b0cee75a2
~/.venvs/mf-data/bin/python -I -B scripts/data/prepare_store.py prepare --sample sample/10BT --out ~/mf-data --cap-tokens 7800000000
~/.venvs/mf-data/bin/python -I -B scripts/data/prepare_store.py verify --store ~/mf-data/9545f2d205efd574
```

- `check` reads the pinned file lists and the Hub's scan flags, and prints the disk space it needs.
- `fetch` downloads the 14 Parquet files of `sample/10BT` and the tokenizer at the two pinned commits. It stops on any file the Hub's scanner has not cleared. On 2026-10-19, 13 of the 14 read `unscanned` and were allowed by name, with one `--allow-unsafe-file sample/10BT/000_00000.parquet` flag per file, from `000` to `012`.
- `prepare` tokenises offline, in about 18 minutes on the M4 Max, and resumes after a stop.
- `verify` re-hashes every file and exits 0 when the store matches its manifest.

The store's name is a hash of its inputs, so the same inputs give `9545f2d205efd574`. Billet's store holds 7,918,559,792 training tokens in 59 shards and 7,643,425 held-out tokens in 7,314 documents, and its `manifest.json` has the SHA-256 `503beba3d802bfc2302d20ed31ceba167e8040e7573e1fad02373ae73fd2ba70`.

## 3. Train

```sh
uv run --frozen --no-sync bake train billet --data ~/mf-data/9545f2d205efd574/train --out ~/mf-runs/billet --guard
uv run --frozen --no-sync bake train billet-twin --data ~/mf-data/9545f2d205efd574/train --out ~/mf-runs/billet-twin --guard
```

- `bake train` starts each run from the configuration it recorded, as both were launched: 2 rows of 2,048 tokens, 32 accumulated, the attention's banded path, a checkpoint every 240 steps, and 3 kept beside the one pinned at the start of the learning-rate decay.
- `--guard` runs the trainer under the Mac guard, which stops it cleanly on battery, heat, memory pressure or low disk.
- A run writes `run.json`, `metrics.jsonl` and its checkpoints to `--out`. The same command again resumes from the newest complete checkpoint and refuses a changed setting.
- A Billet checkpoint is 16.5 GB: 12.1 GB of table and 4.4 GB of dense weights, their moving average and the optimiser state. Its four take 66 GB. The twin's checkpoint holds no table.
- The code as trained ships beside each model's weights.

## 4. Score

```sh
uv run --frozen --no-sync bake evaluate --ckpt ~/mf-runs/billet --data ~/mf-data/9545f2d205efd574/heldout --compute-dtype float32 --records ~/mf-runs/billet/records --out ~/mf-runs/billet/heldout.json
```

It scores the newest checkpoint on every held-out document, in bits per byte, with the raw weights and with their moving average. For Billet it also scores the model with the table switched off. Score the twin with the same command and its own folders. `--ckpt` also takes a release folder from Hugging Face, which it checks against its `SHA256SUMS` first.

To check the published scores without training, score the released weights on the held-out split of [the first run's](#a-first-run) download:

```sh
uv run --frozen --no-sync hf download Cogvera-Labs/Akeratos-Billet --local-dir ~/mf-release/billet
uv run --frozen --no-sync bake evaluate --ckpt ~/mf-release/billet --data ~/mf-speedrun-5min/heldout --compute-dtype float32 --records ~/mf-release/billet-records --out ~/mf-release/billet-heldout.json
```

With the raw weights it gives 0.8866 bits per byte, in 36 minutes on the M4 Max. The twin, `Cogvera-Labs/Akeratos-Billet-Twin`, gives 0.9030 in 18.

## 5. Compare

```sh
uv run --frozen --no-sync python -I scripts/analysis/analysis.py --out report.json --heldout ~/mf-runs/billet/heldout.json ~/mf-runs/billet-twin/heldout.json --split ~/mf-data/9545f2d205efd574/heldout
```

`report.json` then holds the primary readout, `primary`, to set beside [the published one](https://github.com/cogvera-labs/akeratos/blob/main/results/billet/report.json). If yours differs, open an issue: that is a finding.
