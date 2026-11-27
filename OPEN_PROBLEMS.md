# Open problems

Questions Cogvera Labs wants answered and cannot answer alone. Each names the baseline that exists today, the metric, and what counts as an answer. A negative result is an answer: send it.

The metric everywhere is **held-out bits per byte**, which does not depend on the tokenizer. An effect counts only if it is larger than the run-to-run noise at that size. Ten seeds of [the speedrun](docs/speedrun.md)'s reference file, the 10M model trained for 5 minutes, gave a standard deviation of 0.0203 bits per byte. Billet and its twin had one seed each; an answer with two is stronger.

To take one, open a pull request with your result and its logs. [CONTRIBUTING.md](CONTRIBUTING.md) says what to send.

Every figure below is for a MacBook Pro with an M4 Max and 48 GB of memory, the only Mac any training run has used.

## Does the table pay off at equal wall-clock?

**Question.** The table adds work to every step, and Billet and its twin are compared at equal tokens. How does the comparison look at equal wall-clock time on the same Mac?
**Baseline.** Billet and its twin: their released checkpoints and training logs.
**Answer.** Held-out bits per byte against wall-clock for both arms on one stated Mac, with the step-time ratio measured, not assumed.
**On a Mac.** A rerun is two runs of 3.2B tokens; [Reproduce](docs/reproduce.md) gives the time and memory each took.

## The dose curve

**Question.** How does held-out bits per byte change with table size at a fixed core: full, 1/4, 1/16 and off?
**Baseline.** Billet, with the full table, and its twin, with none, are the two ends.
**Answer.** One point per size between them, on the same tokens, with the memory used. A curve that flattens early is as useful as one that keeps falling.
**On a Mac.** Each point is a run of Billet's length.

## Low-bit storage and rare rows

**Question.** How much do 8-bit and 4-bit table storage hurt the least-trained rows?
**Baseline.** The 16-bit table of [Akeratos Billet](https://huggingface.co/Cogvera-Labs/Akeratos-Billet)'s released checkpoint.
**Answer.** Held-out bits per byte for each format, and the same figure split by how often each row was read in training.
**On a Mac.** Evaluation only. Billet's table takes 11.2 GiB in memory.

## The table's cost on each Mac

**Question.** How long do the table's gather, update and write-back take per step on each Apple chip and memory size, and where does the time go?
**Baseline.** The M4 Max rows in [the bench rows](results/bench/README.md).
**Answer.** One row from a Mac not yet listed there: the step-time ratio, the table's bandwidth and peak memory.
**On a Mac.** `uv run bake bench` measures it in 10 minutes or less, on mains power, with a table of 1.0B numbers (1.9 GiB).
