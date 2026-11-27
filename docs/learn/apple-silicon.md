# Training on a Mac

Akeratos trains on Apple silicon Macs, not on rented data-centre GPUs.

## Unified memory

On a typical PC or server, the GPU has its own memory, separate from the CPU's, and data has to be copied across before the GPU can use it. A data-centre GPU has 80 to 192 GB of its own memory.

An Apple silicon chip has one pool of memory that the CPU and the GPU share (*unified memory*). Nothing is copied between them, and the GPU can use most of the machine's memory: tens of gigabytes on a laptop, hundreds on a Mac Studio. The trade-off is compute: a Mac's GPU does far fewer calculations per second than a data-centre GPU.

That combination, lots of memory and modest compute, is what Akeratos's design is built around. The [n-gram memory](n-gram-memory.md) costs memory but almost no compute: two table rows per token. A laptop can hold a 12 GB table next to a 162-million-parameter model, while the compute per token stays that of the small model.

Consumer hardware has the memory to hold a lot of knowledge, but not the compute to train ever larger general models, so Cogvera Labs works on small, specialised ones.

Every training run so far has been on one MacBook Pro with an M4 Max and 48 GB of memory.

## MLX

Akeratos is written on [MLX](https://github.com/ml-explore/mlx), Apple's open-source array framework for machine learning. It looks like NumPy, runs on the Mac's GPU through Metal (Apple's GPU programming interface), and also runs on an ordinary CPU.

Two properties of MLX shape the code:

- **Lazy evaluation.** An MLX operation does not compute anything when it is called; it records what to compute, and the work happens when a result is needed or when the code asks for it with `mx.eval`. The training step chooses those points with care: the gradients are evaluated before the optimiser update is built, so the memory held by the forward pass is freed first.
- **Compilation.** A function can be compiled into one fused program, which runs faster. A compiled step must keep the same shapes and must not depend on changing Python numbers, so values that change during a run, such as the learning rate, are passed in as arrays.

## Speed claims

Numbers measured on a CPU say whether the code is correct, never how fast it is. Every speed figure in Akeratos's documents comes from a run on the Mac, and says which machine and which commit it was measured on. Kernels written for the Mac's GPU, such as the fused attention kernels, are checked against the plain version on the Mac before they are used, and must agree to within twice the plain version's own rounding error.

## Keeping a long run safe

A training run on the laptop takes days. The trainer runs under a guard that checks the machine every few seconds and logs what it sees:

- **Power.** On battery for a minute, the guard stops the trainer cleanly, which saves its place; when the charger is back, it restarts the run.
- **Heat and memory.** It stops the run for good if the chip stays too hot for too long or memory pressure turns critical, and leaves a note saying why. Swap use only raises a warning.
- **Disk.** It refuses to start without room for three checkpoints, and stops the run when fewer than two would fit.
- **Sleep.** It keeps the Mac awake while the run is in progress.

A stopped run resumes from its last checkpoint.

## Running the code without a Mac

Everything except real training runs on Linux: the unit tests, the numerical checks, and short CPU runs of the training loop. MLX's CPU backend gives the same answers as the GPU within tolerance, and bit for bit on repeated CPU runs.
