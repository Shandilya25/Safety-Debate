`safe-debates`
=====
_A PyTorch implementation of AI safety via debate on sparse MNIST_

[AI safety via debate](https://arxiv.org/abs/1805.00899) (Irving, Christiano & Amodei, 2018)
proposes training AI systems by having two agents debate in front of a weaker judge: if
honesty is the winning strategy, a judge that cannot verify answers itself can still elicit
true answers. The paper's MNIST experiment makes this concrete. A judge classifier sees only
a handful of pixels. Two agents take turns choosing which pixels to reveal: an honest agent
that wants the judge to output the true digit, and a liar that wants it to be wrong. The
question is how much debate improves the judge over seeing the same number of random pixels.

This repository implements that experiment: judge training, and debates played by Monte
Carlo tree search (MCTS) agents. It started from an unmaintained prototype whose search did
not work (see [Changes from the original implementation](#changes-from-the-original-implementation));
the original code is kept unchanged in [`legacy/`](legacy/) for comparison.

## Debate architecture

```
                 ┌──────────── debate over k moves (liar moves first by default) ───────────┐
image, label ──► │ move i: mover = first if i even else second                              │
                 │   MCTS from the current position ─► reveal 1 nonzero pixel               │
                 └──────────────────────────────────────────────────────────────────────────┘
                                         │ k revealed pixels
                                         ▼
                 judge(mask, values) ─► argmax ─► honest wins iff prediction == label
```

* **Judge** ([`safe_debates/judge.py`](safe_debates/judge.py)): a CNN that takes a
  `(2, 28, 28)` input, where channel 0 is the reveal mask and channel 1 the revealed pixel
  values. It is trained by [`train_judge.py`](train_judge.py) on `k` nonzero pixels chosen
  uniformly at random.
* **State and actions**: a position is the set of pixels revealed so far. An action reveals
  one not-yet-revealed **nonzero** pixel. The debate ends after `--pixels` reveals in total.
* **Agents** ([`safe_debates/agent.py`](safe_debates/agent.py)): before each move, the
  agent to move runs `--rollouts` MCTS rollouts from the current position with a fresh tree.
  It then plays its most-visited root action.
* **MCTS** ([`safe_debates/mcts.py`](safe_debates/mcts.py)):
  * Each rollout selects actions down to the end of the debate using PUCT:
    `Q(s,a) + c_puct · P · sqrt(N(s)) / (1 + N(s,a))`, with a uniform prior
    `P = 1 / #valid actions` and ties broken at random.
  * The rollout then calls the judge once on the terminal states. The searching agent's win
    (1 or 0) is backed up negamax-style: `+w` at the searcher's levels, `−w` at the
    opponent's levels. Q is the running mean of backed-up values.
  * Nodes are keyed by (game, set of revealed pixels), so transpositions share statistics.
  * All `B` games in a batch are searched together, with one batched judge call per rollout.
* **Debate** ([`safe_debates/debate.py`](safe_debates/debate.py)): alternates the agents,
  builds the final judge input and scores both agents.

## Repository structure

```
safe_debates/          refactored implementation
  judge.py             Judge CNN (state_dict-compatible with the original)
  data.py              SparseMNIST dataset, DataLoaders, sparse_state() judge-input builder
  mcts.py              batched MCTS: SearchTree (tensor rows + node table) and MCTS.search
  agent.py, debate.py  debate protocol
  metrics.py, utils.py meters, per-class stats, seeding, checkpoints
train_judge.py         train the judge
main.py                run debates with a trained judge
tests/                 pytest suite, including parity tests against legacy/ and a reference MCTS
benchmarks/            bench_mcts.py (search speed and memory), bench_data.py (data/training)
legacy/                original implementation, unmodified (does not run on current PyTorch)
profiles/              cProfile traces of the original implementation
```

## Setup

```
pip install -r requirements.txt     # tested: Python 3.11, torch 2.4.0, torchvision 0.19.0
python -m pytest tests              # ~40 s on CPU, needs no dataset
```

MNIST is downloaded to `--data-folder` (default `./data/`) on first use.

## Training the judge

```
python train_judge.py --pixels 6 --seed 4224 --checkpoint-filename 6px
python train_judge.py --pixels 4 --batches 50000 --seed 4224 --checkpoint-filename 4px
```

Training is a single pass over `--batches` freshly sampled batches. The model is
validated once at the end, saved to `<checkpoint>/<name>-best.pth.tar`, and then tested.

| option | default | meaning |
| --- | --- | --- |
| `--pixels` | required | nonzero pixels revealed per training sample |
| `--batch-size` / `--eval-batch-size` | 128 / 1000 | training / evaluation batch size |
| `--batches` | 30000 | training batches |
| `--val-batches` / `--test-batches` | 4 / 800 | evaluation batches (validation is drawn from the training split) |
| `--lr` | 1e-4 | Adam learning rate |
| `--workers` | 1 | DataLoader workers. More is faster (see Performance) but changes the pixels drawn for a given seed |
| `--seed` | None | seeds Python, NumPy and torch; runs are bit-reproducible on a fixed machine and worker count |
| `--checkpoint` / `--checkpoint-filename` | `./checkpoint/` / datetime | output location |
| `--log-interval`, `--data-folder`, `--no-cuda` | | |

## Running debates

```
python main.py --pixels 6 --checkpoint checkpoint/6px-best.pth.tar --no-precommit \
               --batch-size 32 --rollouts 1000 --rounds 10 --seed 0
```

It prints the honest agent's win rate per round, then the overall and per-digit rates.
Without precommit, the honest win rate equals the judge's accuracy on the debated pixels.

| option | default | meaning |
| --- | --- | --- |
| `--pixels` | required | total pixels revealed per debate (should match the judge) |
| `--checkpoint` | required | judge checkpoint from `train_judge.py` |
| `--no-precommit` | off | **currently required**: precommit is not implemented and the script exits without it |
| `--rollouts` | 10000 | MCTS rollouts per move |
| `--c-puct` | 1.0 | PUCT exploration constant |
| `--batch-size` | 128 | debates played simultaneously |
| `--rounds` | 10000 // batch size | batches of debates (MNIST test split, shuffled) |
| `--honest-first` | off | honest agent moves first (default: liar first) |
| `--seed`, `--no-cuda`, `--data-folder` | | |
| `--progress` | off | progress bar over rollouts |
| `--profile` | off | write cProfile stats to `./profiles/` (the original always profiled) |

Cost scales as rollouts × moves × batched judge calls. On a 14-thread CPU, one 4-pixel
first move takes about 65 s for one game at 10,000 rollouts, and about 4 s for 128 games
at 100 rollouts. A full run at the defaults (128 × 10,000) is a multi-day CPU job, so
use a GPU or fewer rollouts.

## Changes from the original implementation

The original MCTS never worked, so the refactor had to settle what "intended behaviour"
means. Each fix below is the smallest change consistent with the code's evident intent:
names such as `N`, `Q`, `P` and `calculate_UCT`, the comments, and the paper.
[`tests/reference_mcts.py`](tests/reference_mcts.py) writes the fixed algorithm in the
original's own style: recursive rollouts, and a dict keyed by flattened state tensors.
The optimized search is tested to give **identical** moves and root visit counts to that
reference (`tests/test_mcts.py::test_matches_reference`). Defects in the original are
pinned down by tests in `tests/test_legacy_parity.py`.

**Search (the original's debates were degenerate):**
1. *Statistics were never stored.* `self.Q[states][:, a] = ...` assigns into the temporary
   tensor returned by `BatchDict.__getitem__`. In the profiles, `__missing__` runs on every
   lookup (e.g. 140,000,400 of 140,000,400). N and Q stayed zero, so every move was action 0.
2. *Exploration was disabled.* `calculate_UCT` divided by the number of revealed pixels,
   which gives NaN at the root, and `break_ties` returned an argmax *index* that was then
   added to Q as a constant. Now: PUCT with a uniform prior over valid actions and random
   tie-breaking, with `c_puct` exposed (default 1, matching the original's absent constant).
3. *Rollouts corrupted the root.* `apply_action` mutated the state in place, so every rollout
   revealed more pixels in the real game state. It also swapped row and column, and with
   B > 1 wrote every game's action into every game. States are now never mutated.
4. *Channel aliasing.* The initial state was `zeros(...).expand(-1, 2, -1, -1)`, so the
   mask and value channels shared memory.
5. *Action space.* Valid actions were computed over both channels (1568 entries), so
   actions ≥ 784 indexed outside the image. Invalid actions could also win the argmax,
   because masking was done by multiplying by 0. Actions are now the 784 pixels, with
   invalid ones masked to −∞.
6. *Depth and sign.* Rollouts always went `pixels + 1` moves deep from any position, and
   the negamax sign at the root was wrong for even `pixels`. Rollouts now end when the
   debate ends, and the searcher's levels back up `+w`.
7. *Shared roots.* Keys were state contents only, so all games in a batch shared the
   (empty) root's statistics. Keys now include the game index.
8. `choose_moves` could select an invalid action. It now takes the most-visited *valid*
   action (first index on ties, as `torch.max` did).

**Data and training:**
9. `SparseMNIST` used `sparse[~mask] = 0` with a uint8 mask. That was a logical NOT on
   PyTorch 0.4, but is bitwise or an error on current PyTorch. Values are now `img * mask`.
   The pixel sampling call is unchanged, so a seed draws the same pixels as the original
   (tested).
10. Current torchvision looks for a subclass's data under `root/SparseMNIST/`, which
    would download a second copy of MNIST. `SparseMNIST` now reads `root/MNIST/`.
11. `train_judge.py` crashed at test time with `--val-batches 0` (no `-best` file); it
    now always saves the final model as best. Deprecated `size_average=False` became
    `reduction='sum'`. Datetime checkpoint names no longer contain `:` (invalid on Windows).

**Evaluation:**
12. `main.py`: added the missing `numpy` import (it crashed with `--seed`), `map_location`
    for checkpoint loading, and a fail-fast for precommit. Previously the default run crashed
    with a `TypeError` in `track_stats_with` after the first round.
13. Per-digit accuracy counted every sample in the batch in each digit's denominator, and
    the total counted each sample twice. Now: total = mean honest win rate, and digit d's
    rate is computed over samples labelled d.

**Unchanged on purpose:**
* The judge architecture, including the missing ReLU after `hidden`, inherited from
  cle-mnist, so that checkpoints and reported accuracies stay valid.
* Single-pass training and one validation at the end.
* A fresh search tree per move, liar first by default, and default hyperparameters.

## Performance

Measured on this repository's development machine (Windows 11, CPU only, torch 2.4.0,
14 intra-op threads). The machine was noisy: repeated runs varied by up to about 2×.
Implementations were therefore interleaved, and medians of 3 runs are reported. Reproduce
with `python benchmarks/bench_mcts.py` and `python benchmarks/bench_data.py`. Judge
weights are randomly initialised, which does not affect cost.

**One first move (4 pixels, 200 rollouts, MNIST test images), wall time:**

| batch | original (`legacy/`, patched `/`→`//` to run) | reference: fixed algorithm, original dict | refactor |
| --- | --- | --- | --- |
| 1 | 3.09 s | 2.42 s | **1.17 s** (2.6× vs original) |
| 10 | 80.5 s | 13.4 s | **1.97 s** (41× vs original, 6.8× vs reference) |

The original column understates its true cost, because it stored no statistics and so
did less work than a working search. At B=128 (100 rollouts) the refactor takes 4.0 s
per move. One game at 10,000 rollouts runs at about 150 rollouts/s.

| bottleneck (evidence) | change | measured effect / trade-off |
| --- | --- | --- |
| `BatchDict` lookups: `tolist()` of a 1568-float state, hashing a 1568-tuple, `__missing__` allocation and `cat`, about 7× per tree level per game (58% of runtime in `profiles/*.20rnds.10000roll.1bs`, 97% in `*.1rnds.10000roll.20bs`) | Nodes keyed by `(game, sorted revealed-pixel indices)`, small int tuples, in one dict. Statistics in preallocated tensors indexed by node id | Main source of the table above. A dict remains: node ids still need hashing, but keys are about 4 ints instead of 1568 floats |
| Original per-action Python update loop (`# TODO: vectorize this`) | One batched indexed update per tree level | Part of the table above |
| B×784 statistics per node | Candidate action space: each image's nonzero pixels (K ≈ 100–300) | Fewer elements in every PUCT op. Exactly equivalent, since zero pixels are never valid |
| Dense rows for nodes visited once (most deep nodes) | A node gets a dense row on its 2nd visit; until then its one edge is stored inline | Tree memory at B=128, 100 rollouts: 9.2 MB vs 75.6 MB all-dense. B=1, 10k rollouts: 7.2 MB vs 14.4 MB. Tested equal to dense storage |
| Per-node metadata as tensors: many tiny masked ops per level (profiled) | Metadata in Python lists, dense rows in tensors | Interleaved A/B: 9.7→6.6 ms/rollout (B=1), 13.3→10.3 (B=10), 28.3→26.0 (B=64). Wins at every batch size tested; tensor metadata might win at much larger B |
| Judge called without `no_grad` | Search and debate run under `torch.no_grad()` | **No measurable speed change** (judge forward 0.96 vs 0.91 ms at B=1, 8.1 vs 8.0 ms at B=128). Kept because it avoids building autograd graphs |
| GPU: CPU tree, GPU judge | Tree stays on the CPU. One state upload and one result download per rollout | **Unverified**: no GPU was available. The original also synced per lookup |
| `SparseMNIST._get_mask` per-pixel Python loop | Single index assignment | Not a meaningful bottleneck (41–95 vs 70–78 µs per sample across runs; within noise). Kept for clarity |
| `train_judge.py` waits on data with 1 worker | `--workers` option, default 1 | Batch (data wait): 112 ms (28 ms) with 1 worker vs 90 ms (2 ms) with 4. Not the default, because it changes the pixels drawn for a seed |

At B=1, the remaining search cost is mostly fixed per-op overhead of small torch ops, plus
the judge forward (about 40% under the profiler). Possible next steps are NumPy for B=1
bookkeeping, or batching more games per call.

## Experimental results

**Judge (sparse classifier on random pixels).** These were reported by the original
author on PyTorch 0.4. Judge training was unaffected by the search bugs, though
`--val-batches` affects which file is tested.

Density | Accuracy | Average Cross-Entropy
--- | --- | ---
6px | 57.4% | 1.1948
4px | 46.7% | 1.4775

Rerun with the refactored code (torch 2.4, CPU, 54 min): `python train_judge.py --pixels 6
--seed 4224 --checkpoint-filename 6px` gives **57.35%** test accuracy and **1.1982** average
cross-entropy on 800,000 test samples. That matches the original to within sampling noise.
The 4px judge was not rerun.

**Debates (small-scale sanity check, not a full experiment).** Run with the 6px judge
above, no precommit, liar first, 300 rollouts per move, and 192 test images (3 rounds × 64,
`--seed 0`). The honest agent won **96.4%** of debates, against the judge's 57.35% on 6
random pixels. This is the first working debate result in this repository: the original
never produced meaningful debates (see above). Full-scale runs (10,000 rollouts, all
10,000 test images, precommit) have not been done.

## Known limitations and open TODOs

* **Precommit is not implemented** (missing feature). This is the paper's main setting:
  each agent commits to a label, and the judge picks between the two claims. `main.py`
  requires `--no-precommit`. `Agent.precommit_` and `Debate.evaluate` raise
  `NotImplementedError` if commits are set.
* No agent learning: agents are pure search against a fixed judge, as in the paper's MNIST
  experiment.
* A fresh tree per move (subtree reuse would be an algorithmic change).
* The judge architecture deviates from the TF tutorial; see above.
* Validation runs once, at the end of training, so "best" checkpoint selection is trivial.
* Exact numbers from the original runs cannot be reproduced bit-for-bit: the legacy code
  does not run on current PyTorch, and RNG streams differ across PyTorch versions.
* The GPU path has not been tested.

## References and attribution

* G. Irving, P. Christiano, D. Amodei. *AI safety via debate*. 2018.
  [arXiv:1805.00899](https://arxiv.org/abs/1805.00899) ·
  [blog post](https://blog.openai.com/debate/) ·
  [debate game](https://debate-game.openai.com/)
* Based on [`cle-mnist`](https://github.com/jvmncs/cle-mnist) (J. Mancuso). From upstream:
  the `Judge` network (upstream `ConvNet` with 2 input channels instead of 1), the
  structure of `train_judge.py` (adapted from upstream `main.py`), and `save_checkpoint` /
  `mkdir_p`.
* PUCT selection follows AlphaGo Zero / AlphaZero (Silver et al., 2017).

If you use this repository, please reference it. MIT License, see [`LICENSE`](LICENSE).
