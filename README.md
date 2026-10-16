`safe-debates`
=====
_A PyTorch implementation of AI safety via debate on sparse MNIST_

[AI safety via debate](https://arxiv.org/abs/1805.00899) (Irving, Christiano & Amodei, 2018)
proposes training AI systems by having two agents debate in front of a weaker judge: if
honesty is the winning strategy, a judge that cannot verify answers itself can still elicit
true answers. The paper's MNIST experiment makes this concrete. A judge classifier sees only
a handful of pixels. Two agents take turns choosing which pixels to reveal: an honest agent
that wants the judge to output the true digit, and a liar that wants it to be wrong. The
question is how much debate improves the judge over seeing the same number of random pixels(see [Changes from the original implementation](#changes-from-the-original-implementation));
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

<!-- * **Judge** ([`safe_debates/judge.py`](safe_debates/judge.py)): a CNN that takes a
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
  builds the final judge input and scores both agents. -->

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


## Running debates

```
python main.py --pixels 6 --checkpoint checkpoint/6px-best.pth.tar --no-precommit \
               --batch-size 32 --rollouts 1000 --rounds 10 --seed 0
```

<!-- ## Changes from the original implementation

The original MCTS never worked, so the refactor had to settle what "intended behaviour"
means. Each fix below is the smallest change consistent with the code's evident intent:
names such as `N`, `Q`, `P` and `calculate_UCT`, the comments, and the paper.
[`tests/reference_mcts.py`](tests/reference_mcts.py) writes the fixed algorithm in the
original's own style: recursive rollouts, and a dict keyed by flattened state tensors.
The optimized search is tested to give **identical** moves and root visit counts to that
reference (`tests/test_mcts.py::test_matches_reference`). Defects in the original are
pinned down by tests in `tests/test_legacy_parity.py`. -->

<!-- ## Performance

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
bookkeeping, or batching more games per call. -->


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
