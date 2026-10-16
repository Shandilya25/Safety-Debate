"""
Time one MCTS move (the first, deepest search) for three implementations on MNIST test images:

  legacy     original legacy/agent.py (only ``/`` -> ``//`` patched so it runs on torch>=1.5;
             note it stores no statistics, so it does *less* work than the others)
  reference  tests/reference_mcts.py: same algorithm as the new code, original data
             structure (dict keyed by flattened state tuples)
  new        safe_debates.mcts

Implementations are interleaved and repeated; the median wall time is reported.

Usage: python benchmarks/bench_mcts.py [--batch-sizes 1 10] [--rollouts 200] [--repeat 3]
"""
import argparse
import os
import statistics
import sys
import time
import types

import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'tests')]

from conftest import import_legacy  # noqa: E402
from reference_mcts import ReferenceAgent  # noqa: E402
from safe_debates.data import sparse_state  # noqa: E402
from safe_debates.judge import Judge  # noqa: E402
from safe_debates.mcts import MCTS  # noqa: E402


def legacy_agent(pixels, rollouts):
    agent_mod = import_legacy('agent')

    def apply_action(cls, action, sparse, images):  # original body with floor division
        width = images.size(-1)
        action_x, action_y = action % width, action // width
        sparse[:, 0, action_x, action_y] = 1
        sparse[:, 1, action_x, action_y] = images[:, 0, action_x, action_y]
        return sparse
    agent_mod.Agent.apply_action = classmethod(apply_action)
    import tqdm
    agent_mod.tqdm.tqdm = lambda it, **kw: it
    args = types.SimpleNamespace(pixels=pixels, rollouts=rollouts, device=torch.device('cpu'))
    agent = agent_mod.Agent(honest=True, args=args)
    agent.precommit_(None, None)
    return agent


def run(impl, judge, images, labels, pixels, rollouts, seed=0):
    B = images.size(0)
    torch.manual_seed(seed)
    start = time.perf_counter()
    extra = ''
    if impl == 'legacy':
        agent = legacy_agent(pixels - 1, rollouts)  # original searches depth pixels+1
        state = torch.zeros(B, 2, 28, 28)
        agent.simulate_(judge, state, images, labels)
    elif impl == 'reference':
        agent = ReferenceAgent(True, pixels, rollouts)
        with torch.no_grad():
            agent.simulate_(judge, sparse_state(images, torch.zeros(B, 0, dtype=torch.long)),
                            images, labels)
        extra = 'dict entries={}'.format(len(agent.N) + len(agent.Q))
    else:
        mcts = MCTS(rollouts)
        mcts.search(judge, images, labels, torch.zeros(B, 0, dtype=torch.long), pixels, True)
        t = mcts.tree
        dense = t.num_nodes * t.num_actions * 8
        extra = 'nodes={} dense rows={} tree={:.1f} MB (all-dense would be {:.1f} MB)'.format(
            t.num_nodes, t.num_rows, t.nbytes() / 2 ** 20, dense / 2 ** 20)
    return time.perf_counter() - start, extra


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 10])
    parser.add_argument('--rollouts', type=int, default=200)
    parser.add_argument('--pixels', type=int, default=4)
    parser.add_argument('--impls', nargs='+', default=['legacy', 'reference', 'new'])
    parser.add_argument('--threads', type=int, default=None)
    parser.add_argument('--repeat', type=int, default=3)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    data = MNIST(os.path.join(ROOT, 'data'), train=False, transform=ToTensor())
    torch.manual_seed(0)
    judge = Judge().eval()
    print('torch {} | threads {} | pixels {} | rollouts {}'.format(
        torch.__version__, torch.get_num_threads(), args.pixels, args.rollouts))
    for B in args.batch_sizes:
        images = torch.stack([data[i][0] for i in range(B)])
        labels = torch.tensor([data[i][1] for i in range(B)])
        run('new', judge, images, labels, args.pixels, 5)  # warm up
        times, extras = {impl: [] for impl in args.impls}, {}
        for rep in range(args.repeat):
            for impl in args.impls:
                seconds, extras[impl] = run(impl, judge, images, labels, args.pixels,
                                            args.rollouts, seed=rep)
                times[impl].append(seconds)
        for impl in args.impls:
            seconds = statistics.median(times[impl])
            print('B={:<4d} {:<10s} {:8.2f} s/move  {:9.1f} rollouts/s  {}'.format(
                B, impl, seconds, args.rollouts / seconds, extras[impl]), flush=True)

if __name__ == '__main__':
    main()
