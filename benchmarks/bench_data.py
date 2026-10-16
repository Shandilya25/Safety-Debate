"""
SparseMNIST sampling cost (original per-pixel loop vs vectorized mask) and judge training
throughput for different DataLoader worker counts.

Usage: python benchmarks/bench_data.py [--pixels 4] [--batches 200]
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'tests')]

from conftest import import_legacy  # noqa: E402
from safe_debates.data import SparseMNIST  # noqa: E402
from safe_debates.judge import Judge  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pixels', type=int, default=4)
    parser.add_argument('--batches', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    data = os.path.join(ROOT, 'data')
    new = SparseMNIST(args.pixels, args.batches * args.batch_size, data, train=True)
    legacy_mask = import_legacy('prepare_data').SparseMNIST._get_mask

    imgs = [new.data[i].unsqueeze(0).float() / 255 for i in range(2000)]
    new[0]
    for name, fn in (('legacy _get_mask', lambda img: legacy_mask(new, img)),
                     ('new _get_mask', new._get_mask),
                     ('new __getitem__', None)):
        torch.manual_seed(0)
        start = time.perf_counter()
        for i, img in enumerate(imgs):
            fn(img) if fn else new[i]
        print('{:<18s} {:7.1f} us/sample'.format(name, (time.perf_counter() - start) / len(imgs) * 1e6))

    for workers in (0, 1, 4):
        torch.manual_seed(0)
        model = Judge()
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        loader = DataLoader(new, batch_size=args.batch_size, num_workers=workers)
        load = 0.
        tick = start = time.perf_counter()
        for ix, (sparse, _, label) in enumerate(loader):
            if ix == 0:  # exclude worker start-up (slow process spawn on Windows)
                tick = start = time.perf_counter()
            load += time.perf_counter() - tick
            opt.zero_grad()
            F.cross_entropy(model(sparse), label).backward()
            opt.step()
            tick = time.perf_counter()
        total = time.perf_counter() - start
        n = args.batches - 1
        print('workers={}: {:6.1f} ms/batch, of which waiting for data {:5.1f} ms'.format(
            workers, total / n * 1e3, load / n * 1e3), flush=True)


if __name__ == '__main__':
    main()
