import importlib
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGACY = os.path.join(ROOT, 'legacy')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def import_legacy(name):
    """Import a module from legacy/ (the original, unmodified implementation)."""
    sys.path.insert(0, LEGACY)
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(LEGACY)


def random_images(batch, nonzero, seed=0):
    """(batch, 1, 28, 28) images with exactly ``nonzero`` nonzero pixels each."""
    g = torch.Generator().manual_seed(seed)
    images = torch.zeros(batch, 784)
    for b in range(batch):
        pix = torch.randperm(784, generator=g)[:nonzero]
        images[b, pix] = torch.rand(nonzero, generator=g) * 0.9 + 0.1
    return images.view(batch, 1, 28, 28)


@pytest.fixture
def judge():
    from safe_debates.judge import Judge
    torch.manual_seed(0)
    return Judge().eval()
