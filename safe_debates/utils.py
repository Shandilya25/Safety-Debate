import os
import random
import shutil

import numpy as np
import torch


def seed_everything(seed):
    """Seed Python, NumPy and torch (DataLoader workers derive their seeds from torch)."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device(no_cuda):
    use_cuda = not no_cuda and torch.cuda.is_available()
    return use_cuda, torch.device("cuda" if use_cuda else "cpu")


def save_checkpoint(state, is_best, title, filename='checkpoint.pth.tar'):
    """Save ``state`` to ``<title>-<filename>``; copy to ``<title>-best.pth.tar`` if best.

    From cle-mnist.
    """
    filepath = title + '-' + filename
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, title + '-best.pth.tar')


def load_judge_state(path, device):
    """Load a judge ``state_dict`` from a train_judge.py checkpoint onto ``device``."""
    return torch.load(path, map_location=device, weights_only=True)['state_dict']


def mkdir_p(path):
    os.makedirs(path, exist_ok=True)
