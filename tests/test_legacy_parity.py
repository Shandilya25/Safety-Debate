"""
Checks of the refactor against the original implementation in ``legacy/``.

Where the original behaves as intended, outputs must match exactly.  Where it is defective,
the test pins down the defect so the documented behavior change is visible and reviewable.
"""
import os

import pytest
import torch
from torch.utils.data import TensorDataset

from conftest import import_legacy, random_images
from safe_debates.data import SparseMNIST
from safe_debates.debate import Debate
from safe_debates.judge import Judge


def _fake_mnist(cls, *args, **kwargs):
    """Instantiate an MNIST subclass without touching the filesystem."""
    obj = cls.__new__(cls)
    torch.utils.data.Dataset.__init__(obj)
    obj.train, obj.transform, obj.target_transform = True, None, None
    images = (random_images(20, 60, seed=5).squeeze(1) * 255).to(torch.uint8)
    obj.data, obj.targets = images, torch.arange(20) % 10
    obj.pixels, obj._n, obj._generator = args[0], args[1], torch.default_generator
    return obj


def test_judge_state_dict_compatible_and_identical():
    legacy = import_legacy('judge').Judge()
    new = Judge()
    new.load_state_dict(legacy.state_dict())
    x = torch.rand(5, 2, 28, 28)
    legacy.eval(), new.eval()
    assert torch.equal(legacy(x), new(x))


def test_sparse_mnist_same_pixels_for_same_seed():
    """Same RNG stream (pixel choices) as the original for a given seed."""
    legacy_cls = import_legacy('prepare_data').SparseMNIST
    old, new = _fake_mnist(legacy_cls, 4, 20), _fake_mnist(SparseMNIST, 4, 20)
    # the original __getitem__ no longer runs: ``sparse[~mask]`` with a uint8 mask
    with pytest.raises(RuntimeError, match='boolean'):
        old[0]
    for ix in range(20):
        img = old.data[ix].unsqueeze(0).type(torch.float32) / 255
        torch.manual_seed(ix)
        old_mask, _ = old._get_mask(img)
        torch.manual_seed(ix)
        new_sparse, new_img, new_t = new[ix]
        assert torch.equal(old_mask.type(torch.float32), new_sparse[:1])   # identical mask
        assert new_sparse[0].sum() == 4
        assert torch.equal(new_img, img) and new_t == old.targets[ix]
        assert torch.equal(new_sparse[1], img[0] * new_sparse[0])


def test_sparse_mnist_uses_mnist_folder():
    obj = SparseMNIST.__new__(SparseMNIST)
    obj.root = 'data'
    assert obj.raw_folder == os.path.join('data', 'MNIST', 'raw')


def test_legacy_batchdict_drops_updates():
    """Why the original search never learned: chained assignment hits a temporary."""
    BatchDict = import_legacy('utilities').BatchDict
    stats = BatchDict(torch.device('cpu'))
    state = torch.zeros(1, 2, 28, 28)
    stats[state][:, 3] = 5.
    assert len(stats) == 0 and stats[state][0, 3] == 0
    assert stats[state].shape == (1, 1568)  # also: two channels' worth of actions


def test_legacy_apply_action_defects_and_fix():
    images = random_images(2, 784, seed=0)
    action = torch.tensor([1 * 28 + 2, 5 * 28 + 6])     # (row 1, col 2) and (row 5, col 6)
    legacy_agent = import_legacy('agent')
    with pytest.raises(IndexError):                      # true division on torch >= 1.5
        legacy_agent.Agent.apply_action(action, torch.zeros(2, 2, 28, 28), images)

    state = torch.zeros(2, 2, 28, 28)
    new = Debate.apply_action(action, state, images)
    assert state.sum() == 0                              # parent state not mutated
    assert new[:, 0].flatten(1).sum(1).tolist() == [1., 1.]   # one pixel per game
    assert new[0, 0, 1, 2] == 1 and new[1, 0, 5, 6] == 1      # row-major, not transposed
    assert new[0, 1, 1, 2] == images[0, 0, 1, 2]


def test_legacy_initial_state_aliases_channels():
    images = torch.rand(1, 1, 28, 28)
    sparse = torch.zeros_like(images).expand(-1, 2, -1, -1)
    sparse[:, 0, 3, 3] = 1
    assert sparse[0, 1, 3, 3] == 1                       # mask write leaks into values


def test_legacy_uct_is_nan_at_root():
    import types
    legacy_agent = import_legacy('agent')
    BatchDict = import_legacy('utilities').BatchDict
    agent = types.SimpleNamespace(N=BatchDict(torch.device('cpu')))
    assert torch.isnan(legacy_agent.Agent.calculate_UCT(agent, torch.zeros(1, 2, 28, 28))).all()


def test_debate_end_to_end(judge):
    """A full debate reveals exactly ``pixels`` distinct nonzero pixels per game."""
    from argparse import Namespace
    from torch.utils.data import DataLoader
    from safe_debates.agent import Agent
    images = random_images(6, 25, seed=2)
    loader = DataLoader(TensorDataset(images, torch.arange(6)), batch_size=3)
    args = Namespace(pixels=4, rollouts=10, liar_first=True)
    debate = Debate((Agent(True, args), Agent(False, args)), loader, args)
    result = debate.play(judge, torch.device('cpu'))
    rev = result['revealed']
    assert rev.shape == (3, 4)
    assert all(len(set(r)) == 4 for r in rev.tolist())
    assert (images[:3].view(3, -1).gather(1, rev) > 0).all()
    assert torch.equal(result['helper']['wins'] + result['liar']['wins'], torch.ones(3))
