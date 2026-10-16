import pytest
import torch

from conftest import random_images
from reference_mcts import ReferenceAgent
from safe_debates.data import sparse_state
from safe_debates.mcts import MCTS, SearchTree, argmax_random_ties


@pytest.mark.parametrize('honest', [True, False])
@pytest.mark.parametrize('nonzero,pixels,already', [(8, 4, 0), (12, 3, 1), (30, 4, 2), (6, 5, 0)])
def test_matches_reference(judge, honest, nonzero, pixels, already):
    """Optimized search reproduces the dict-based reference: same moves and root visits."""
    B = 3
    images = random_images(B, nonzero, seed=nonzero)
    labels = torch.tensor([1, 7, 3])
    # reveal ``already`` pixels first so searches start mid-debate
    revealed = torch.stack([(images[b].view(-1) > 0).nonzero().squeeze(1)[:already]
                            for b in range(B)])

    ref = ReferenceAgent(honest, pixels, rollouts=150, tie_break='first')
    ref_moves, ref_visits = ref.simulate_(judge, sparse_state(images, revealed), images, labels)

    mcts = MCTS(rollouts=150, tie_break='first')
    moves, visits = mcts.search(judge, images, labels, revealed, pixels, honest)

    assert torch.equal(moves, ref_moves)
    assert torch.equal(visits, ref_visits)
    assert visits.sum(1).eq(150).all()


def test_tree_matches_dense_storage():
    """Lazy row promotion is equivalent to storing a dense row for every node."""
    g = torch.Generator().manual_seed(0)
    K, nodes = 5, 7
    tree = SearchTree(K, row_capacity=1)
    ids = tree.lookup(list(range(nodes)))
    assert ids == list(range(nodes))
    dense_N, dense_Q = torch.zeros(nodes, K), torch.zeros(nodes, K)
    for _ in range(60):
        batch = torch.randperm(nodes, generator=g)[:3].tolist()
        actions = torch.randint(K, (3,), generator=g).tolist()
        values = torch.randn(3, generator=g).tolist()
        tree.update(batch, actions, values)
        for n, a, v in zip(batch, actions, values):
            v = torch.tensor(v)
            dense_Q[n, a] = (dense_N[n, a] * dense_Q[n, a] + v) / (dense_N[n, a] + 1)
            dense_N[n, a] += 1
        N, Q, visits = tree.stats(ids)
        assert torch.equal(N, dense_N) and torch.equal(Q, dense_Q)
        assert torch.equal(visits, dense_N.sum(1))


def test_transpositions_share_nodes():
    tree = SearchTree(4)
    a = tree.lookup([(0, 1, 3)])
    b = tree.lookup([(0, 1, 3), (1, 1, 3)])
    assert b[0] == a[0] and b[1] != a[0]


def test_moves_are_valid_and_unrevealed(judge):
    images = random_images(4, 10, seed=3)
    labels = torch.arange(4)
    revealed = torch.zeros(4, 0, dtype=torch.long)
    mcts = MCTS(rollouts=20)
    for _ in range(4):
        move, _ = mcts.search(judge, images, labels, revealed, 4, honest=True)
        assert (images.view(4, -1).gather(1, move.unsqueeze(1)) > 0).all()
        assert not (revealed == move.unsqueeze(1)).any()
        revealed = torch.cat((revealed, move.unsqueeze(1)), 1)


def test_search_is_seed_reproducible(judge):
    images, labels = random_images(2, 15, seed=1), torch.tensor([0, 1])
    revealed = torch.zeros(2, 0, dtype=torch.long)
    results = []
    for _ in range(2):
        torch.manual_seed(123)
        results.append(MCTS(rollouts=50).search(judge, images, labels, revealed, 4, False))
    assert torch.equal(results[0][0], results[1][0])
    assert torch.equal(results[0][1], results[1][1])


def test_random_ties_are_uniform():
    torch.manual_seed(0)
    scores = torch.tensor([[1., 3., 3., float('-inf'), 3.]]).expand(6000, -1)
    counts = torch.bincount(argmax_random_ties(scores), minlength=5).float() / 6000
    assert counts[0] == 0 and counts[3] == 0
    assert torch.allclose(counts[[1, 2, 4]], torch.full((3,), 1 / 3), atol=0.03)


def test_too_few_pixels_raises(judge):
    images = random_images(1, 2)
    with pytest.raises(ValueError):
        MCTS(rollouts=1).search(judge, images, torch.tensor([0]),
                                torch.zeros(1, 0, dtype=torch.long), 3, True)
