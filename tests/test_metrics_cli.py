import pytest
import torch

import main
import train_judge
from safe_debates.metrics import AverageMeter, track_stats_


def test_track_stats_per_class_accuracy():
    total, classes = AverageMeter(), [AverageMeter() for _ in range(10)]
    labels = torch.tensor([0, 0, 1, 1, 1, 2])
    wins = torch.tensor([1., 0., 1., 1., 0., 1.])
    track_stats_(total, classes, None, wins, labels, precommit=False)
    assert total.avg == pytest.approx(4 / 6) and total.count == 6
    assert classes[0].avg == pytest.approx(0.5)
    assert classes[1].avg == pytest.approx(2 / 3)
    assert classes[2].avg == 1. and classes[3].count == 0


def test_main_requires_no_precommit():
    with pytest.raises(SystemExit):
        main.parse_args(['--pixels', '4', '--checkpoint', 'x'])
    args = main.parse_args(['--pixels', '4', '--checkpoint', 'x', '--no-precommit'])
    assert args.liar_first and not args.precommit


def test_train_judge_defaults():
    args = train_judge.parse_args(['--pixels', '6'])
    assert (args.batch_size, args.batches, args.lr, args.workers) == (128, 30000, 1e-4, 1)
