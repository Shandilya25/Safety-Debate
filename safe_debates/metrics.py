import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0


def track_stats_(total_meter, class_meters, preds, wins, labels, precommit):
    """Accumulate honest-agent win rate overall and per true digit."""
    if precommit:
        raise NotImplementedError('Statistics for precommit debates are not implemented')
    wins, labels = wins.float().cpu(), labels.cpu()
    total_meter.update(wins.mean().item(), n=wins.numel())
    for digit in labels.unique().tolist():
        selected = wins[labels == digit]
        class_meters[digit].update(selected.mean().item(), n=selected.numel())
