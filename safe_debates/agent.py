from .mcts import MCTS


class Agent(object):
    """A debater that picks pixels to reveal by Monte Carlo tree search against the judge."""
    def __init__(self, honest, args):
        self.honest = honest
        self.pixels = args.pixels
        self.rollouts = args.rollouts
        self.commits = None
        self.opponent_commits = None
        self.mcts = MCTS(args.rollouts, c_puct=getattr(args, 'c_puct', 1.),
                         progress=getattr(args, 'progress', False))

    def simulate_(self, judge, revealed, images, labels):
        """
        Run ``rollouts`` rollouts from the current position of a batch of debates.

        revealed: (B, d) long pixel indices revealed so far.  Returns (B,) pixel indices.
        A fresh tree is built for every move, as in the original implementation.
        """
        if self.commits is not None:
            raise NotImplementedError('precommit debates are not implemented')
        moves, _ = self.mcts.search(judge, images, labels, revealed, self.pixels, self.honest)
        return moves

    def precommit_(self, commits, opponent_commits):
        self.commits = commits
        self.opponent_commits = opponent_commits

    def reset_(self):
        self.mcts.tree = None
