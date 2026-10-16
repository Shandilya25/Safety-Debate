"""
Reference MCTS written in the style of ``legacy/agent.py``: recursive rollouts over full
(B, 2, 28, 28) states and per-state dense statistics in a dict keyed by flattened state
contents.  It fixes only the documented defects of the original (statistics never stored,
in-place mutation of the parent state, transposed/cross-batch reveal, 1568-wide action
space, NaN prior, tie-breaking returning an index, wrong depth and sign parity, roots
shared across games).  It is slow on purpose and serves as the oracle for the optimized
``safe_debates.mcts`` implementation and as the "old data structure" benchmark baseline.
"""
import torch


class StateDict(dict):
    """dict of per-state (784,) tensors keyed by (game, flattened state contents)."""
    def rows(self, states):
        out = []
        for b, state in enumerate(states):
            key = (b,) + tuple(state.contiguous().view(-1).tolist())
            if key not in self:
                self[key] = torch.zeros(state[0].numel())
            out.append(self[key])
        return out


class ReferenceAgent(object):
    def __init__(self, honest, pixels, rollouts, c_puct=1., tie_break='first'):
        self.honest = honest
        self.pixels = pixels
        self.rollouts = rollouts
        self.c_puct = c_puct
        self.tie_break = tie_break
        self.N, self.Q = StateDict(), StateDict()

    def rollout_(self, judge, states, images, labels, level, remaining):
        if level == remaining:
            preds = judge(states).argmax(-1)
            wins = (preds == labels) if self.honest else (preds != labels)
            return wins.type(torch.float)

        valid = self.get_valid_actions(states, images)
        N, Q = torch.stack(self.N.rows(states)), torch.stack(self.Q.rows(states))
        prior = 1. / valid.sum(-1, keepdim=True).type(torch.float)
        U = self.c_puct * prior * N.sum(-1).sqrt().unsqueeze(1) / (1 + N)
        scores = (Q + U).masked_fill(~valid, float('-inf'))
        if self.tie_break == 'first':
            actions = scores.argmax(-1)
        else:
            raise ValueError(self.tie_break)
        successors = self.apply_action(actions, states, images)
        V = self.rollout_(judge, successors, images, labels, level + 1, remaining)
        value = V if level % 2 == 0 else -V

        for b, (Nrow, Qrow) in enumerate(zip(self.N.rows(states), self.Q.rows(states))):
            a = actions[b]
            Qrow[a] = (Nrow[a] * Qrow[a] + value[b]) / (Nrow[a] + 1)
            Nrow[a] += 1
        return V

    @torch.no_grad()
    def simulate_(self, judge, states, images, labels):
        remaining = self.pixels - int(states[0, 0].sum())
        for _ in range(self.rollouts):
            self.rollout_(judge, states, images, labels, 0, remaining)
        valid = self.get_valid_actions(states, images)
        N = torch.stack(self.N.rows(states)).masked_fill(~valid, -1.)
        return N.argmax(-1), N.clamp(min=0)

    @staticmethod
    def get_valid_actions(states, images):
        return ((states[:, 0] == 0) & (images[:, 0] > 0)).view(states.size(0), -1)

    @staticmethod
    def apply_action(action, sparse, images):
        width = images.size(-1)
        games = torch.arange(sparse.size(0))
        row, col = action // width, action % width
        sparse = sparse.clone()
        sparse[games, 0, row, col] = 1
        sparse[games, 1, row, col] = images[games, 0, row, col]
        return sparse

    def reset_(self):
        self.N.clear()
        self.Q.clear()
