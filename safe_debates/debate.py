import torch

from .data import sparse_state


class Debate(object):
    """Alternating pixel-reveal debate between an honest agent and a liar, judged at the end."""
    def __init__(self, agents, data_loader, args):
        self.data_loader = data_loader
        self.pixels = args.pixels
        for agent in agents:
            if agent.honest:
                self.helper = agent
            else:
                self.liar = agent
        self.first = self.liar if args.liar_first else self.helper
        self.second = self.helper if args.liar_first else self.liar
        self.reset_()
        # TODO: precommit (agents commit to a label before the debate; see README)

    @torch.no_grad()
    def play(self, judge, device):
        """Play one batch of debates; returns each agent's evaluation and the labels."""
        images, labels = next(self.iterator)
        images, labels = images.to(device), labels.to(device)
        revealed = torch.zeros(images.size(0), 0, dtype=torch.long)

        for i in range(self.pixels):
            mover = self.first if i % 2 == 0 else self.second
            action = mover.simulate_(judge, revealed, images, labels)
            mover.reset_()
            revealed = torch.cat((revealed, action.unsqueeze(1)), dim=1)

        final_states = sparse_state(images, revealed.to(device))
        return dict(helper=self.evaluate(self.helper, judge, final_states, labels),
                    liar=self.evaluate(self.liar, judge, final_states, labels),
                    labels=labels,
                    revealed=revealed)

    @classmethod
    def evaluate(cls, agent, judge, final_states, labels):
        """Judge predictions and per-game wins for ``agent`` (no-precommit rules)."""
        if agent.commits is not None:
            raise NotImplementedError('precommit debates are not implemented')
        preds = judge(final_states).argmax(-1)
        wins = (preds == labels) if agent.honest else (preds != labels)
        return dict(preds=preds, wins=wins.type(torch.float))

    @classmethod
    def apply_action(cls, action, sparse, images):
        """Return a copy of ``sparse`` (B, 2, H, W) with flat pixel ``action[b]`` revealed."""
        width = images.size(-1)
        games = torch.arange(sparse.size(0), device=sparse.device)
        row, col = action // width, action % width
        sparse = sparse.clone()
        sparse[games, 0, row, col] = 1
        sparse[games, 1, row, col] = images[games, 0, row, col]
        return sparse

    def reset_(self):
        self.iterator = iter(self.data_loader)
        self.first.reset_()
        self.second.reset_()
