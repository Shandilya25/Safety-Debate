"""
Batched Monte Carlo tree search for pixel-reveal debates.

A search runs for B independent games at once.  Actions are indexed per game in a compact
*candidate* space: candidate k of game b is the k-th nonzero pixel of image b (ascending
pixel order), so K = max nonzero pixels in the batch instead of 784.

Nodes are identified by (game, set of pixels revealed since the search root).  Keying by
the revealed *set* reproduces the original implementation's keying by state contents:
transpositions share statistics, but games in a batch never do.

A Python dict maps node keys (tuples of small ints) to integer node ids.  Most deep nodes
are visited exactly once, so a node gets a dense (K,) row of visit counts / values only on
its second visit; until then its single visited edge is stored inline.  This is exactly
equivalent to dense per-node storage (tests/test_mcts.py).

The tree always lives on the CPU; only the judge forward pass runs on the judge's device,
costing one host<->device round trip per rollout.
"""
import torch

from .data import IMAGE_SIZE


def puct_scores(N, Q, visits, valid, c_puct):
    """AlphaZero-style PUCT with a uniform prior over valid actions; invalid -> -inf."""
    prior = 1. / valid.sum(-1, keepdim=True).type(torch.float)
    U = c_puct * prior * visits.sqrt().unsqueeze(1) / (1 + N)
    return (Q + U).masked_fill(~valid, float('-inf'))


def argmax_random_ties(scores):
    """Row-wise argmax, breaking ties uniformly at random."""
    ties = scores == scores.max(-1, keepdim=True)[0]
    return torch.rand(scores.shape).masked_fill_(~ties, -1.).argmax(-1)


def argmax_first(scores):
    return scores.argmax(-1)


TIE_BREAKS = {'random': argmax_random_ties, 'first': argmax_first}


def _grow(tensor, size, fill=0):
    """Return ``tensor`` with at least ``size`` rows, doubling capacity when needed."""
    if size <= tensor.size(0):
        return tensor
    new = tensor.new_full((max(size, 2 * tensor.size(0)),) + tensor.shape[1:], fill)
    new[:tensor.size(0)] = tensor
    return new


class SearchTree(object):
    """
    Visit counts N and mean values Q for (node, action) edges.

    Per-node metadata (total visits, dense row id, the single visited edge) is kept in
    Python lists: searches touch only B nodes per level, and for the batch sizes used here
    list access is cheaper than many tiny tensor ops (see README, "Performance").  Dense
    (K,) rows of N and Q live in preallocated tensors that double in size when full.
    """
    def __init__(self, num_actions, row_capacity=256):
        self.num_actions = num_actions
        self._ids = {}
        self.visits = []
        self.row = []
        self.first_action = []
        self.first_value = []
        self.num_rows = 0
        self.N = torch.zeros(row_capacity, num_actions)
        self.Q = torch.zeros(row_capacity, num_actions)

    @property
    def num_nodes(self):
        return len(self._ids)

    def lookup(self, keys):
        """Node ids (list) for hashable ``keys``, creating unseen nodes."""
        ids = self._ids
        out = []
        for key in keys:
            node = ids.get(key)
            if node is None:
                node = ids[key] = len(ids)
                self.visits.append(0)
                self.row.append(-1)
                self.first_action.append(0)
                self.first_value.append(0.)
            out.append(node)
        return out

    def stats(self, nodes):
        """Dense (len(nodes), K) N and Q tensors, plus total visits per node."""
        visits = [self.visits[n] for n in nodes]
        N = torch.zeros(len(nodes), self.num_actions)
        Q = torch.zeros_like(N)
        dense, rows, single, actions, values = [], [], [], [], []
        for i, n in enumerate(nodes):
            if self.row[n] >= 0:
                dense.append(i)
                rows.append(self.row[n])
            elif visits[i] == 1:
                single.append(i)
                actions.append(self.first_action[n])
                values.append(self.first_value[n])
        if dense:
            N[dense] = self.N[rows]
            Q[dense] = self.Q[rows]
        if single:
            N[single, actions] = 1.
            Q[single, actions] = torch.tensor(values)
        return N, Q, torch.tensor(visits, dtype=torch.float)

    def update(self, nodes, actions, values):
        """Back up ``values`` along edges (nodes[i], actions[i]); nodes must be distinct."""
        rows, acts, vals, promoted = [], [], [], []
        for n, a, v in zip(nodes, actions, values):
            visits = self.visits[n]
            self.visits[n] = visits + 1
            if visits == 0:
                self.first_action[n] = a
                self.first_value[n] = v
                continue
            if self.row[n] < 0:
                self.row[n] = self.num_rows
                self.num_rows += 1
                promoted.append(n)
            rows.append(self.row[n])
            acts.append(a)
            vals.append(v)

        if promoted:
            self.N = _grow(self.N, self.num_rows)
            self.Q = _grow(self.Q, self.num_rows)
            new_rows = [self.row[n] for n in promoted]
            first = [self.first_action[n] for n in promoted]
            self.N[new_rows] = 0.
            self.Q[new_rows] = 0.
            self.N[new_rows, first] = 1.
            self.Q[new_rows, first] = torch.tensor([self.first_value[n] for n in promoted])
        if rows:
            v = torch.tensor(vals)
            n, q = self.N[rows, acts], self.Q[rows, acts]
            self.Q[rows, acts] = (n * q + v) / (n + 1)
            self.N[rows, acts] = n + 1

    def nbytes(self):
        """Approximate memory held by the tree (tensor rows plus list metadata)."""
        rows = (self.N.numel() + self.Q.numel()) * 4
        return rows + 8 * 4 * len(self.visits) + 200 * len(self._ids)


class MCTS(object):
    """
    Search for the next pixel to reveal in B debates simultaneously.

    Each rollout selects actions with PUCT down to the end of the debate (all ``pixels``
    revealed), evaluates the judge once on the batch of terminal states and backs up the
    searching agent's win (honest: judge correct; liar: judge wrong).  Values are negated
    at the opponent's levels (negamax).  The move played is the most visited root action.
    """
    def __init__(self, rollouts, c_puct=1., tie_break='random', progress=False):
        self.rollouts = rollouts
        self.c_puct = c_puct
        self.tie_break = TIE_BREAKS[tie_break]
        self.progress = progress
        self.tree = None

    @torch.no_grad()
    def search(self, judge, images, labels, revealed, total_moves, honest):
        """
        images: (B, 1, 28, 28) and labels: (B,) on the judge's device.
        revealed: (B, d) long pixel indices already revealed (CPU).
        Returns the (B,) pixel indices to reveal next and (B, 784) root visit counts.
        """
        device = images.device
        B = images.size(0)
        remaining = total_moves - revealed.size(1)
        flat = images.view(B, -1)

        # candidate actions: nonzero pixels of each image, ascending, padded to K
        nonzero = (flat > 0).cpu()
        counts = nonzero.sum(1)
        K = int(counts.max())
        order = torch.arange(flat.size(1)) + (~nonzero).long() * flat.size(1)
        cand = order.argsort(1)[:, :K]
        cand_ok = torch.arange(K) < counts.unsqueeze(1)
        root_revealed = torch.zeros_like(nonzero).scatter_(1, revealed, True).gather(1, cand)
        root_valid = cand_ok & ~root_revealed
        if (root_valid.sum(1) < remaining).any():
            raise ValueError('an image has fewer unrevealed nonzero pixels than moves left')

        root_mask = torch.zeros_like(flat).scatter_(1, revealed.to(device), 1.)
        games = torch.arange(B)
        signs = [1 if level % 2 == 0 else -1 for level in range(remaining)]
        self.tree = tree = SearchTree(K)
        roots = tree.lookup([(b,) for b in range(B)])

        rollouts = range(self.rollouts)
        if self.progress:
            import tqdm
            rollouts = tqdm.tqdm(rollouts, leave=False)
        for _ in rollouts:
            node, taken = roots, root_revealed.clone()
            path_nodes, path_actions = [], []
            for level in range(remaining):
                N, Q, visits = tree.stats(node)
                scores = puct_scores(N, Q, visits, cand_ok & ~taken, self.c_puct)
                action = self.tie_break(scores)
                taken[games, action] = True
                action = action.tolist()
                path_nodes.append(node)
                path_actions.append(action)
                if level + 1 < remaining:
                    node = tree.lookup([(b, *sorted(p))
                                        for b, p in enumerate(zip(*path_actions))])

            pixels = cand.gather(1, torch.tensor(path_actions).t()).to(device)
            mask = root_mask.scatter(1, pixels, 1.)
            states = torch.stack((mask, flat * mask), 1).view(-1, 2, IMAGE_SIZE, IMAGE_SIZE)
            preds = judge(states).argmax(-1)
            wins = (preds == labels) if honest else (preds != labels)
            wins = wins.type(torch.float).tolist()
            for level in range(remaining):
                values = wins if signs[level] > 0 else [-w for w in wins]
                tree.update(path_nodes[level], path_actions[level], values)

        N = tree.stats(roots)[0].masked_fill_(~root_valid, -1.)
        moves = cand.gather(1, N.argmax(-1, keepdim=True)).squeeze(1)
        root_visits = torch.zeros(B, flat.size(1)).scatter_(1, cand, N.clamp(min=0))
        return moves, root_visits

