"""Fast inference for a model trained offline with scikit-learn.

Runtime uses Python's stdlib; no sklearn import/installation is required for play.
The frozen model predicts equity against a uniformly random PLO5 opponent.
It is not an opponent-adjusted equity or a guarantee that a raise is profitable.
"""
import json
import os
from collections import Counter
from itertools import combinations

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'preflop_equity_model.json')


def features(hole):
    ranks = [c >> 2 for c in hole]
    suits = [c & 3 for c in hole]
    rc = Counter(ranks)
    sc = Counter(suits)
    high = sorted(ranks, reverse=True)
    unique = sorted(rc)
    sorted_suits = sorted(sc.values(), reverse=True)
    pairs = sorted((r for r, n in rc.items() if n >= 2), reverse=True)
    connected = [sum(1 for r in unique if low <= r <= low + 4) for low in range(5)]
    # Ace-6-7-8-9 wheel: ace is ranked 8 in this deck.
    connected.append(sum(r in unique for r in (8, 0, 1, 2, 3)))
    suit_highs = sorted((max(c >> 2 for c in hole if (c & 3) == s)
                         for s, n in sc.items() if n >= 2), reverse=True)
    cross_suit_high_pairs = sum(
        ((a >> 2) >= 5 or (b >> 2) >= 5)
        for a, b in combinations(hole, 2) if (a & 3) == (b & 3)
    )
    paired_rank_suit_diversity = sum(
        len({c & 3 for c in hole if (c >> 2) == r}) for r in pairs
    )
    return [
        *high,
        len(unique), max(rc.values()), len(pairs),
        pairs[0] if pairs else -1, pairs[1] if len(pairs) > 1 else -1,
        *([*sorted_suits, 0, 0, 0, 0][:4]),
        sum(v >= 2 for v in sc.values()),
        sum(v * (v - 1) // 2 for v in sc.values()),
        suit_highs[0] if suit_highs else -1,
        suit_highs[1] if len(suit_highs) > 1 else -1,
        cross_suit_high_pairs, paired_rank_suit_diversity,
        max(connected), sorted(connected, reverse=True)[1],
        sum(v >= 4 for v in connected),
        sum(v >= 3 for v in connected),
        sum(r >= 6 for r in ranks), sum(r >= 7 for r in ranks),
        sum(r == 8 for r in ranks),
        max(high) - min(high),
        *[rc.get(r, 0) for r in range(9)],
    ]


class EquityModel:
    def __init__(self):
        with open(MODEL_PATH, encoding='utf-8') as f:
            model = json.load(f)
        self.base = model['base']
        self.trees = model['trees']
        self.learning_rate = model['learning_rate']

    def predict(self, hole):
        v = features(hole)
        score = self.base
        for tree in self.trees:
            idx = 0
            while tree[idx][0] != -1:
                feature, threshold, left, right, value = tree[idx]
                idx = left if v[feature] <= threshold else right
            score += tree[idx][4]  # sklearn leaves are already shrinkage-adjusted
        return max(0.01, min(0.99, score))


MODEL = None

def warmup_model():
    global MODEL
    if MODEL is None:
        MODEL = EquityModel()


def preflop_equity(hole):
    if MODEL is None:
        warmup_model()
    return MODEL.predict(hole)