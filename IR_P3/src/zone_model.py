"""Method A: Boolean weighted zones and exhaustive simplex grid learning."""
from dataclasses import dataclass
from itertools import product
import math
from data import rank, tokenize
from metrics import evaluate

def zone_scores(zones, weights):
    return [round(math.fsum(match * weight for match, weight in zip(zone, weights)), 12)
            for zone in zones]

def weight_grid(divisions: int):
    if divisions < 1:
        raise ValueError("Grid divisions must be positive")
    for a in range(divisions + 1):
        for b in range(divisions - a + 1):
            for c in range(divisions - a - b + 1):
                yield [a / divisions, b / divisions, c / divisions,
                       (divisions - a - b - c) / divisions]

def learn_zone_weights(collection, zones, qrels, train_ids, divisions, k):
    boolean_patterns = list(product((0, 1), repeat=4))
    query_patterns = {qid: [tuple(zone) for zone in zones[qid]] for qid in train_ids}
    best_score, best_weights = -1.0, None
    cached_training_scores = {}

    for weights in weight_grid(divisions):
        scores = zone_scores(boolean_patterns, weights)
        pattern_scores = dict(zip(boolean_patterns, scores))
        score_groups = {score: group for group, score in enumerate(sorted(set(scores)))}
        # Reuse a result only when ordering, ties and positive-score matches agree.
        ranking_signature = tuple((score_groups[score], score > 0) for score in scores)

        if ranking_signature not in cached_training_scores:
            rankings = {}
            for qid in train_ids:
                query_scores = [pattern_scores[pattern] for pattern in query_patterns[qid]]
                candidates = [score > 0 for score in query_scores]
                rankings[qid] = rank(collection.ids, query_scores, k, candidates)
            cached_training_scores[ranking_signature] = evaluate(rankings, qrels)["macro"]["f1"]

        training_score = cached_training_scores[ranking_signature]
        if training_score > best_score:
            best_score, best_weights = training_score, weights[:]
    return best_weights, best_score

@dataclass
class ZoneRanker:
    """Method A: score(d,q) = sum of four learned Boolean zone weights."""

    weights: list[float]
    cutoff: int = 10

    def __post_init__(self):
        if (len(self.weights) != 4 or not all(math.isfinite(w) and w >= 0 for w in self.weights)
                or not math.isclose(sum(self.weights), 1.0) or self.cutoff < 1):
            raise ValueError("Expected four nonnegative unit-sum weights and a positive cutoff")

    def score(self, collection, query):
        return zone_scores(collection.zone_features(query), self.weights)

    def rank(self, collection, query, top_k=None):
        k = self.cutoff if top_k is None else top_k
        if k < 1:
            raise ValueError("Cutoff must be positive")
        if not tokenize(query):
            return []
        scores = self.score(collection, query)
        return rank(collection.ids, scores, k, [s > 0 for s in scores])