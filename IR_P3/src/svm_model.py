"""Method B: train a linear SVM and reuse its 22 learned real valued weights"""
from dataclasses import dataclass
import math
import warnings
from data import rank, tokenize
from features import FEATURE_NAMES

def candidate_mask(features, mode):
    if mode not in {"all", "matching"}:
        raise ValueError("Invalid SVM candidate policy")
    return None if mode == "all" else [row[1] > 0 for row in features]

def svm_scores(rows, model):
    return [math.fsum(x * w for x, w in zip(row, model["weights"])) + model["intercept"]
            for row in rows]

def fit_scaler(rows):
    if not rows or any(len(r) != len(FEATURE_NAMES) for r in rows):
        raise ValueError("Expected nonempty training rows with 22 features")
    means, scales = [], []
    for column in zip(*rows):
        mean = math.fsum(column) / len(column)
        variance = math.fsum((x - mean) ** 2 for x in column) / len(column)
        means.append(mean)
        scales.append(math.sqrt(variance) if variance > 0 else 1.0)
    return means, scales

def scale_rows(rows, means, scales):
    return [[(x - mean) / scale for x, mean, scale in zip(row, means, scales)] for row in rows]

def fit_linear_svm(scaled_rows, labels, means, scales, c, seed):
    """gitting a linear SVM and convert its coefficients to original feature units"""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.svm import LinearSVC

    if set(labels) != {0, 1}:
        raise ValueError("SVM training requires relevant and nonrelevant pairs")
    solver = LinearSVC(C=c, class_weight="balanced", dual=False,
                       max_iter=20000, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        solver.fit(scaled_rows, labels)
    standardized = [float(w) for w in solver.coef_[0]]
    weights = [w / s for w, s in zip(standardized, scales)]
    intercept = float(solver.intercept_[0]) - math.fsum(w * m for w, m in zip(weights, means))
    return {"weights": weights, "intercept": intercept,
            "standardized_weights": standardized,
            "standardized_intercept": float(solver.intercept_[0]),
            "training_means": means, "training_scales": scales}

@dataclass
class SVMRanker:
    """Method B: original feature unit linear decision score w*x + intercept"""

    weights: list[float]
    intercept: float
    standardized_weights: list[float]
    standardized_intercept: float
    training_means: list[float]
    training_scales: list[float]
    cutoff: int = 10
    candidates: str = "all"
    c: float = 0.01

    def __post_init__(self):
        vectors = (self.weights, self.standardized_weights, self.training_means, self.training_scales)
        if (any(len(v) != len(FEATURE_NAMES) or not all(math.isfinite(x) for x in v) for v in vectors)
                or not all(s > 0 for s in self.training_scales)
                or not math.isfinite(self.intercept) or not math.isfinite(self.standardized_intercept)
                or not math.isfinite(self.c) or self.c <= 0 or self.cutoff < 1):
            raise ValueError("Invalid 22-feature linear SVM parameters")
        candidate_mask([], self.candidates)

    def score_features(self, rows):
        if any(len(row) != len(FEATURE_NAMES) for row in rows):
            raise ValueError("Expected all 22 features in the assignment's order")
        return svm_scores(rows, {"weights": self.weights, "intercept": self.intercept})

    def score(self, collection, query):
        return self.score_features(collection.features(query))

    def rank(self, collection, query, top_k=None):
        k = self.cutoff if top_k is None else top_k
        if k < 1:
            raise ValueError("Cutoff must be positive")
        if not tokenize(query):
            return []
        rows = collection.features(query)
        return rank(collection.ids, self.score_features(rows), k, candidate_mask(rows, self.candidates))