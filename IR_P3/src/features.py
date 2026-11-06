"""Query Document features, computed using the standard libraries"""
from collections import Counter
import math
from data import tokenize

FEATURE_NAMES = (
    "query_term_number", "query_term_ratio", "stream_length", "idf",
    "tf_sum", "tf_min", "tf_max", "tf_mean", "tf_variance",
    "normalized_tf_sum", "normalized_tf_min", "normalized_tf_max",
    "normalized_tf_mean", "normalized_tf_variance",
    "tf_idf_sum", "tf_idf_min", "tf_idf_max", "tf_idf_mean", "tf_idf_variance",
    "boolean_model", "vector_space_model", "bm25",
)

def statistics(values) -> list[float]:
    if not values:
        return [0.0] * 5
    total = math.fsum(values)
    mean = total / len(values)
    variance = math.fsum((v - mean) ** 2 for v in values) / len(values)
    return [total, min(values), max(values), mean, variance]

class FeatureExtractor:
    def __init__(self, corpus):
        self.corpus = corpus

    def features(self, query: str) -> list[list[float]]:
        counts = Counter(tokenize(query))
        query_length = sum(counts.values())
        terms = sorted(counts)
        if not terms:
            return [[0.0, 0.0, float(length)] + [0.0] * 19 for length in self.corpus.lengths]
        unseen_idf = math.log(len(self.corpus.ids) + 1) + 1
        idfs = [self.corpus.idf.get(term, unseen_idf) for term in terms]
        query_vector = [counts[term] * idf for term, idf in zip(terms, idfs)]
        query_norm = math.sqrt(math.fsum(weight * weight for weight in query_vector))
        bm25_idfs = [math.log1p((len(self.corpus.ids) - self.corpus.df[term] + 0.5)
                               / (self.corpus.df[term] + 0.5)) for term in terms]
        rows = []
        for document_counts, length, doc_norm in zip(self.corpus.streams, self.corpus.lengths, self.corpus.norms):
            tf = [float(document_counts[term]) for term in terms]
            matched = sum(frequency > 0 for frequency in tf)
            normalized = [frequency / length if length else 0.0 for frequency in tf]
            tf_idf = [frequency * idf for frequency, idf in zip(tf, idfs)]
            dot_product = math.fsum(document_weight * query_weight
                                    for document_weight, query_weight in zip(tf_idf, query_vector))
            cosine = dot_product / (doc_norm * query_norm) if doc_norm and query_norm else 0.0
            norm = 1.2 * (0.25 + 0.75 * length / self.corpus.average_length)
            bm25 = math.fsum(idf * frequency * 2.2 / (frequency + norm)
                             for frequency, idf in zip(tf, bm25_idfs))
            rows.append([float(query_length), matched / len(terms), float(length),
                         math.fsum(idf for frequency, idf in zip(tf, idfs) if frequency > 0)]
                        + statistics(tf) + statistics(normalized) + statistics(tf_idf)
                        + [float(matched == len(terms)), cosine, bm25])
        return rows