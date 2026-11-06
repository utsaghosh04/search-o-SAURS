"""Read Cranfield data and prepare the searchable collection."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "cran"
FIELDS = ("title", "author", "bibliography", "body")


@dataclass(frozen=True)
class Document:
    id: int
    title: str = ""
    author: str = ""
    bibliography: str = ""
    body: str = ""


@dataclass(frozen=True)
class Query:
    id: int
    source_id: int
    text: str


def tokenize(text: str) -> list[str]:
    """Lowercase text and extract alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def read_records(path: str | Path) -> list[tuple[int, dict[str, str]]]:
    records, seen, fields = [], set(), {}
    record_id, field = None, None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        marker = re.fullmatch(r"\.([A-Z])(?:\s+(.*))?", line.strip())
        if marker:
            tag, value = marker.groups()
            if tag == "I":
                if record_id is not None:
                    records.append((record_id, {k: " ".join(v) for k, v in fields.items()}))
                record_id = int(value or "")
                if record_id <= 0 or record_id in seen:
                    raise ValueError(f"Nonpositive or duplicate record ID {record_id} in {path}")
                seen.add(record_id)
                fields, field = {}, None
            else:
                if record_id is None:
                    raise ValueError(f"Field before first .I in {path}")
                field = tag
                fields.setdefault(field, [])
                if value:
                    fields[field].append(value)
        elif line.strip():
            if record_id is None or field is None:
                raise ValueError(f"Text outside a field in {path}")
            fields[field].append(line.strip())
    if record_id is not None:
        records.append((record_id, {k: " ".join(v) for k, v in fields.items()}))
    if not records:
        raise ValueError(f"No records in {path}")
    return records


def read_documents(path: str | Path) -> list[Document]:
    return [Document(i, *(f.get(tag, "") for tag in ("T", "A", "B", "W")))
            for i, f in read_records(path)]


def read_queries(path: str | Path, id_mode: str = "ordinal") -> list[Query]:
    if id_mode not in {"ordinal", "source"}:
        raise ValueError("Query ID mode must be ordinal or source")
    return [Query(n if id_mode == "ordinal" else i, i, f.get("W", ""))
            for n, (i, f) in enumerate(read_records(path), 1)]


def read_qrels(path: str | Path) -> dict[int, dict[int, int]]:
    result = {}
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"Expected query document grade at {path}:{n}")
        qid, did, grade = map(int, parts)
        if qid <= 0 or did <= 0 or grade not in {-1, 0, 1, 2, 3, 4, 5}:
            raise ValueError(f"Invalid Cranfield judgment at {path}:{n}")
        if did in result.setdefault(qid, {}):
            raise ValueError(f"Duplicate judgment for query {qid}, document {did}")
        result[qid][did] = grade
    if not result:
        raise ValueError(f"No judgments in {path}")
    return result


class CranfieldCorpus:
    def __init__(self, documents: list[Document], zone_match="any", stream="all"):
        if not documents or len({d.id for d in documents}) != len(documents):
            raise ValueError("Collection must have documents with unique IDs")
        if zone_match not in {"any", "all"} or stream not in {"all", "body"}:
            raise ValueError("Invalid Boolean matching or feature stream setting")
        self.zone_match, self.stream = zone_match, stream
        self.documents = sorted(documents, key=lambda d: d.id)
        self.ids = [d.id for d in self.documents]
        self.zone_terms = [[set(tokenize(getattr(d, f))) for f in FIELDS]
                           for d in self.documents]
        self.streams = [Counter(tokenize(d.body if stream == "body" else
                                        " ".join(getattr(d, f) for f in FIELDS)))
                        for d in self.documents]
        self.lengths = [sum(s.values()) for s in self.streams]
        self.average_length = sum(self.lengths) / len(self.ids) or 1.0
        self.df = Counter()
        for s in self.streams:
            self.df.update(s.keys())
        self.idf = {t: math.log((len(self.ids) + 1) / (df + 1)) + 1
                    for t, df in self.df.items()}
        self.norms = [math.sqrt(math.fsum((tf * self.idf[t]) ** 2 for t, tf in s.items()))
                      for s in self.streams]

    def zone_features(self, query: str) -> list[list[int]]:
        terms = set(tokenize(query))
        if self.zone_match == "all":
            return [[int(bool(terms) and terms <= zone) for zone in zones]
                    for zones in self.zone_terms]
        return [[int(bool(terms & zone)) for zone in zones] for zones in self.zone_terms]

    def features(self, query):
        """Compute the 22 query-document features."""
        from features import FeatureExtractor
        return FeatureExtractor(self).features(query)




def rank(ids, scores, k, candidates=None) -> list[tuple[int, float]]:
    if k < 1:
        raise ValueError("Cutoff must be positive")
    if len(scores) != len(ids) or not all(math.isfinite(s) for s in scores):
        raise ValueError("Expected one finite score per document")
    if candidates is not None and len(candidates) != len(ids):
        raise ValueError("Expected one candidate flag per document")
    positions = [i for i in range(len(ids)) if candidates is None or candidates[i]]
    positions.sort(key=lambda i: (-scores[i], ids[i]))
    return [(ids[i], float(scores[i])) for i in positions[:k]]


def write_run(path: Path, results: dict, tag: str):
    """TREC text: query_id Q0 document_id rank score run_tag; no header."""
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for qid, rows in sorted(results.items()):
            for n, (did, score) in enumerate(rows, 1):
                stream.write(f"{qid} Q0 {did} {n} {score:.17g} {tag}\n")

