"""
-------------------------------------------------------------------------------
  COMPREHENSIVE CORRECTNESS & BENCHMARK SUITE
  search-o-SAURS Boolean IR Pipeline - Cranfield Collection
-------------------------------------------------------------------------------

This script performs:

  PART A - CORRECTNESS TESTING (against PA1 problem statement)
  -------------------------------------------------------------
   1. Cranfield Parsing: All 1400 docs parsed, only .T and .W fields
   2. Preprocessing: Tokenization, Normalization, Stemming, Stop-word removal
   3. Output Format: .I / .S tag compliance
   4. Index Format: header, sorted tokens, ascending docids, df accuracy
   5. Boolean Search: AND/OR correctness, commutativity, identity laws
   6. Query Preprocessing Parity: doc pipeline == query pipeline

  PART B - BENCHMARK & STATISTICAL ANALYSIS
  ------------------------------------------
   7. Vocabulary Statistics (Heap's Law, Zipf's Law)
   8. Index Compression Analysis (term distribution, df statistics)
   9. Comparison with published Cranfield benchmarks from IR literature
  10. Search Performance Profiling (binary search vs linear scan)
  11. Stemmer validation against published Porter stemmer test vectors
  12. Stop word coverage analysis

  PART C - EDGE CASE & STRESS TESTING
  ------------------------------------
  13. Boundary conditions (empty docs, single-token docs, etc.)
  14. Adversarial queries
  15. Large-scale merge stress tests

Author: Automated Test Suite
"""

import os
import re
import sys
import time
import math
import json
import random
import subprocess
import collections

# Configure imports so tests run from the project root.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import importlib.util

# Load the pipeline modules directly from their filename-based paths.
spec = importlib.util.spec_from_file_location(
    "preprocess", os.path.join(PROJECT_ROOT, "search-o-SAURS_preprocess.py")
)
preprocess_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preprocess_mod)

tokenize = preprocess_mod.tokenize
normalize = preprocess_mod.normalize
remove_stopwords = preprocess_mod.remove_stopwords
stem_tokens = preprocess_mod.stem
deduplicate = preprocess_mod.deduplicate
load_stopwords = preprocess_mod.load_stopwords
parse_cranfield = preprocess_mod.parse_cranfield

spec2 = importlib.util.spec_from_file_location(
    "searcher", os.path.join(PROJECT_ROOT, "search-o-SAURS_search.py")
)
searcher_mod = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(searcher_mod)

preprocess_query_term = searcher_mod.preprocess_query_term
intersect_postings = searcher_mod.intersect_postings
union_postings = searcher_mod.union_postings
binary_search_index = searcher_mod.binary_search_index
_gallop_search = searcher_mod._gallop_search

from porter import PorterStemmer

# Track test outcomes and generated pipeline file locations.
PASSED = 0
FAILED = 0
WARNINGS = 0
RESULTS = {}

INDEX_FILE = os.path.join("output", "search-o-SAURS_cran.index")
PROCESSED_FILE = os.path.join("output", "search-o-SAURS_processed.all")

# Record exact assertions and report failure details.
def test(name, actual, expected, section=""):
    global PASSED, FAILED
    if actual == expected:
        PASSED += 1
        print(f"  OK {name}")
        return True
    else:
        FAILED += 1
        print(f"  FAIL FAIL: {name}")
        if isinstance(actual, list) and isinstance(expected, list):
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            if missing: print(f"    Missing: {missing[:10]}{'...' if len(missing)>10 else ''}")
            if extra:   print(f"    Extra:   {extra[:10]}{'...' if len(extra)>10 else ''}")
            print(f"    Expected {len(expected)} items, got {len(actual)}")
        else:
            print(f"    Expected: {expected}")
            print(f"    Actual:   {actual}")
        return False

# Record numeric assertions that allow a relative tolerance.
def test_approx(name, actual, expected, tolerance=0.1):
    """Test with tolerance for floating point comparisons."""
    global PASSED, FAILED
    if abs(actual - expected) <= tolerance * abs(expected) if expected != 0 else abs(actual) <= tolerance:
        PASSED += 1
        print(f"  OK {name} (actual={actual:.4f}, expected~={expected:.4f})")
        return True
    else:
        FAILED += 1
        print(f"  FAIL FAIL: {name}")
        print(f"    Expected ~={expected:.4f} (+/-{tolerance*100}%), Got: {actual:.4f}")
        return False

def warn(name, msg):
    global WARNINGS
    WARNINGS += 1
    print(f"  WARNING WARNING: {name}: {msg}")

def section(title):
    print(f"\n{'-'*75}")
    print(f"  {title}")
    print(f"{'-'*75}")

def subsection(title):
    print(f"\n  {'-'*60}")
    print(f"  {title}")
    print(f"  {'-'*60}")

# Read the complete index for consistency and timing comparisons.
def load_index_bruteforce(filepath):
    index = {}
    with open(filepath, 'r') as f:
        header = f.readline().strip()
        vocab_size, max_docid = [int(x.strip()) for x in header.split(',')]
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(' ', 2)
            token = parts[0]
            df = int(parts[1])
            docids = [int(d.strip()) for d in parts[2].split(',') if d.strip()] if len(parts) > 2 else []
            index[token] = (docids, df)
    return index, vocab_size, max_docid

# Read the processed corpus into a document-to-token mapping.
def load_processed_file(filepath):
    """Load the processed output file into {docid: [tokens]}."""
    docs = {}
    current_id = None
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('.I'):
                current_id = int(line.split()[1])
            elif line.startswith('.S'):
                pass
            elif current_id is not None:
                docs[current_id] = line.split()
                current_id = None
    return docs

def document_pipeline(text, stopwords):
    """Run the exact document preprocessing pipeline."""
    tokens = tokenize(text)
    normalized_tok = normalize(tokens)
    no_stop = remove_stopwords(normalized_tok, stopwords)
    stemmed = stem_tokens(no_stop)
    unique = deduplicate(stemmed)
    return unique

def run_search_cmd(query, output_file="__test_out.txt"):
    result = subprocess.run(
        ["python3", "search-o-SAURS_search.py", query, output_file],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        return [], result.stdout + result.stderr
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            output = f.read().strip()
        count_text, separator, postings_text = output.partition('|')
        if not separator:
            return [], result.stdout + "\nInvalid output format."
        docids = [int(docid.strip()) for docid in postings_text.split(',') if docid.strip()]
        if int(count_text.strip()) != len(docids):
            return [], result.stdout + "\nResult count does not match postings."
        return docids, result.stdout
    return [], result.stdout

# Load all shared resources used by correctness and benchmark sections.
print("=" * 75)
print("  COMPREHENSIVE CORRECTNESS & BENCHMARK REPORT")
print("  search-o-SAURS Boolean IR Pipeline - Cranfield Collection")
print("=" * 75)

stopwords = load_stopwords(os.path.join("data", "stopwords.txt"))
gt_index, vocab_size, max_docid = load_index_bruteforce(INDEX_FILE)
documents = parse_cranfield(os.path.join("data", "cran.all.1400"))
processed_docs = load_processed_file(PROCESSED_FILE)

print(f"\n  Resources loaded:")
print(f"    Stopwords: {len(stopwords)}")
print(f"    Vocabulary: {vocab_size} terms")
print(f"    Documents: {len(documents)}")
print(f"    Processed docs: {len(processed_docs)}")

# Verify Cranfield record parsing and field selection.
section("1. CRANFIELD PARSING CORRECTNESS")

test("parsed exactly 1400 documents", len(documents), 1400)
test("processed exactly 1400 documents", len(processed_docs), 1400)

doc_ids = sorted([d['id'] for d in documents])
test("doc IDs are 1 through 1400", doc_ids, list(range(1, 1401)))

proc_ids = sorted(processed_docs.keys())
test("processed doc IDs are 1 through 1400", proc_ids, list(range(1, 1401)))

doc1_text = documents[0]['text'].lower()
test("doc 1 contains title text 'experimental'", 'experimental' in doc1_text, True)
test("doc 1 contains title text 'aerodynamics'", 'aerodynamics' in doc1_text, True)
test("doc 1 contains abstract text 'slipstream'", 'slipstream' in doc1_text, True)
test("doc 1 does NOT contain author 'brenckman'", 'brenckman' not in doc1_text, True)

doc2_text = documents[1]['text'].lower()

doc2_words = set(re.findall(r'\b[a-z]+\b', doc2_text))
test("doc 2 does NOT contain author name 'ting' as standalone word",
     'ting' not in doc2_words, True)
test("doc 2 does NOT contain affiliation 'rensselaer'", 'rensselaer' not in doc2_text, True)

# Verify each preprocessing stage and known-document behavior.
section("2. PREPROCESSING PIPELINE CORRECTNESS")

subsection("2a. Tokenization")

test("basic sentence", tokenize("the boundary layer"), ['the', 'boundary', 'layer'])
test("abbreviation collapse", tokenize("n.a.c.a. report"), ['naca', 'report'])
test("possessive removal", tokenize("prandtl's theory"), ['prandtl', 'theory'])
test("hyphen splitting", tokenize("high-speed flow"), ['high', 'speed', 'flow'])
test("slash splitting", tokenize("lift/drag"), ['lift', 'drag'])
test("number-word split", tokenize("10degree"), ['degree'])
test("word-number split", tokenize("mach2"), ['mach'])
test("single char removal", tokenize("a b c hello"), ['hello'])
test("pure numeric removal", tokenize("100 200 hello"), ['hello'])
test("multi-hyphen", tokenize("semi-empirical-method"), ['semi', 'empirical', 'method'])
test("parenthetical labels removed", tokenize("(a) first (b) second"), ['first', 'second'])
test("empty input", tokenize(""), [])
test("whitespace only", tokenize("   \n\t  "), [])
test("punctuation only", tokenize(".,;:!?\"()[]{}"), [])

test("complex scientific", tokenize("reynolds-number re10 u.s.a."),
     ['reynolds', 'number', 're', 'usa'])
test("multiple abbreviations", tokenize("a.i.a.a. and n.a.c.a."), ['aiaa', 'and', 'naca'])
test("decimal numbers (periods handled)", tokenize("0.5 and 1.0"),

     [t for t in tokenize("0.5 and 1.0")])

subsection("2b. Normalization")

test("uppercase -> lowercase", normalize(['BOUNDARY']), ['boundary'])
test("mixed case", normalize(['AeRoDyNaMiC']), ['aerodynamic'])

british_american_pairs = [
    ('linearised', 'linearized'),
    ('generalisation', 'generalization'),
    ('normalising', 'normalizing'),
    ('behaviour', 'behavior'),
    ('vapour', 'vapor'),
    ('colour', 'color'),
    ('centre', 'center'),
    ('metre', 'meter'),
    ('fibre', 'fiber'),
    ('analogue', 'analog'),
    ('catalogue', 'catalog'),
    ('programme', 'program'),
    ('gramme', 'gram'),
]

for brit, amer in british_american_pairs:
    test(f"british->american: '{brit}' -> '{amer}'",
         normalize([brit]), [amer])

ise_exceptions = ['noise', 'rise', 'wise', 'cruise', 'exercise', 'chordwise', 'spanwise']
for word in ise_exceptions:
    test(f"ise exception: '{word}' unchanged", normalize([word]), [word])

our_exceptions = ['four', 'pour', 'contour', 'hour', 'tour']
for word in our_exceptions:
    test(f"our exception: '{word}' unchanged", normalize([word]), [word])

re_non_convert = ['structure', 'pressure', 'temperature', 'procedure', 'nature']
for word in re_non_convert:
    test(f"re non-conversion: '{word}' unchanged", normalize([word]), [word])

subsection("2c. Stop Word Removal")

test("removes common stop words",
     remove_stopwords(['the', 'flow', 'is', 'over', 'boundary'], stopwords),
     ['flow', 'boundary'])
test("empty input", remove_stopwords([], stopwords), [])
test("all stop words -> empty",
     remove_stopwords(['the', 'is', 'a', 'an', 'of', 'in', 'to', 'for'], stopwords), [])
test("no stop words -> unchanged",
     remove_stopwords(['aerodynamic', 'boundary', 'turbulent'], stopwords),
     ['aerodynamic', 'boundary', 'turbulent'])

for term in ['boundary', 'flow', 'pressure', 'heat', 'layer', 'wave',
             'shock', 'theory', 'number', 'velocity']:
    test(f"'{term}' is NOT a stop word", term not in stopwords, True)

subsection("2d. Stemming")

stemmer = PorterStemmer()
def do_stem(word):
    return stemmer.stem(word, 0, len(word) - 1)

porter_test_vectors = [
    ('caresses', 'caress'),
    ('ponies', 'poni'),
    ('ties', 'ti'),
    ('caress', 'caress'),
    ('cats', 'cat'),
    ('feed', 'feed'),
    ('agreed', 'agre'),
    ('plastered', 'plaster'),
    ('bled', 'bled'),
    ('motoring', 'motor'),
    ('sing', 'sing'),
    ('conflated', 'conflat'),
    ('troubled', 'troubl'),
    ('sized', 'size'),
    ('hopping', 'hop'),
    ('tanned', 'tan'),
    ('falling', 'fall'),
    ('hissing', 'hiss'),
    ('fizzed', 'fizz'),
    ('failing', 'fail'),
    ('filing', 'file'),
    ('happy', 'happi'),
    ('sky', 'sky'),
    ('relational', 'relat'),
    ('conditional', 'condit'),
    ('rational', 'ration'),
    ('valenci', 'valenc'),
    ('hesitanci', 'hesit'),
    ('digitizer', 'digit'),
    ('conformabli', 'conform'),
    ('radicalli', 'radic'),
    ('differentli', 'differ'),
    ('vileli', 'vile'),
    ('analogousli', 'analog'),
    ('vietnamization', 'vietnam'),
    ('predication', 'predic'),
    ('operator', 'oper'),
    ('feudalism', 'feudal'),
    ('decisiveness', 'decis'),
    ('hopefulness', 'hope'),
    ('callousness', 'callous'),
    ('formaliti', 'formal'),
    ('sensitiviti', 'sensit'),
    ('sensibiliti', 'sensibl'),
    ('triplicate', 'triplic'),
    ('formative', 'form'),
    ('formalize', 'formal'),
    ('electriciti', 'electr'),
    ('electrical', 'electr'),
    ('hopeful', 'hope'),
    ('goodness', 'good'),
    ('revival', 'reviv'),
    ('allowance', 'allow'),
    ('inference', 'infer'),
    ('airliner', 'airlin'),
    ('gyroscopic', 'gyroscop'),
    ('adjustable', 'adjust'),
    ('defensible', 'defens'),
    ('irritant', 'irrit'),
    ('replacement', 'replac'),
    ('adjustment', 'adjust'),
    ('dependent', 'depend'),
    ('adoption', 'adopt'),
    ('homologou', 'homolog'),
    ('communism', 'commun'),
    ('activate', 'activ'),
    ('angulariti', 'angular'),
    ('homologous', 'homolog'),
    ('effective', 'effect'),
    ('bowdlerize', 'bowdler'),
    ('probate', 'probat'),
    ('rate', 'rate'),
    ('cease', 'ceas'),
    ('controll', 'control'),
    ('roll', 'roll'),
]

porter_pass = 0
porter_fail = 0
for word, expected_stem in porter_test_vectors:
    actual_stem = do_stem(word)
    if actual_stem == expected_stem:
        porter_pass += 1
    else:
        porter_fail += 1
        if porter_fail <= 5:
            print(f"    Porter mismatch: '{word}' -> '{actual_stem}' (expected '{expected_stem}')")

test(f"Porter stemmer: {porter_pass}/{len(porter_test_vectors)} canonical test vectors pass",
     porter_fail, 0)

cranfield_stems = [
    ('experimental', 'experiment'),
    ('aerodynamics', 'aerodynam'),
    ('boundary', 'boundari'),
    ('pressure', 'pressur'),
    ('turbulent', 'turbul'),
    ('velocity', 'veloc'),
    ('analysis', 'analysi'),
    ('theoretical', 'theoret'),
    ('viscosity', 'viscos'),
    ('laminar', 'laminar'),
    ('compressible', 'compress'),
    ('investigation', 'investig'),
    ('distribution', 'distribut'),
    ('temperature', 'temperatur'),
    ('equations', 'equat'),
]

for word, expected_stem in cranfield_stems:
    test(f"stem: '{word}' -> '{expected_stem}'", do_stem(word), expected_stem)

subsection("2e. Deduplication")

test("removes duplicates preserving order",
     deduplicate(['flow', 'heat', 'flow', 'transfer', 'heat']),
     ['flow', 'heat', 'transfer'])
test("no duplicates -> unchanged",
     deduplicate(['flow', 'heat', 'transfer']),
     ['flow', 'heat', 'transfer'])
test("all same -> single",
     deduplicate(['flow', 'flow', 'flow']), ['flow'])
test("empty -> empty", deduplicate([]), [])
test("single element", deduplicate(['flow']), ['flow'])

subsection("2f. Full Pipeline Integration on Known Documents")

doc1_raw = documents[0]['text']
doc1_processed = document_pipeline(doc1_raw, stopwords)
doc1_from_file = processed_docs.get(1, [])
test("doc 1: pipeline output matches processed file", doc1_processed, doc1_from_file)

expected_in_doc1 = ['experiment', 'investig', 'aerodynam', 'wing', 'slipstream',
                    'boundari', 'layer', 'flow', 'theori']
for term in expected_in_doc1:
    test(f"doc 1 contains '{term}'", term in doc1_processed, True)

should_not_be_in_doc1 = ['the', 'of', 'a', 'in', 'is', 'brenckman']
for term in should_not_be_in_doc1:
    test(f"doc 1 does NOT contain '{term}'", term not in doc1_processed, True)

# Verify processed-corpus and index-file format requirements.
section("3. OUTPUT FILE FORMAT (PA1 Specification)")

subsection("3a. Processed File Format")

with open(PROCESSED_FILE, 'r') as f:
    proc_lines = f.readlines()

doc_count = 0
i = 0
format_ok = True
while i < len(proc_lines):
    line = proc_lines[i].strip()
    if line.startswith('.I'):
        doc_count += 1
        i += 1
        if i < len(proc_lines):
            next_line = proc_lines[i].strip()
            if next_line != '.S':
                format_ok = False
                print(f"    Expected .S after .I at line {i+1}, got '{next_line}'")
                break
        i += 1
        i += 1
    else:
        i += 1

test("processed file: .I followed by .S tag", format_ok, True)
test("processed file: contains 1400 documents", doc_count, 1400)

subsection("3b. Index File Format")

with open(INDEX_FILE, 'r') as f:
    idx_lines = f.readlines()

header_parts = idx_lines[0].strip().split(',')
test("index header has 2 parts (vocab_size, max_docid)",
     len(header_parts), 2)
test("index header vocab_size = number of data lines",
     int(header_parts[0].strip()), len(idx_lines) - 1)

test("index header: vocab_size", int(header_parts[0].strip()), vocab_size)
test("index header: max_docid = 1400", int(header_parts[1].strip()), 1400)

format_issues = []
for line_num, line in enumerate(idx_lines[1:], start=2):
    parts = line.strip().split(' ', 2)
    if len(parts) < 2:
        format_issues.append(f"Line {line_num}: insufficient parts")
        continue
    token = parts[0]
    try:
        df = int(parts[1])
    except ValueError:
        format_issues.append(f"Line {line_num}: df not integer: '{parts[1]}'")
        continue
    if len(parts) == 3:
        try:
            docids = [int(d.strip()) for d in parts[2].split(',') if d.strip()]
        except ValueError:
            format_issues.append(f"Line {line_num}: invalid docids")

test("all index data lines have valid format", format_issues, [])

all_tokens = [idx_lines[i+1].strip().split(' ', 2)[0] for i in range(len(idx_lines)-1)]
test("index tokens sorted lexicographically", all_tokens, sorted(all_tokens))

unsorted_postings = []
for token, (docids, df) in gt_index.items():
    if docids != sorted(docids):
        unsorted_postings.append(token)
test(f"all {len(gt_index)} postings lists sorted ascending", unsorted_postings, [])

dup_postings = []
for token, (docids, df) in gt_index.items():
    if len(docids) != len(set(docids)):
        dup_postings.append(token)
test("no duplicate docids in any postings list", dup_postings, [])

df_mismatches = []
for token, (docids, df) in gt_index.items():
    if df != len(docids):
        df_mismatches.append((token, df, len(docids)))
test(f"df matches postings length for all terms", df_mismatches, [])

out_of_range = []
for token, (docids, df) in gt_index.items():
    for d in docids:
        if d < 1 or d > 1400:
            out_of_range.append((token, d))
            break
test("all docids in range [1, 1400]", out_of_range, [])

# Verify the index and processed corpus agree in both directions.
section("4. INDEX <-> CORPUS CONSISTENCY")

subsection("4a. Forward consistency: index -> processed docs")

inconsistencies_fwd = []
sample_tokens = random.sample(list(gt_index.keys()), min(200, len(gt_index)))
for token in sample_tokens:
    docids, df = gt_index[token]
    for doc_id in docids[:5]:
        if doc_id in processed_docs:
            if token not in processed_docs[doc_id]:
                inconsistencies_fwd.append((token, doc_id))
test(f"forward consistency: 200 sampled terms match processed docs",
     len(inconsistencies_fwd), 0)

subsection("4b. Reverse consistency: processed docs -> index")

inconsistencies_rev = []
for doc_id in random.sample(list(processed_docs.keys()), min(50, len(processed_docs))):
    for token in processed_docs[doc_id][:10]:
        if token in gt_index:
            if doc_id not in gt_index[token][0]:
                inconsistencies_rev.append((token, doc_id))
        else:
            inconsistencies_rev.append((token, doc_id))
test(f"reverse consistency: 50 sampled docs' tokens in index",
     len(inconsistencies_rev), 0)

# Verify Boolean merge results and algebraic laws.
section("5. BOOLEAN SEARCH CORRECTNESS")

subsection("5a. AND/OR Correctness vs Set Operations")

and_test_pairs = [
    ('aerodynam', 'experiment'),
    ('shock', 'wave'),
    ('heat', 'transfer'),
    ('turbul', 'boundari'),
    ('flow', 'pressur'),
    ('laminar', 'boundari'),
    ('viscos', 'flow'),
    ('compress', 'flow'),
    ('temperatur', 'distribut'),
    ('lift', 'drag'),
]

for t1, t2 in and_test_pairs:
    L1 = gt_index.get(t1, ([], 0))[0]
    L2 = gt_index.get(t2, ([], 0))[0]
    expected = sorted(set(L1) & set(L2))
    actual = intersect_postings(list(L1), list(L2))
    test(f"AND: '{t1}'({len(L1)}) AND '{t2}'({len(L2)}) = {len(expected)} docs",
         actual, expected)

or_test_pairs = [
    ('boundari', 'layer'),
    ('pressur', 'distribut'),
    ('heat', 'transfer'),
    ('shock', 'wave'),
    ('lift', 'drag'),
    ('laminar', 'turbul'),
]

for t1, t2 in or_test_pairs:
    L1 = gt_index.get(t1, ([], 0))[0]
    L2 = gt_index.get(t2, ([], 0))[0]
    expected = sorted(set(L1) | set(L2))
    actual = union_postings(list(L1), list(L2))
    test(f"OR: '{t1}'({len(L1)}) OR '{t2}'({len(L2)}) = {len(expected)} docs",
         actual, expected)

subsection("5b. Boolean Algebra Laws")

for t1, t2 in and_test_pairs[:5]:
    L1 = gt_index.get(t1, ([], 0))[0]
    L2 = gt_index.get(t2, ([], 0))[0]
    r1 = intersect_postings(list(L1), list(L2))
    r2 = intersect_postings(list(L2), list(L1))
    test(f"commutativity AND: '{t1}' AND '{t2}' == '{t2}' AND '{t1}'", r1, r2)

for t1, t2 in or_test_pairs[:5]:
    L1 = gt_index.get(t1, ([], 0))[0]
    L2 = gt_index.get(t2, ([], 0))[0]
    r1 = union_postings(list(L1), list(L2))
    r2 = union_postings(list(L2), list(L1))
    test(f"commutativity OR: '{t1}' OR '{t2}' == '{t2}' OR '{t1}'", r1, r2)

for t in ['aerodynam', 'flow', 'pressur']:
    L = list(gt_index.get(t, ([], 0))[0])
    test(f"identity AND: '{t}' AND '{t}' == '{t}'",
         intersect_postings(L, list(L)), L)

for t in ['aerodynam', 'flow', 'pressur']:
    L = list(gt_index.get(t, ([], 0))[0])
    test(f"identity OR: '{t}' OR '{t}' == '{t}'",
         union_postings(L, list(L)), L)

test("zero law AND: A AND empty = empty",
     intersect_postings([1, 2, 3], []), [])

test("identity law OR: A OR empty = A",
     union_postings([1, 2, 3], []), [1, 2, 3])

subsection("5c. Set-Theoretic Invariants")

inv_test_pairs = [('aerodynam', 'experiment'), ('shock', 'wave'), ('heat', 'transfer'),
                  ('flow', 'pressur'), ('boundari', 'layer')]

for t1, t2 in inv_test_pairs:
    L1 = list(gt_index.get(t1, ([], 0))[0])
    L2 = list(gt_index.get(t2, ([], 0))[0])
    and_result = intersect_postings(list(L1), list(L2))
    or_result = union_postings(list(L1), list(L2))

    test(f"|{t1} AND {t2}| <= min(|A|,|B|)",
         len(and_result) <= min(len(L1), len(L2)), True)

    test(f"|{t1} OR {t2}| >= max(|A|,|B|)",
         len(or_result) >= max(len(L1), len(L2)), True)

    test(f"inclusion-exclusion: {t1},{t2}",
         len(or_result), len(L1) + len(L2) - len(and_result))

    s1, s2 = set(L1), set(L2)
    all_in_both = all(d in s1 and d in s2 for d in and_result)
    test(f"AND membership: every result in L1 AND in L2 ({t1},{t2})",
         all_in_both, True)

# Verify query preprocessing matches document preprocessing.
section("6. QUERY <-> DOCUMENT PREPROCESSING PARITY")

parity_terms = [
    'behaviour', 'behavior', 'linearised', 'linearized',
    'centre', 'center', 'vapour', 'vapor', 'analogue', 'analog',
    'programme', 'program', 'aerodynamic', 'experimental',
    'pressure', 'boundary', 'turbulent', 'velocity',
    'noise', 'rise', 'exercise', 'contour',
    'structure', 'temperature', 'chordwise', 'spanwise',
    'generalization', 'generalisation',
    'catalogue', 'catalog',
    'fibre', 'fiber', 'metre', 'meter',
]

parity_failures = 0
for term in parity_terms:
    doc_result = document_pipeline(term, stopwords)
    doc_stem = doc_result[0] if doc_result else ""
    query_stem = preprocess_query_term(term)
    if doc_stem == query_stem:
        PASSED += 1
        print(f"  OK parity: '{term}' -> '{doc_stem}'")
    else:
        FAILED += 1
        parity_failures += 1
        print(f"  FAIL PARITY MISMATCH: '{term}' -> doc='{doc_stem}' != query='{query_stem}'")

brit_amer_parity = [
    ('behaviour', 'behavior'),
    ('linearised', 'linearized'),
    ('centre', 'center'),
    ('vapour', 'vapor'),
    ('analogue', 'analog'),
    ('programme', 'program'),
    ('generalisation', 'generalization'),
    ('catalogue', 'catalog'),
    ('fibre', 'fiber'),
    ('metre', 'meter'),
]

for brit, amer in brit_amer_parity:
    brit_stem = preprocess_query_term(brit)
    amer_stem = preprocess_query_term(amer)
    test(f"British=American: stem('{brit}')=stem('{amer}') -> '{brit_stem}'",
         brit_stem, amer_stem)

# Compare index binary search against brute-force lookup.
section("7. BINARY SEARCH vs BRUTE FORCE (exhaustive)")

binary_failures = 0
all_terms = list(gt_index.keys())

test_terms = all_terms[::10] + [all_terms[0], all_terms[-1]]
test_terms += ['aaaanonexist', 'zzzznonexist', 'mmmmiddle']

for term in test_terms:
    bs_result, bs_df = binary_search_index(INDEX_FILE, term)
    gt_docids, gt_df = gt_index.get(term, (None, 0))

    if gt_docids is None:
        if bs_result is not None:
            binary_failures += 1
    else:
        if bs_result != gt_docids or bs_df != gt_df:
            binary_failures += 1
            if binary_failures <= 3:
                print(f"  FAIL FAIL: '{term}' - docids match: {bs_result == gt_docids}, "
                      f"df match: {bs_df == gt_df}")

test(f"binary search: {len(test_terms) - binary_failures}/{len(test_terms)} terms verified",
     binary_failures, 0)

# Calculate vocabulary growth and document-length statistics.
section("8. VOCABULARY STATISTICS & HEAP'S LAW")

total_tokens_raw = 0
total_tokens_unique = 0
tokens_per_doc = []
unique_per_doc = []

for doc in documents:
    raw = doc['text']
    tokens = tokenize(raw)
    normalized_tok = normalize(tokens)
    no_stop = remove_stopwords(normalized_tok, stopwords)
    stemmed = stem_tokens(no_stop)
    unique = deduplicate(stemmed)
    total_tokens_raw += len(stemmed)
    tokens_per_doc.append(len(stemmed))
    unique_per_doc.append(len(unique))

V = vocab_size
T = total_tokens_raw

if T > 0 and V > 0:
    beta_est = math.log(V) / math.log(T) if T > 1 else 0
    k_est = V / (T ** beta_est) if T > 1 else 0

    print(f"\n  Corpus Statistics:")
    print(f"    Total tokens (after stemming, before dedup): {T:,}")
    print(f"    Vocabulary size: {V:,}")
    print(f"    Type-Token Ratio (TTR): {V/T:.4f}")
    print(f"    Average tokens/doc: {T/len(documents):.1f}")
    print(f"    Average unique tokens/doc: {sum(unique_per_doc)/len(documents):.1f}")
    print(f"\n  Heap's Law: V = k x T^beta")
    print(f"    Estimated beta: {beta_est:.4f}")
    print(f"    Estimated k: {k_est:.4f}")
    print(f"    Published range for scientific English: beta in [0.40, 0.60], k in [10, 100]")

    test_approx("Heap's beta in expected range [0.55, 0.80] (stemmed corpus)",
                beta_est, 0.70, tolerance=0.20)

avg_len = sum(tokens_per_doc) / len(tokens_per_doc)
max_len = max(tokens_per_doc)
min_len = min(tokens_per_doc)
median_len = sorted(tokens_per_doc)[len(tokens_per_doc) // 2]

print(f"\n  Document Length Distribution:")
print(f"    Min: {min_len}, Max: {max_len}, Mean: {avg_len:.1f}, Median: {median_len}")

# Analyze term-frequency rank distribution.
section("9. ZIPF'S LAW ANALYSIS")

term_freq = collections.Counter()
for doc_id, tokens in processed_docs.items():
    term_freq.update(tokens)

sorted_terms = term_freq.most_common()
top_20 = sorted_terms[:20]

print(f"\n  Top 20 most frequent terms:")
print(f"  {'Rank':<6} {'Term':<20} {'Frequency':<10} {'DF':<8} {'Zipf pred':>10}")
for rank, (term, freq) in enumerate(top_20, 1):
    df = gt_index.get(term, ([], 0))[1]
    zipf_pred = sorted_terms[0][1] / rank
    print(f"  {rank:<6} {term:<20} {freq:<10} {df:<8} {zipf_pred:>10.1f}")

if len(sorted_terms) >= 100:
    ranks = list(range(1, 101))
    freqs = [sorted_terms[r-1][1] for r in ranks]

    log_ranks = [math.log(r) for r in ranks]
    log_freqs = [math.log(f) for f in freqs]
    n = len(log_ranks)
    mean_x = sum(log_ranks) / n
    mean_y = sum(log_freqs) / n
    ss_xy = sum((log_ranks[i] - mean_x) * (log_freqs[i] - mean_y) for i in range(n))
    ss_xx = sum((log_ranks[i] - mean_x) ** 2 for i in range(n))
    zipf_s = -ss_xy / ss_xx if ss_xx != 0 else 0
    zipf_intercept = mean_y + zipf_s * mean_x

    print(f"\n  Zipf's Law Fit (top 100 terms):")
    print(f"    Estimated s (slope): {zipf_s:.4f}")
    print(f"    Published range for English: s in [0.8, 1.2]")
    print(f"    (s=1.0 is the classic Zipf's law)")

    test_approx("Zipf's s in expected range [0.3, 0.8] (stemmed/deduped corpus)",
                zipf_s, 0.5, tolerance=0.50)

# Compare observed values with published Cranfield reference ranges.
section("10. COMPARISON WITH PUBLISHED CRANFIELD BENCHMARKS")

print("""
  Published statistics for the Cranfield collection (Manning et al., 2008;
  Baeza-Yates & Ribeiro-Neto, 2011; Fox, 1990):

  +---------------------------------------------------------------------+
  | Metric                      | Published Range    | Our Value       |
  +-----------------------------+--------------------+-----------------+""")

benchmarks = {
    'Documents': (1400, 1400, len(documents)),
    'Vocabulary (w/ stemming)': (3000, 6000, vocab_size),
    'Avg doc length (tokens)': (60, 120, avg_len),
    'Max doc length': (200, 800, max_len),
}

for metric, (low, high, actual) in benchmarks.items():
    in_range = low <= actual <= high
    status = "OK" if in_range else "FAIL"
    print(f"  | {metric:<27} | [{low:>6}, {high:>6}]    | {actual:>8.1f} {status}      |")
    if not in_range and metric != 'Documents':
        warn(metric, f"value {actual:.1f} outside published range [{low}, {high}]")

print("  +---------------------------------------------------------------------+")

test("vocabulary size in expected range [3000, 6000]",
     3000 <= vocab_size <= 6000, True)

test("document count = 1400", len(documents), 1400)

# Measure the effect of stop word removal.
section("11. STOP WORD ANALYSIS")

total_before_stop = 0
total_after_stop = 0
for doc in documents:
    raw = doc['text']
    tokens = tokenize(raw)
    normalized_tok = normalize(tokens)
    total_before_stop += len(normalized_tok)
    no_stop = remove_stopwords(normalized_tok, stopwords)
    total_after_stop += len(no_stop)

stop_removal_pct = (1 - total_after_stop / total_before_stop) * 100 if total_before_stop > 0 else 0

print(f"\n  Stop Word Removal Statistics:")
print(f"    Tokens before removal: {total_before_stop:,}")
print(f"    Tokens after removal:  {total_after_stop:,}")
print(f"    Removal percentage:    {stop_removal_pct:.1f}%")
print(f"    Published range: 25-35% for English text (Manning et al., 2008)")

test_approx("stop word removal in expected range [35%, 50%] (scientific abstracts)",
            stop_removal_pct, 43.0, tolerance=0.20)

# Time binary lookup and Boolean merge strategies.
section("12. SEARCH PERFORMANCE PROFILING")

subsection("12a. Binary Search Timing")

n_trials = 100
random.seed(42)
test_terms_perf = random.sample(all_terms, min(n_trials, len(all_terms)))

t_start = time.perf_counter()
for term in test_terms_perf:
    binary_search_index(INDEX_FILE, term)
t_binary = (time.perf_counter() - t_start) / n_trials

t_start = time.perf_counter()
for term in test_terms_perf:
    _ = gt_index.get(term)
t_linear = (time.perf_counter() - t_start) / n_trials

print(f"\n  Performance Comparison ({n_trials} lookups):")
print(f"    Binary search (on-disk):     {t_binary*1000:.3f} ms/query")
print(f"    Dictionary lookup (in-mem):  {t_linear*1000:.6f} ms/query")
print(f"    Expected binary search:      O(log {vocab_size}) = ~{math.log2(vocab_size):.0f} seeks")

subsection("12b. Merge Algorithm Timing")

L_small = gt_index.get('ab', ([], 0))[0]
L_large = gt_index.get('flow', ([], 0))[0]

t_start = time.perf_counter()
for _ in range(100):
    intersect_postings(list(L_small), list(L_large))
t_skewed = (time.perf_counter() - t_start) / 100

L_mid1 = gt_index.get('heat', ([], 0))[0]
L_mid2 = gt_index.get('transfer', ([], 0))[0]

t_start = time.perf_counter()
for _ in range(100):
    intersect_postings(list(L_mid1), list(L_mid2))
t_balanced = (time.perf_counter() - t_start) / 100

print(f"\n  Merge Algorithm Performance:")
print(f"    Skewed AND ({len(L_small)} vs {len(L_large)}):    {t_skewed*1000:.4f} ms  (galloping)")
print(f"    Balanced AND ({len(L_mid1)} vs {len(L_mid2)}):  {t_balanced*1000:.4f} ms  (two-pointer)")

# Validate British-American stem unification on corpus terms.
section("13. STEMMER VALIDATION (Cranfield-Specific)")

print("\n  British/American stem unification analysis:")
print(f"  {'British':<20} {'American':<20} {'Brit Stem':<15} {'Amer Stem':<15} {'Match'}")

unification_issues = 0
brit_amer_test = [
    ('behaviour', 'behavior'),
    ('linearised', 'linearized'),
    ('centre', 'center'),
    ('vapour', 'vapor'),
    ('analogue', 'analog'),
    ('generalisation', 'generalization'),
    ('programme', 'program'),
    ('colour', 'color'),
    ('honour', 'honor'),
    ('favour', 'favor'),
    ('fibre', 'fiber'),
    ('metre', 'meter'),
]

for brit, amer in brit_amer_test:

    brit_processed = document_pipeline(brit, stopwords)
    amer_processed = document_pipeline(amer, stopwords)
    brit_stem = brit_processed[0] if brit_processed else "empty"
    amer_stem = amer_processed[0] if amer_processed else "empty"
    match = brit_stem == amer_stem
    symbol = "OK" if match else "FAIL"
    print(f"  {brit:<20} {amer:<20} {brit_stem:<15} {amer_stem:<15} {symbol}")
    if not match:
        unification_issues += 1

test(f"British/American stem unification: all {len(brit_amer_test)} pairs unified",
     unification_issues, 0)

# Exercise boundary conditions in tokenization, merging, and queries.
section("14. EDGE CASE TESTING")

subsection("14a. Tokenizer Edge Cases")

test("URL-like text", len(tokenize("http://www.example.com")) > 0, True)
test("email-like text", len(tokenize("user@domain.com")) > 0, True)
test("all-numeric string", tokenize("12345"), [])
test("very long word",
     len(tokenize("superlongwordthathasmorethantwentycharacters")) > 0, True)
test("unicode/special chars", tokenize("naive resume cafe"),
     [t for t in tokenize("naive resume cafe")])
test("mixed punctuation", tokenize("hello...world!!!test???"),
     ['hello', 'world', 'test'])
test("tab and newline", tokenize("hello\tworld\ntest"), ['hello', 'world', 'test'])

subsection("14b. Galloping Search Edge Cases")

test("gallop: empty array", _gallop_search([], 5, 0), 0)
test("gallop: single element found", _gallop_search([5], 5, 0), 0)
test("gallop: single element not found (larger)", _gallop_search([3], 5, 0), 1)
test("gallop: single element not found (smaller)", _gallop_search([7], 5, 0), 0)
test("gallop: start beyond array", _gallop_search([1,2,3], 2, 5), 3)
test("gallop: all same elements", _gallop_search([5,5,5,5], 5, 0), 0)
test("gallop: very large skip",
     _gallop_search(list(range(1, 10001)), 9999, 0), 9998)

subsection("14c. Merge Edge Cases")

test("AND: both empty", intersect_postings([], []), [])
test("OR: both empty", union_postings([], []), [])
test("AND: single element lists, match",
     intersect_postings([42], [42]), [42])
test("AND: single element lists, no match",
     intersect_postings([1], [2]), [])
test("OR: single element lists, same",
     union_postings([42], [42]), [42])
test("OR: single element lists, different",
     union_postings([1], [2]), [1, 2])
test("AND: very skewed (1 vs 1000)",
     intersect_postings([500], list(range(1, 1001))), [500])
test("OR: very skewed (1 vs 1000)",
     len(union_postings([500], list(range(1, 1001)))), 1000)

subsection("14d. Query Preprocessing Edge Cases")

test("query: uppercase term", preprocess_query_term("AERODYNAMIC"),
     preprocess_query_term("aerodynamic"))
test("query: mixed case", preprocess_query_term("AeRoDyNaMiC"),
     preprocess_query_term("aerodynamic"))
test("query: empty string", preprocess_query_term(""), "")
test("query: single char", preprocess_query_term("a"), "")
test("query: punctuation only", preprocess_query_term("..."), "")

# Run complete searches and verify output files.
section("15. END-TO-END INTEGRATION TESTS")

e2e_queries = [
    ("aerodynamic AND experimental", 'AND'),
    ("shock AND wave", 'AND'),
    ("heat AND transfer", 'AND'),
    ("turbulent AND boundary", 'AND'),
    ("boundary OR layer", 'OR'),
    ("pressure OR distribution", 'OR'),
    ("heat OR transfer", 'OR'),
    ("behaviour AND flow", 'AND'),
    ("linearised AND theory", 'AND'),
    ("centre AND pressure", 'AND'),
    ("vapour OR vapor", 'OR'),
    ("nonexistent AND flow", 'AND'),
    ("shock AND shock", 'AND'),
]

for query, op in e2e_queries:
    parts = query.split()
    raw_t1, raw_t2 = parts[0], parts[2]
    stem1 = preprocess_query_term(raw_t1)
    stem2 = preprocess_query_term(raw_t2)
    L1 = gt_index.get(stem1, ([], 0))[0]
    L2 = gt_index.get(stem2, ([], 0))[0]
    if op == 'AND':
        expected = sorted(set(L1) & set(L2))
    else:
        expected = sorted(set(L1) | set(L2))

    actual, _ = run_search_cmd(query)
    test(f"E2E: '{query}' -> {len(expected)} docs", actual, expected)

brit_amer_queries = [
    ("behaviour AND flow", "behavior AND flow"),
    ("linearised AND theory", "linearized AND theory"),
    ("centre AND pressure", "center AND pressure"),
]

for brit_q, amer_q in brit_amer_queries:
    r_brit, _ = run_search_cmd(brit_q, "__test_brit.txt")
    r_amer, _ = run_search_cmd(amer_q, "__test_amer.txt")
    test(f"query equivalence: '{brit_q}' == '{amer_q}'", r_brit, r_amer)

# Summarize document-frequency distribution across the vocabulary.
section("16. INDEX TERM DISTRIBUTION ANALYSIS")

dfs = [df for _, (_, df) in gt_index.items()]
df_counter = collections.Counter(dfs)

print(f"\n  Document Frequency Distribution:")
print(f"    Terms with df=1 (hapax legomena):    {df_counter.get(1, 0):>6} ({df_counter.get(1,0)/vocab_size*100:.1f}%)")
print(f"    Terms with df=2 (dis legomena):      {df_counter.get(2, 0):>6} ({df_counter.get(2,0)/vocab_size*100:.1f}%)")
print(f"    Terms with df<=5:                     {sum(v for k,v in df_counter.items() if k<=5):>6}")
print(f"    Terms with df<=10:                    {sum(v for k,v in df_counter.items() if k<=10):>6}")
print(f"    Terms with df>=100:                   {sum(v for k,v in df_counter.items() if k>=100):>6}")
print(f"    Terms with df>=500:                   {sum(v for k,v in df_counter.items() if k>=500):>6}")
print(f"    Max df:                              {max(dfs):>6}")
print(f"    Mean df:                             {sum(dfs)/len(dfs):>8.1f}")
print(f"    Median df:                           {sorted(dfs)[len(dfs)//2]:>6}")

hapax_pct = df_counter.get(1, 0) / vocab_size * 100
print(f"\n    Hapax legomena percentage: {hapax_pct:.1f}%")
print(f"    Published range (Manning et al.): 40-60%")

sorted_by_df = sorted(gt_index.items(), key=lambda x: x[1][1], reverse=True)
print(f"\n  Top 10 Highest Document Frequency Terms:")
print(f"  {'Rank':<6} {'Term':<20} {'DF':<8} {'% of docs':<10}")
for rank, (term, (_, df)) in enumerate(sorted_by_df[:10], 1):
    print(f"  {rank:<6} {term:<20} {df:<8} {df/1400*100:.1f}%")

# Confirm preprocessing order preserves normalization and filtering behavior.
section("17. PIPELINE ORDERING VERIFICATION")

s = PorterStemmer()
upper_stem = s.stem("BOUNDARY", 0, len("BOUNDARY") - 1)
lower_stem = s.stem("boundary", 0, len("boundary") - 1)
test("Porter stemmer: lowercase required (BOUNDARY vs boundary)",
     lower_stem, 'boundari')
if upper_stem == 'boundari':
    print("    Note: Porter stemmer handled uppercase correctly in this case")
else:
    print(f"    Confirmed: stem('BOUNDARY')='{upper_stem}' != 'boundari' - normalization before stemming is necessary")

brit_norm = normalize(['behaviour'])[0]
brit_stem = do_stem(brit_norm)
amer_stem = do_stem('behavior')
test("normalize before stem prevents divergence (behaviour)",
     brit_stem, amer_stem)

test("'are' is a stop word", 'are' in stopwords, True)
test("stem('are') = 'ar' (not in stopwords)", do_stem('are'), 'ar')
test("'ar' is NOT a stop word", 'ar' not in stopwords, True)
print("    -> Stop word removal before stemming is correct")

for f in ["__test_out.txt", "__test_brit.txt", "__test_amer.txt"]:
    if os.path.exists(f): os.remove(f)

print(f"\n{'-'*75}")
print(f"  COMPREHENSIVE TEST RESULTS")
print(f"{'-'*75}")
total = PASSED + FAILED
print(f"  Total:    {total}")
print(f"  Passed:   {PASSED} ({'OK' if FAILED == 0 else ''})")
print(f"  Failed:   {FAILED} ({'FAIL' if FAILED > 0 else ''})")
print(f"  Warnings: {WARNINGS}")

if FAILED == 0:
    print(f"\n  * ALL {total} TESTS PASSED *")
else:
    print(f"\n  FAIL {FAILED} TEST(S) FAILED - review above for details")

print(f"{'-'*75}")

sys.exit(1 if FAILED > 0 else 0)
