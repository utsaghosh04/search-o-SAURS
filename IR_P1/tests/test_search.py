import os
import re
import sys
import subprocess

# Configure imports so tests use the pipeline modules from this project.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
from porter import PorterStemmer

import importlib.util

# Load the preprocessor and searcher without requiring import-safe filenames.
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

# Track results and locate the generated index used as ground truth.
PASSED = 0
FAILED = 0
INDEX_FILE = os.path.join("output", "search-o-SAURS_cran.index")

# Record an assertion outcome and show helpful diagnostics on failure.
def test(name, actual, expected):
    global PASSED, FAILED
    if actual == expected:
        PASSED += 1
        print(f"  OK {name}")
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

def section(title):
    print(f"\n{'-'*70}")
    print(f"  {title}")
    print(f"{'-'*70}")

# Read the full index for comparisons against on-disk binary search.
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

# Run the command-line searcher and parse its one-line result format.
def run_search_cmd(query, output_file="__test_out.txt"):
    result = subprocess.run(
        [sys.executable, "search-o-SAURS_search.py", query, output_file],
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

def document_pipeline(text, stopwords):
    """Run the exact document preprocessing pipeline."""
    tokens = tokenize(text)
    normalized = normalize(tokens)
    no_stop = remove_stopwords(normalized, stopwords)
    stemmed = stem_tokens(no_stop)
    unique = deduplicate(stemmed)
    return unique

# Load shared test resources before exercising the pipeline.
print("=" * 70)
print("  COMPREHENSIVE BOOLEAN SEARCH PIPELINE TEST SUITE")
print("=" * 70)

stopwords = load_stopwords(os.path.join("data", "stopwords.txt"))
gt_index, vocab_size, max_docid = load_index_bruteforce(INDEX_FILE)
documents = parse_cranfield(os.path.join("data", "cran.all.1400"))
print(f"  Resources loaded: {len(stopwords)} stopwords, {vocab_size} terms, "
      f"{len(documents)} documents")

# Verify token boundary handling and noise filtering.
section("1. TOKENIZATION")

test("abbreviation: 'u.s.a.' -> 'usa'",
     tokenize("u.s.a. test"), ['usa', 'test'])
test("abbreviation: 'a.i.a.a.' -> 'aiaa'",
     tokenize("a.i.a.a. paper"), ['aiaa', 'paper'])

test("possessive: \"prandtl's\" -> 'prandtl'",
     tokenize("prandtl's method"), ['prandtl', 'method'])

test("hyphen: 'high-speed' -> 'high','speed'",
     tokenize("high-speed flow"), ['high', 'speed', 'flow'])
test("hyphen: 'two-dimensional' -> 'two','dimensional'",
     tokenize("two-dimensional"), ['two', 'dimensional'])

test("slash: 'lift/drag' -> 'lift','drag'",
     tokenize("lift/drag ratio"), ['lift', 'drag', 'ratio'])

test("numword: '10degree' -> '10','degree' -> 'degree' (10 filtered as pure numeric)",
     tokenize("10degree"), ['degree'])
test("wordnum: 'mach2' -> 'mach','2' -> 'mach' (2 filtered)",
     tokenize("mach2"), ['mach'])

test("filter: single chars removed",
     tokenize("a b c test"), ['test'])
test("filter: pure numerics removed",
     tokenize("100 200 test"), ['test'])
test("filter: mixed alphanumeric split by Stage 6 (f16 -> f + 16 -> both filtered)",
     tokenize("f16 test"), ['test'])
test("filter: alphanumeric with 2+ alpha chars kept after split",
     tokenize("re10 test"), ['re', 'test'])

test("empty string",
     tokenize(""), [])
test("only punctuation",
     tokenize("... --- ///"), [])

# Verify case folding and British-to-American spelling rules.
section("2. NORMALIZATION")

test("case: 'AERODYNAMIC' -> 'aerodynamic'",
     normalize(['AERODYNAMIC']), ['aerodynamic'])
test("case: 'MiXeD' -> 'mixed'",
     normalize(['MiXeD']), ['mixed'])

test("ise->ize: 'linearised' -> 'linearized'",
     normalize(['linearised']), ['linearized'])
test("isation->ization: 'generalisation' -> 'generalization'",
     normalize(['generalisation']), ['generalization'])
test("ising->izing: 'normalising' -> 'normalizing'",
     normalize(['normalising']), ['normalizing'])

test("ise exception: 'noise' unchanged",
     normalize(['noise']), ['noise'])
test("ise exception: 'rise' unchanged",
     normalize(['rise']), ['rise'])
test("ise exception: 'wise' unchanged",
     normalize(['wise']), ['wise'])
test("ise exception: 'exercise' unchanged",
     normalize(['exercise']), ['exercise'])
test("ise exception: 'chordwise' unchanged",
     normalize(['chordwise']), ['chordwise'])
test("ise exception: 'spanwise' unchanged",
     normalize(['spanwise']), ['spanwise'])

test("our->or: 'behaviour' -> 'behavior'",
     normalize(['behaviour']), ['behavior'])
test("our->or: 'vapour' -> 'vapor'",
     normalize(['vapour']), ['vapor'])
test("our->or: 'colour' -> 'color'",
     normalize(['colour']), ['color'])

test("our exception: 'four' unchanged",
     normalize(['four']), ['four'])
test("our exception: 'pour' unchanged",
     normalize(['pour']), ['pour'])
test("our exception: 'contour' unchanged",
     normalize(['contour']), ['contour'])
test("our exception: 'hour' unchanged",
     normalize(['hour']), ['hour'])

test("re->er: 'centre' -> 'center'",
     normalize(['centre']), ['center'])
test("re->er: 'metre' -> 'meter'",
     normalize(['metre']), ['meter'])
test("re->er: 'fibre' -> 'fiber'",
     normalize(['fibre']), ['fiber'])

test("re safe: 'structure' unchanged",
     normalize(['structure']), ['structure'])
test("re safe: 'pressure' unchanged",
     normalize(['pressure']), ['pressure'])
test("re safe: 'temperature' unchanged",
     normalize(['temperature']), ['temperature'])

test("ogue->og: 'analogue' -> 'analog'",
     normalize(['analogue']), ['analog'])
test("ogue->og: 'catalogue' -> 'catalog'",
     normalize(['catalogue']), ['catalog'])
test("ogue exception: 'vogue' unchanged",
     normalize(['vogue']), ['vogue'])

test("mme->m: 'programme' -> 'program'",
     normalize(['programme']), ['program'])
test("mme->m: 'gramme' -> 'gram'",
     normalize(['gramme']), ['gram'])

test("min length: single char removed",
     normalize(['a', 'test']), ['test'])

# Verify stop word filtering preserves meaningful terms.
section("3. STOP WORD REMOVAL")

test("removes common stop words",
     remove_stopwords(['the', 'flow', 'is', 'over', 'boundary'], stopwords),
     ['flow', 'boundary'])
test("keeps non-stop words",
     remove_stopwords(['aerodynamic', 'experimental'], stopwords),
     ['aerodynamic', 'experimental'])
test("empty input",
     remove_stopwords([], stopwords), [])
test("all stop words -> empty",
     remove_stopwords(['the', 'is', 'a', 'an', 'of'], stopwords), [])

# Verify Porter stemming on representative terms.
section("4. STEMMING")

s = PorterStemmer()
def do_stem(word):
    return s.stem(word, 0, len(word) - 1)

test("stem: 'experimental' -> 'experiment'", do_stem('experimental'), 'experiment')
test("stem: 'aerodynamics' -> 'aerodynam'", do_stem('aerodynamics'), 'aerodynam')
test("stem: 'boundary' -> 'boundari'", do_stem('boundary'), 'boundari')
test("stem: 'pressure' -> 'pressur'", do_stem('pressure'), 'pressur')
test("stem: 'turbulent' -> 'turbul'", do_stem('turbulent'), 'turbul')
test("stem: 'velocity' -> 'veloc'", do_stem('velocity'), 'veloc')
test("stem: 'analysis' -> 'analysi'", do_stem('analysis'), 'analysi')
test("stem: 'theoretical' -> 'theoret'", do_stem('theoretical'), 'theoret')

test("stem: lowercase 'boundary' -> 'boundari'", do_stem('boundary'), 'boundari')

# Verify duplicate terms are removed without changing order.
section("5. DEDUPLICATION")

test("removes duplicates preserving order",
     deduplicate(['flow', 'heat', 'flow', 'transfer', 'heat']),
     ['flow', 'heat', 'transfer'])
test("no duplicates -> unchanged",
     deduplicate(['flow', 'heat', 'transfer']),
     ['flow', 'heat', 'transfer'])
test("all same -> single",
     deduplicate(['flow', 'flow', 'flow']), ['flow'])
test("empty -> empty",
     deduplicate([]), [])

# Verify queries and documents produce identical normalized stems.
section("6. QUERY <-> DOCUMENT PREPROCESSING PARITY")

parity_terms = [
    'behaviour', 'behavior', 'linearised', 'linearized',
    'centre', 'center', 'vapour', 'vapor', 'analogue', 'analog',
    'programme', 'program', 'aerodynamic', 'experimental',
    'pressure', 'boundary', 'turbulent', 'velocity',
    'noise', 'rise', 'exercise',
    'contour',
    'structure', 'temperature',
    'chordwise', 'spanwise',
]

parity_failures = 0
for term in parity_terms:

    doc_result = document_pipeline(term, stopwords)
    doc_stem = doc_result[0] if doc_result else ""

    query_stem = preprocess_query_term(term)

    if doc_stem == query_stem:
        PASSED += 1
        print(f"  OK parity: '{term}' -> doc='{doc_stem}' == query='{query_stem}'")
    else:
        FAILED += 1
        parity_failures += 1
        print(f"  FAIL PARITY MISMATCH: '{term}' -> doc='{doc_stem}' != query='{query_stem}'")

# Verify index structure, ordering, and postings data.
section("7. INDEX CORRECTNESS")

test("index header: vocab_size = 4183", vocab_size, 4183)
test("index header: max_docid = 1400", max_docid, 1400)

all_tokens = sorted(gt_index.keys())
actual_order = list(gt_index.keys())
test("index tokens are sorted lexicographically", actual_order, all_tokens)

df_mismatches = []
for token, (docids, df) in gt_index.items():
    if df != len(docids):
        df_mismatches.append((token, df, len(docids)))
test(f"df matches postings length for all {len(gt_index)} terms",
     df_mismatches, [])

unsorted_terms = []
for token, (docids, df) in gt_index.items():
    if docids != sorted(docids):
        unsorted_terms.append(token)
test(f"postings sorted ascending for all {len(gt_index)} terms",
     unsorted_terms, [])

dup_terms = []
for token, (docids, df) in gt_index.items():
    if len(docids) != len(set(docids)):
        dup_terms.append(token)
test(f"no duplicate docids in any postings list",
     dup_terms, [])

test("doc 1 has 'aerodynam' in index",
     1 in gt_index.get('aerodynam', ([], 0))[0], True)
test("doc 1 has 'wing' in index",
     1 in gt_index.get('wing', ([], 0))[0], True)
test("doc 1 has 'slipstream' in index",
     1 in gt_index.get('slipstream', ([], 0))[0], True)

# Compare binary-search results with full-index ground truth.
section("8. BINARY SEARCH CORRECTNESS (exhaustive)")

import random
random.seed(42)
all_terms = list(gt_index.keys())

sample_size = min(50, len(all_terms))
step = len(all_terms) // sample_size
sample_terms = [all_terms[i * step] for i in range(sample_size)]

sample_terms.extend([all_terms[0], all_terms[-1]])

sample_terms.extend(['aaaanonexist', 'mmmnonexist', 'zzzznonexist'])

binary_failures = 0
for term in sample_terms:
    bs_result, bs_df = binary_search_index(INDEX_FILE, term)
    gt_docids, gt_df = gt_index.get(term, (None, 0))

    if gt_docids is None:

        if bs_result is not None:
            FAILED += 1
            binary_failures += 1
            print(f"  FAIL FAIL: '{term}' should be None, got {len(bs_result)} docs")
        else:
            PASSED += 1
    else:
        if bs_result == gt_docids and bs_df == gt_df:
            PASSED += 1
        else:
            FAILED += 1
            binary_failures += 1
            print(f"  FAIL FAIL: '{term}' - docids match: {bs_result == gt_docids}, df match: {bs_df == gt_df}")

print(f"  {'OK' if binary_failures == 0 else 'FAIL'} Binary search: {len(sample_terms) - binary_failures}/{len(sample_terms)} terms verified")

# Verify galloping and two-pointer Boolean merge behavior.
section("9. AND/OR MERGE CORRECTNESS")

test("gallop: find 5 in [1,3,5,7,9]",
     _gallop_search([1,3,5,7,9], 5, 0), 2)
test("gallop: find 1 in [1,3,5,7,9]",
     _gallop_search([1,3,5,7,9], 1, 0), 0)
test("gallop: find 9 in [1,3,5,7,9]",
     _gallop_search([1,3,5,7,9], 9, 0), 4)
test("gallop: find 4 (not present) -> returns index of 5",
     _gallop_search([1,3,5,7,9], 4, 0), 2)
test("gallop: find 10 (beyond end) -> returns len",
     _gallop_search([1,3,5,7,9], 10, 0), 5)
test("gallop: find from offset",
     _gallop_search([1,3,5,7,9,11,13,15], 11, 3), 5)

and_pairs = [
    ('aerodynam', 'experiment'),
    ('shock', 'wave'),
    ('heat', 'transfer'),
    ('turbul', 'boundari'),
    ('flow', 'pressur'),
    ('behavior', 'flow'),
    ('ab', 'flow'),
    ('zurich', 'flow'),
]

for t1, t2 in and_pairs:
    L1 = gt_index.get(t1, ([], 0))[0]
    L2 = gt_index.get(t2, ([], 0))[0]
    expected = sorted(set(L1) & set(L2))
    actual = intersect_postings(L1, L2)
    test(f"AND: '{t1}'({len(L1)}) AND '{t2}'({len(L2)}) = {len(expected)} docs",
         actual, expected)

or_pairs = [
    ('boundari', 'layer'),
    ('pressur', 'distribut'),
    ('heat', 'transfer'),
    ('shock', 'wave'),
]

for t1, t2 in or_pairs:
    L1 = gt_index.get(t1, ([], 0))[0]
    L2 = gt_index.get(t2, ([], 0))[0]
    expected = sorted(set(L1) | set(L2))
    actual = union_postings(L1, L2)
    test(f"OR: '{t1}'({len(L1)}) OR '{t2}'({len(L2)}) = {len(expected)} docs",
         actual, expected)

test("AND: empty AND non-empty = empty",
     intersect_postings([], [1, 2, 3]), [])
test("AND: non-empty AND empty = empty",
     intersect_postings([1, 2, 3], []), [])
test("AND: identical lists",
     intersect_postings([1, 3, 5], [1, 3, 5]), [1, 3, 5])
test("AND: no overlap",
     intersect_postings([1, 3, 5], [2, 4, 6]), [])
test("OR: empty OR non-empty = non-empty",
     union_postings([], [1, 2, 3]), [1, 2, 3])
test("OR: identical lists (no duplicates in result)",
     union_postings([1, 3, 5], [1, 3, 5]), [1, 3, 5])

# Exercise the full command-line pipeline with representative queries.
section("10. END-TO-END INTEGRATION (raw query -> correct results)")

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

# Check known document memberships and Boolean algebra properties.
section("11. DOCUMENT-LEVEL SPOT CHECKS (verify specific docs are in results)")

result, _ = run_search_cmd("aerodynamic AND experimental")
test("doc 1 in 'aerodynamic AND experimental'", 1 in result, True)

result, _ = run_search_cmd("heat AND transfer")
test("doc 1 NOT in 'heat AND transfer'", 1 not in result, True)

r_brit, _ = run_search_cmd("behaviour AND flow", "__test_brit.txt")
r_amer, _ = run_search_cmd("behavior AND flow", "__test_amer.txt")
test("British/American query equivalence: behaviour==behavior",
     r_brit, r_amer)

r_brit2, _ = run_search_cmd("linearised AND theory", "__test_brit2.txt")
r_amer2, _ = run_search_cmd("linearized AND theory", "__test_amer2.txt")
test("British/American query equivalence: linearised==linearized",
     r_brit2, r_amer2)

r_brit3, _ = run_search_cmd("centre AND pressure", "__test_brit3.txt")
r_amer3, _ = run_search_cmd("center AND pressure", "__test_amer3.txt")
test("British/American query equivalence: centre==center",
     r_brit3, r_amer3)

result, _ = run_search_cmd("flow AND pressure")
test("results sorted ascending", result, sorted(result))

r1, _ = run_search_cmd("shock AND wave", "__test_comm1.txt")
r2, _ = run_search_cmd("wave AND shock", "__test_comm2.txt")
test("AND commutativity: 'shock AND wave' == 'wave AND shock'", r1, r2)

r1, _ = run_search_cmd("boundary OR layer", "__test_comm3.txt")
r2, _ = run_search_cmd("layer OR boundary", "__test_comm4.txt")
test("OR commutativity: 'boundary OR layer' == 'layer OR boundary'", r1, r2)

# Check set-theoretic invariants across sampled postings lists.
section("12. PIPELINE INVARIANT CHECKS")

inv_failures = 0
test_pairs = [('aerodynam', 'experiment'), ('shock', 'wave'), ('heat', 'transfer'),
              ('ab', 'flow'), ('behavior', 'flow')]
for t1, t2 in test_pairs:
    L1 = gt_index.get(t1, ([], 0))[0]
    L2 = gt_index.get(t2, ([], 0))[0]
    result = intersect_postings(L1, L2)
    if len(result) > min(len(L1), len(L2)):
        inv_failures += 1
        print(f"  FAIL INVARIANT VIOLATED: |{t1} AND {t2}| = {len(result)} > min({len(L1)},{len(L2)})")
    else:
        PASSED += 1
test(f"AND invariant: |AANDB| <= min(|A|,|B|) for all pairs", inv_failures, 0)

inv_failures2 = 0
for t1, t2 in test_pairs:
    L1 = gt_index.get(t1, ([], 0))[0]
    L2 = gt_index.get(t2, ([], 0))[0]
    result = union_postings(L1, L2)
    if len(result) < max(len(L1), len(L2)):
        inv_failures2 += 1
        print(f"  FAIL INVARIANT VIOLATED: |{t1} OR {t2}| = {len(result)} < max({len(L1)},{len(L2)})")
    else:
        PASSED += 1
test(f"OR invariant: |AORB| >= max(|A|,|B|) for all pairs", inv_failures2, 0)

inv_failures3 = 0
for t1, t2 in test_pairs:
    L1 = gt_index.get(t1, ([], 0))[0]
    L2 = gt_index.get(t2, ([], 0))[0]
    and_result = intersect_postings(L1, L2)
    or_result = union_postings(L1, L2)

    expected_union_size = len(L1) + len(L2) - len(and_result)
    if len(or_result) != expected_union_size:
        inv_failures3 += 1
        print(f"  FAIL INCLUSION-EXCLUSION VIOLATED for {t1},{t2}: "
              f"|AORB|={len(or_result)} != |A|+|B|-|AANDB|={expected_union_size}")
    else:
        PASSED += 1
test(f"inclusion-exclusion: |AORB| = |A|+|B|-|AANDB| for all pairs", inv_failures3, 0)

inv_failures4 = 0
for t1, t2 in test_pairs:
    L1 = gt_index.get(t1, ([], 0))[0]
    L2 = gt_index.get(t2, ([], 0))[0]
    s1, s2 = set(L1), set(L2)
    result = intersect_postings(L1, L2)
    for docid in result:
        if docid not in s1 or docid not in s2:
            inv_failures4 += 1
            break
    else:
        PASSED += 1
test(f"AND membership: every result docid in L1 AND in L2", inv_failures4, 0)

for f in ["__test_out.txt", "__test_brit.txt", "__test_amer.txt",
          "__test_brit2.txt", "__test_amer2.txt", "__test_brit3.txt",
          "__test_amer3.txt", "__test_comm1.txt", "__test_comm2.txt",
          "__test_comm3.txt", "__test_comm4.txt"]:
    if os.path.exists(f): os.remove(f)

print(f"\n{'-'*70}")
total = PASSED + FAILED
if FAILED == 0:
    print(f"  ALL {total} TESTS PASSED OK")
else:
    print(f"  {PASSED}/{total} PASSED, {FAILED} FAILED FAIL")
print(f"{'-'*70}")
