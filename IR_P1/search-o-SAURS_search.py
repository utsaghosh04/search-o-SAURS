import sys
import os
import re
from porter import PorterStemmer

def read_index_header(filepath):
    """
    Reads only the header line of the index file.
    Returns: (vocab_size, max_docid)
    """
    with open(filepath, 'r') as f:
        header = f.readline().strip()
        parts = header.split(',')
        return int(parts[0].strip()), int(parts[1].strip())


def binary_search_index(filepath, target_term):
    """
    Binary search directly on the sorted index file.

    ALGORITHM:
    ─────────
    The index file is sorted lexicographically by token. Instead of loading
    all 4,183 lines into memory (O(V) space and time), we perform binary
    search by byte position on the file — finding any term in O(log V)
    seeks ≈ 12 disk reads for our vocabulary.

    OPTIMIZATION: Buffered backward scan
    ─────────────────────────────────────
    When seeking to a midpoint, we need to find the start of the current
    line. Instead of reading backwards byte-by-byte (O(line_length) seeks),
    we read 256-byte chunks backwards and scan the buffer in memory:
      Byte-by-byte: up to 1,445 individual seek+read for long lines
      Buffered:     up to 6 chunk reads (1,445 / 256 ≈ 6)

    COMPLEXITY:
      Time:  O(log V)  where V = vocabulary size
      Space: O(1)      only one line in memory at a time
      I/O:   ~12 seeks for 4183 terms (log₂ 4183 ≈ 12)

    INDEX FORMAT (each data line):
      token df docid1,docid2,...
      e.g.: "experiment 339 1,11,12,16,..."
      The df (document frequency) is parsed and returned alongside postings.

    Returns:
        (list[int], int) — (postings list, document frequency) if found
        (None, 0)        — if term is not in the index
    """
    if not os.path.exists(filepath):
        print(f"Error: Index file not found at {filepath}", file=sys.stderr)
        sys.exit(1)

    CHUNK_SIZE = 256  # bytes to read at a time for backward scan

    with open(filepath, 'rb') as f:
        # Skip the header line (vocab_size, max_docid)
        f.readline()
        data_start = f.tell()

        f.seek(0, 2)
        file_end = f.tell()

        low = data_start
        high = file_end
        seek_count = 0

        while low < high:
            mid = (low + high) // 2
            seek_count += 1

            # ── Buffered backward scan to find line start ──
            # Read backwards in CHUNK_SIZE chunks instead of byte-by-byte.
            # This reduces I/O from O(line_length) to O(line_length / CHUNK_SIZE).
            if mid > data_start:
                scan_pos = mid
                line_start = data_start  # fallback if no newline found
                while scan_pos > data_start:
                    # Read a chunk ending at scan_pos
                    read_start = max(data_start, scan_pos - CHUNK_SIZE)
                    chunk_len = scan_pos - read_start
                    f.seek(read_start)
                    chunk = f.read(chunk_len)

                    # Find the LAST newline in this chunk
                    newline_pos = chunk.rfind(b'\n')
                    if newline_pos != -1:
                        # Line starts right after this newline
                        line_start = read_start + newline_pos + 1
                        break
                    # No newline in this chunk, continue scanning backwards
                    scan_pos = read_start
            else:
                line_start = data_start

            f.seek(line_start)
            raw_line = f.readline()

            if not raw_line:
                high = mid
                continue

            line = raw_line.decode('utf-8', errors='replace').strip()

            if not line:
                high = mid
                continue

            # ── Parse line: "token df docid1,docid2,..." ──
            parts = line.split(' ', 2)
            if len(parts) < 2:
                high = mid
                continue

            token = parts[0]
            df = int(parts[1]) if len(parts) >= 2 else 0
            postings_str = parts[2] if len(parts) == 3 else ""

            if token == target_term:
                if postings_str:
                    docids = [int(d.strip()) for d in postings_str.split(',') if d.strip()]
                else:
                    docids = []
                print(f"  [Binary Search] Found '{target_term}' (df={df}) in {seek_count} seeks")
                return docids, df
            elif token < target_term:
                low = line_start + len(raw_line)
            else:
                high = line_start

        print(f"  [Binary Search] '{target_term}' not found after {seek_count} seeks")
        return None, 0


# ═══════════════════════════════════════════════════════════════════════════
# BOOLEAN MERGE ALGORITHMS
# ═══════════════════════════════════════════════════════════════════════════

def _gallop_search(arr, target, start):
    """
    Galloping (exponential) search: Find the position of 'target' in 'arr'
    starting from index 'start'.

    ALGORITHM:
    ──────────
    1. Exponential jump: start at step=1, double each time (1,2,4,8,16,...)
       until we overshoot (arr[pos] >= target).
    2. Binary search: within the last interval [prev_pos, pos].

    COMPLEXITY: O(log d) where d = distance from 'start' to the target's
    position. This is much better than linear scan when d is small relative
    to the remaining list length.

    Returns:
        int — index where arr[index] >= target (or len(arr) if not found)
    """
    n = len(arr)
    if start >= n:
        return n

    # Phase 1: Exponential jump
    step = 1
    pos = start
    while pos < n and arr[pos] < target:
        pos += step
        step *= 2  # double the step size each iteration

    # Phase 2: Binary search in [prev_bound, min(pos, n-1)]
    lo = pos - step // 2  # undo last doubling to get lower bound
    if lo < start:
        lo = start
    hi = min(pos, n - 1)

    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] < target:
            lo = mid + 1
        elif arr[mid] > target:
            hi = mid - 1
        else:
            return mid  # exact match
    return lo  # first position >= target


# ── Skew ratio threshold for galloping ──
# When len(longer) / len(shorter) > GALLOP_THRESHOLD, galloping search is
# faster than two-pointer merge. Empirically, 10x is a good threshold.
GALLOP_THRESHOLD = 10


def intersect_postings(L1, L2):
    """
    AND (intersection) with adaptive algorithm selection.

    Two algorithms are available:
      1. Two-pointer merge: O(n + m)  — optimal when lists are similar size
      2. Galloping search:  O(k · log(n/k)) — optimal when lists are skewed

    DECISION:
      If len(longer) / len(shorter) > 10, use galloping.
      Otherwise, use two-pointer.

    Example:
      "ab AND flow" → ab: 1 doc, flow: 730 docs
        Two-pointer:  O(1 + 730) = 731 comparisons
        Galloping:    O(1 · log(730)) = ~10 comparisons   ← 73x faster
    """
    if not L1 or not L2:
        return []

    # Ensure shorter is the "driver" list
    if len(L1) > len(L2):
        shorter, longer = L2, L1
    else:
        shorter, longer = L1, L2

    ratio = len(longer) / len(shorter) if len(shorter) > 0 else float('inf')

    if ratio > GALLOP_THRESHOLD:
        # ── GALLOPING SEARCH ──
        # For each element in the shorter list, gallop through the longer
        # list to find it. O(k · log(n/k)) total.
        result = []
        longer_idx = 0
        for doc in shorter:
            # Gallop in the longer list starting from current position
            longer_idx = _gallop_search(longer, doc, longer_idx)
            if longer_idx < len(longer) and longer[longer_idx] == doc:
                result.append(doc)
                longer_idx += 1  # advance past the match
        algo = "galloping"
        comparisons = f"O({len(shorter)}·log({len(longer)}/{len(shorter)}))"
    else:
        # ── TWO-POINTER MERGE ──
        # Standard O(n + m) linear merge.
        result = []
        i, j = 0, 0
        while i < len(L1) and j < len(L2):
            if L1[i] == L2[j]:
                result.append(L1[i])
                i += 1
                j += 1
            elif L1[i] < L2[j]:
                i += 1
            else:
                j += 1
        algo = "two-pointer"
        comparisons = f"O({len(L1)}+{len(L2)})"

    print(f"  [AND] Algorithm: {algo} | Lists: {len(shorter)} vs {len(longer)} (ratio {ratio:.1f}x) | {comparisons}")
    return result


def union_postings(L1, L2):
    """
    OR (union) using two-pointer merge in O(n + m).

    NOTE: Galloping doesn't help for union — we must visit every element
    in both lists regardless, since all elements appear in the result.
    Two-pointer merge is optimal.
    """
    result = []
    i, j = 0, 0
    while i < len(L1) or j < len(L2):
        if i < len(L1) and j < len(L2):
            if L1[i] == L2[j]:
                result.append(L1[i])
                i += 1
                j += 1
            elif L1[i] < L2[j]:
                result.append(L1[i])
                i += 1
            else:
                result.append(L2[j])
                j += 1
        elif i < len(L1):
            result.append(L1[i])
            i += 1
        else:
            result.append(L2[j])
            j += 1
    print(f"  [OR] Algorithm: two-pointer | Lists: {len(L1)} + {len(L2)} | O({len(L1)}+{len(L2)})")
    return result

def preprocess_query_term(term):
    """
    Applies the same preprocessing pipeline to a query term as was applied
    to documents during indexing. This is CRITICAL for correctness — if the
    query normalization differs from the index normalization, terms won't match.

    Pipeline: clean → lowercase → British→American normalize → stem
    """
    # Strip any punctuation, keep alphanumeric only
    term_cleaned = re.sub(r'[^a-zA-Z0-9]', '', term)

    if not term_cleaned:
        return ""

    # Case folding
    token = term_cleaned.lower()

    # ── British → American spelling normalization ──
    # Must mirror the EXACT same rules as in the preprocessing normalize()
    # function. If you change the rules there, update here too.

    ISE_EXCEPTIONS = {
        'noise', 'rise', 'wise', 'cruise', 'promise', 'otherwise',
        'exercise', 'comprise', 'surprise', 'precise', 'concise',
        'advise', 'devise', 'revise', 'supervise', 'improvise',
        'arise', 'enterprise', 'demise', 'expertise', 'franchise',
        'merchandise', 'paradise', 'practise', 'surmise', 'disguise',
        'chordwise', 'spanwise', 'streamwise', 'lengthwise', 'crosswise',
        'likewise', 'piecewise', 'stepwise', 'pointwise',
    }

    if token not in ISE_EXCEPTIONS:
        if token.endswith('isation'):
            token = token[:-7] + 'ization'
        elif token.endswith('isations'):
            token = token[:-8] + 'izations'
        elif token.endswith('ised'):
            token = token[:-4] + 'ized'
        elif token.endswith('ising'):
            token = token[:-5] + 'izing'
        elif token.endswith('ises'):
            token = token[:-4] + 'izes'
        elif token.endswith('ise'):
            token = token[:-3] + 'ize'

    OUR_EXCEPTIONS = {
        'four', 'pour', 'your', 'our', 'contour', 'detour', 'tour',
        'glamour', 'velour', 'amour', 'dour', 'scour', 'hour',
    }
    if token not in OUR_EXCEPTIONS and token.endswith('our'):
        if len(token) > 4:
            token = token[:-3] + 'or'

    RE_SAFE_ENDINGS = ('tre', 'bre')
    if any(token.endswith(ending) for ending in RE_SAFE_ENDINGS):
        if len(token) > 3:
            token = token[:-2] + 'er'

    OGUE_EXCEPTIONS = {'vogue', 'rogue', 'brogue'}
    if token not in OGUE_EXCEPTIONS and token.endswith('ogue'):
        token = token[:-4] + 'og'

    # Pattern E: -mme → -m (programme→program)
    if token.endswith('mme') and len(token) > 4:
        token = token[:-2]

    if len(token) < 2:
        return ""

    # Stemming
    stemmer = PorterStemmer()
    return stemmer.stem(token, 0, len(token) - 1)


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 search-o-SAURS_search.py \"<Query>\" <Output_File_Path> [Index_File_Path]", file=sys.stderr)
        print("Example: python3 search-o-SAURS_search.py \"aerodynamic AND experimental\" output/results/out.txt", file=sys.stderr)
        sys.exit(1)
        
    query_str = sys.argv[1]
    output_filepath = sys.argv[2]
    index_filepath = sys.argv[3] if len(sys.argv) > 3 else os.path.join("output", "search-o-SAURS_cran.index")
    
    # Parse Query
    # Split query by whitespace
    query_parts = query_str.split()
    if len(query_parts) != 3:
        print(f"Error: Query must involve exactly two terms and one connective (AND/OR). Got: '{query_str}'", file=sys.stderr)
        sys.exit(1)
        
    term1_raw = query_parts[0]
    operator = query_parts[1].upper()
    term2_raw = query_parts[2]
    
    if operator not in ('AND', 'OR'):
        print(f"Error: Connective must be AND or OR. Got: '{operator}'", file=sys.stderr)
        sys.exit(1)
        
    # Preprocess query terms
    term1 = preprocess_query_term(term1_raw)
    term2 = preprocess_query_term(term2_raw)
    
    print(f"Original Query: '{query_str}'")
    print(f"Parsed Terms: '{term1}' {operator} '{term2}'")
    
    # ── Search the index file using binary search ──
    # Instead of loading all 4,183 entries into memory, we perform
    # O(log n) binary search directly on the sorted file.
    print(f"Searching index file: {index_filepath}")
    vocab_size, max_docid = read_index_header(index_filepath)
    print(f"  Index contains {vocab_size:,} terms, max docid = {max_docid}")
    
    # Binary search for each query term
    # Returns (postings_list, document_frequency) or (None, 0)
    L1, df1 = binary_search_index(index_filepath, term1)
    L2, df2 = binary_search_index(index_filepath, term2)
    
    # Handle terms not found in index
    if L1 is None:
        print(f"  Warning: '{term1}' not found in index. Treating as empty postings list.")
        L1 = []
    if L2 is None:
        print(f"  Warning: '{term2}' not found in index. Treating as empty postings list.")
        L2 = []
    
    print(f"Postings for '{term1}' (size {len(L1)}): {L1[:10]}..." if len(L1) > 10 else f"Postings for '{term1}' (size {len(L1)}): {L1}")
    print(f"Postings for '{term2}' (size {len(L2)}): {L2[:10]}..." if len(L2) > 10 else f"Postings for '{term2}' (size {len(L2)}): {L2}")
    
    # Perform Search
    if operator == 'AND':
        results = intersect_postings(L1, L2)
    else: # operator == 'OR'
        results = union_postings(L1, L2)
        
    print(f"Match count: {len(results)}")
    
    # Write to output file
    # We write docids one per line
    try:
        with open(output_filepath, 'w') as out:
            for docid in results:
                out.write(f"{docid}\n")
        print(f"Results written to {output_filepath}")
    except IOError as e:
        print(f"Error writing to output file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
