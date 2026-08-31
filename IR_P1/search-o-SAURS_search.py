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
    Binary search directly on the sorted indxe file.

    The index file is sorted lexicographically by token. Instead of loading
    all 4,183 lines into memory (O(V) space and time), we perform binary
    search by byte position on the file.

    COMPLEXITY:
      Time:  O(log V) where V = vocabulary size
      Space: O(1) only one line in memory at a time

    INDEX FORMAT (each data line):
      token df docid1,docid2,...
      e.g.: "experiment 339 1,11,12,16,..."

    Returns:
        (list[int], int) - (postings list, document frequency) if found
        (None, 0) - if term is not in the index
    """
    if not os.path.exists(filepath):
        print(f"Error: Index file not found at {filepath}", file=sys.stderr)
        sys.exit(1)

    CHUNK_SIZE = 256

    # Use byte offsets to search the sorted index without loading it all.
    with open(filepath, 'rb') as f:

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

            # Find the beginning of the line containing the midpoint.
            if mid > data_start:
                scan_pos = mid
                line_start = data_start
                while scan_pos > data_start:

                    # Scan backward in chunks instead of one byte at a time.
                    read_start = max(data_start, scan_pos - CHUNK_SIZE)
                    chunk_len = scan_pos - read_start
                    f.seek(read_start)
                    chunk = f.read(chunk_len)

                    newline_pos = chunk.rfind(b'\n')
                    if newline_pos != -1:

                        line_start = read_start + newline_pos + 1
                        break

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

            # Parse a token, its document frequency, and its postings list.
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

def _gallop_search(arr, target, start):
    """
    Galloping (exponential) search: Find the position of 'target' in 'arr'
    starting from indxe 'start'.

    1. Exponential jump: start at step=1, double each time (1,2,4,8,16,...)
       until we oversoot (arr[pos] >= target).
    2. Binary search: within the last interval [prev_pos, pos].

    COMPLEXITY: O(log d) where d = distance from 'start' to the target's
    position.

    Returns:
        int - index where arr[index] >= target (or len(arr) if not found)
    """
    n = len(arr)
    if start >= n:
        return n

    step = 1
    pos = start
    while pos < n and arr[pos] < target:
        pos += step
        step *= 2

    lo = pos - step // 2
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
            return mid
    return lo

# Use galloping search only when the postings lists are sufficiently skewed.
GALLOP_THRESHOLD = 10

def intersect_postings(L1, L2):
    """
    AND (intersection) with adaptive algorithm selection.

    Two algorithms are available:
      1. Two-pointer merge: O(n + m) - optimal when lists are similar size
      2. Galloping search: O(k * log(n/k)) - optimal when lists are skewed

      If len(longer) / len(shorter) > 10, use galloping.
      Otherwise, use two-pointer.
    """
    if not L1 or not L2:
        return []

    # Drive the search with the shorter postings list.
    if len(L1) > len(L2):
        shorter, longer = L2, L1
    else:
        shorter, longer = L1, L2

    ratio = len(longer) / len(shorter) if len(shorter) > 0 else float('inf')

    # Choose the merge strategy from the postings-list size ratio.
    if ratio > GALLOP_THRESHOLD:

        # Gallop through the larger list from the previous match.
        result = []
        longer_idx = 0
        for doc in shorter:

            longer_idx = _gallop_search(longer, doc, longer_idx)
            if longer_idx < len(longer) and longer[longer_idx] == doc:
                result.append(doc)
                longer_idx += 1
        algo = "galloping"
        comparisons = f"O({len(shorter)}*log({len(longer)}/{len(shorter)}))"
    else:

        # Walk both similarly sized lists in sorted order.
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

    NOTE: Galloping doesn't help for union - we must visit every element
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
    to documents during indexing. This is CRITICAL for correctness - if the
    query normalization differs from the index normalization, terms won't match.

    Pipeline: clean -> lowercase -> British-American normalize -> stem
    """

    # Match document preprocessing by removing punctuation and folding case.
    term_cleaned = re.sub(r'[^a-zA-Z0-9]', '', term)

    if not term_cleaned:
        return ""

    token = term_cleaned.lower()

    # Mirror the document normalizer's spelling rules and exceptions.
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

    if token.endswith('mme') and len(token) > 4:
        token = token[:-2]

    if len(token) < 2:
        return ""

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

    # Validate and split the required two-term Boolean query format.
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

    # Normalize and stem query terms before looking them up.
    term1 = preprocess_query_term(term1_raw)
    term2 = preprocess_query_term(term2_raw)

    print(f"Original Query: '{query_str}'")
    print(f"Parsed Terms: '{term1}' {operator} '{term2}'")

    print(f"Searching index file: {index_filepath}")
    vocab_size, max_docid = read_index_header(index_filepath)
    print(f"  Index contains {vocab_size:,} terms, max docid = {max_docid}")

    # Retrieve postings lists directly from the sorted index file.
    L1, df1 = binary_search_index(index_filepath, term1)
    L2, df2 = binary_search_index(index_filepath, term2)

    if L1 is None:
        print(f"  Warning: '{term1}' not found in index. Treating as empty postings list.")
        L1 = []
    if L2 is None:
        print(f"  Warning: '{term2}' not found in index. Treating as empty postings list.")
        L2 = []

    print(f"Postings for '{term1}' (size {len(L1)}): {L1[:10]}..." if len(L1) > 10 else f"Postings for '{term1}' (size {len(L1)}): {L1}")
    print(f"Postings for '{term2}' (size {len(L2)}): {L2[:10]}..." if len(L2) > 10 else f"Postings for '{term2}' (size {len(L2)}): {L2}")

    # Combine postings according to the Boolean operator.
    if operator == 'AND':
        results = intersect_postings(L1, L2)
    else:
        results = union_postings(L1, L2)

    print(f"Match count: {len(results)}")

    # Write the result count and comma-separated document IDs on one line.
    try:
        with open(output_filepath, 'w') as out:
            out.write(f"{len(results)} | {', '.join(str(docid) for docid in results)}\n")
        print(f"Results written to {output_filepath}")
    except IOError as e:
        print(f"Error writing to output file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
