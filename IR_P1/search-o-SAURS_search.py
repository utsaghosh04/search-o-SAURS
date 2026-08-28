import sys
import os
import re
import time
from datetime import datetime
from porter import PorterStemmer

def load_index(filepath):
    """
    Loads the inverted index file.
    Returns:
        index: dict of {token: list of doc_ids}
        vocab_size: int
        max_docid: int
    """
    index = {}
    vocab_size = 0
    max_docid = 0
    
    if not os.path.exists(filepath):
        print(f"Error: Index file not found at {filepath}", file=sys.stderr)
        sys.exit(1)
        
    with open(filepath, 'r') as f:
        # Read the first line containing vocab_size and max_docid
        header = f.readline().strip()
        if header:
            parts = header.split(',')
            if len(parts) == 2:
                vocab_size = int(parts[0].strip())
                max_docid = int(parts[1].strip())
                
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            parts = line_str.split(' ', 1)
            if len(parts) == 2:
                token = parts[0]
                docids_str = parts[1]
                docids = [int(d.strip()) for d in docids_str.split(',') if d.strip()]
                index[token] = docids
            elif len(parts) == 1:
                token = parts[0]
                index[token] = []
                
    return index, vocab_size, max_docid

def preprocess_query_term(term):
    """
    Applies normalization and stemming to a single query term.
    """
    # Normalization: Lowercase
    term_normalized = term.lower()
    # Strip any punctuation
    term_cleaned = re.sub(r'[^a-zA-Z0-9]', '', term_normalized)
    
    if not term_cleaned:
        return ""
        
    # Stemming
    stemmer = PorterStemmer()
    return stemmer.stem(term_cleaned, 0, len(term_cleaned) - 1)

def intersect_postings(L1, L2):
    """
    Computes intersection of L1 and L2 in O(len(L1) + len(L2)) using two-pointer sweep.
    """
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
    return result

def union_postings(L1, L2):
    """
    Computes union of L1 and L2 in O(len(L1) + len(L2)) using two-pointer sweep.
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
    return result

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 search-o-SAURS_search.py \"<Query>\" <Output_File_Path> [Index_File_Path]", file=sys.stderr)
        print("Example: python3 search-o-SAURS_search.py \"aerodynamic AND experimental\" results.txt", file=sys.stderr)
        sys.exit(1)
        
    query_str = sys.argv[1]
    output_filepath = sys.argv[2]
    index_filepath = sys.argv[3] if len(sys.argv) > 3 else "search-o-SAURS_cran.index"
    
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
    
    # Load Index
    print(f"Loading index from {index_filepath}...")
    index, _, _ = load_index(index_filepath)
    
    # Get postings list
    L1 = index.get(term1, [])
    L2 = index.get(term2, [])
    
    print(f"Postings for '{term1}' (size {len(L1)}): {L1[:10]}..." if len(L1) > 10 else f"Postings for '{term1}' (size {len(L1)}): {L1}")
    print(f"Postings for '{term2}' (size {len(L2)}): {L2[:10]}..." if len(L2) > 10 else f"Postings for '{term2}' (size {len(L2)}): {L2}")
    
    # Perform Search
    start_time = time.time()
    
    if operator == 'AND':
        results = intersect_postings(L1, L2)
    else: # operator == 'OR'
        results = union_postings(L1, L2)
        
    search_time_ms = (time.time() - start_time) * 1000
    print(f"Match count: {len(results)}")
    
    # Write to output file in incremental append mode
    try:
        with open(output_filepath, 'a') as out:
            out.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n")
            out.write(f"Query: {query_str}\n")
            out.write(f"Search Time: {search_time_ms:.3f} ms\n")
            out.write(f"Match Count: {len(results)}\n")
            out.write(f"Results: {', '.join(map(str, results)) if results else 'None'}\n")
            out.write("-" * 50 + "\n\n")
            
        print(f"Results appended to {output_filepath}")
    except IOError as e:
        print(f"Error appending to output file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
