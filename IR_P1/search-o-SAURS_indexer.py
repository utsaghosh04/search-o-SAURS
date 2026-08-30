import sys
import os

def build_index(input_filepath, output_filepath):
    postings = {}
    max_docid = 0
    
    current_id = None
    expecting_tokens = False
    
    try:
        with open(input_filepath, 'r') as f:
            for line in f:
                line_str = line.strip()
                if line_str.startswith('.I'):
                    parts = line_str.split()
                    if len(parts) > 1:
                        current_id = int(parts[1])
                        max_docid = max(max_docid, current_id)
                    expecting_tokens = False
                elif line_str.startswith('.S'):
                    expecting_tokens = True
                else:
                    if expecting_tokens and current_id is not None:
                        tokens = line_str.split()
                        for token in tokens:
                            if not token:
                                continue
                            if token not in postings:
                                postings[token] = []
                            # Add document ID in ascending order, avoiding duplicates
                            if not postings[token] or postings[token][-1] != current_id:
                                postings[token].append(current_id)
                        expecting_tokens = False
    except FileNotFoundError:
        print(f"Error: Preprocessed file not found at {input_filepath}", file=sys.stderr)
        sys.exit(1)
        
    sorted_tokens = sorted(postings.keys())
    vocab_size = len(sorted_tokens)
    
    try:
        with open(output_filepath, 'w') as out:
            # First line: vocabulary_size, max_docid
            out.write(f"{vocab_size}, {max_docid}\n")
            # Subsequent lines: token df docid1,docid2,...
            # The df (document frequency) enables the search engine to know
            # the postings list size without parsing it — used for query
            # term ordering in AND optimization.
            for token in sorted_tokens:
                df = len(postings[token])
                docids_str = ",".join(str(doc_id) for doc_id in postings[token])
                out.write(f"{token} {df} {docids_str}\n")
    except IOError as e:
        print(f"Error writing to index file: {e}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Indexing completed.")
    print(f"Vocabulary Size: {vocab_size}")
    print(f"Maximum DocID Indexed: {max_docid}")
    print(f"Index written to {output_filepath}")

def main():
    input_file = os.path.join("output", "search-o-SAURS_processed.all")
    output_file = os.path.join("output", "search-o-SAURS_cran.index")
    build_index(input_file, output_file)

if __name__ == "__main__":
    main()
