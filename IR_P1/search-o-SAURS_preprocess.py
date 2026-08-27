import re
import sys
from porter import PorterStemmer

def tokenize(text):
    """
    Tokenization: Convert a document text into a list of token strings.
    We split on non-alphanumeric characters, keeping words/numbers.
    """
    return re.findall(r'[a-zA-Z0-9]+', text)

def normalize(tokens):
    """
    Normalization: Convert all tokens to lowercase.
    """
    return [token.lower() for token in tokens]

def remove_stopwords(tokens, stopwords):
    """
    Stop word removal: Filter out tokens that are in the stopwords set.
    """
    return [token for token in tokens if token not in stopwords]

def stem(tokens):
    """
    Stemming: Convert tokens to their stems using the Porter stemmer.
    """
    stemmer = PorterStemmer()
    stemmed = []
    for token in tokens:
        # The stemmer expects the string, start index, and end index
        stemmed_token = stemmer.stem(token, 0, len(token) - 1)
        stemmed.append(stemmed_token)
    return stemmed

def load_stopwords(filepath):
    stopwords = set()
    try:
        with open(filepath, 'r') as f:
            for line in f:
                word = line.strip().lower()
                if word:
                    stopwords.add(word)
    except FileNotFoundError:
        print(f"Error: Stopwords file not found at {filepath}", file=sys.stderr)
        sys.exit(1)
    return stopwords

def parse_cranfield(filepath):
    """
    Parses the cranfield dataset.
    Returns a list of dicts: [{'id': doc_id, 'text': combined_title_and_abstract}]
    """
    documents = []
    current_doc = None
    current_field = None
    
    title_buffer = []
    abstract_buffer = []
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line_str = line.strip()
                if line_str.startswith('.I'):
                    # Save the previous document if any
                    if current_doc is not None:
                        combined_text = " ".join(title_buffer) + " " + " ".join(abstract_buffer)
                        current_doc['text'] = combined_text
                        documents.append(current_doc)
                    
                    doc_id = int(line_str.split()[1])
                    current_doc = {'id': doc_id, 'text': ''}
                    title_buffer = []
                    abstract_buffer = []
                    current_field = None
                elif line_str.startswith('.T'):
                    current_field = 'T'
                elif line_str.startswith('.A'):
                    current_field = 'A'
                elif line_str.startswith('.B'):
                    current_field = 'B'
                elif line_str.startswith('.W'):
                    current_field = 'W'
                else:
                    if current_field == 'T':
                        title_buffer.append(line_str)
                    elif current_field == 'W':
                        abstract_buffer.append(line_str)
            
            # Save the last document
            if current_doc is not None:
                combined_text = " ".join(title_buffer) + " " + " ".join(abstract_buffer)
                current_doc['text'] = combined_text
                documents.append(current_doc)
    except FileNotFoundError:
        print(f"Error: Cranfield file not found at {filepath}", file=sys.stderr)
        sys.exit(1)
        
    return documents

def main():
    stopwords_file = "stopwords.txt"
    dataset_file = "cran.all.1400"
    output_file = "search-o-SAURS_processed.all"
    
    # Load resources
    stopwords = load_stopwords(stopwords_file)
    documents = parse_cranfield(dataset_file)
    
    print(f"Loaded {len(stopwords)} stopwords.")
    print(f"Loaded {len(documents)} documents.")
    
    # Process documents and write output
    with open(output_file, 'w') as out:
        for doc in documents:
            doc_id = doc['id']
            raw_text = doc['text']
            
            # 1. Tokenization
            tokens = tokenize(raw_text)
            # 2. Normalization
            normalized = normalize(tokens)
            # 3. Stop Word Removal
            no_stopwords = remove_stopwords(normalized, stopwords)
            # 4. Stemming
            stemmed = stem(no_stopwords)
            
            # Write document header
            out.write(f".I {doc_id}\n")
            # Write preprocessed tokens under .S tag
            out.write(".S\n")
            out.write(" ".join(stemmed) + "\n")
            
    print(f"Preprocessing completed. Output written to {output_file}")

if __name__ == "__main__":
    main()
