import re
import os
import sys
from porter import PorterStemmer

def tokenize(text):
    """
    Tokenization: Convert a document text into a list of meaningful token strings.

    This is a multi-stage tokenizer that makes explicit decisons for edge cases
    commonly found in scientific text (Cranfield aeronautics corpus):

    Stage 1: Normalize whitespace and pre-clean the raw text.
    Stage 2: Handle abrevations with periods (e.g., "u.s.a." -> "usa")
    Stage 3: Handle possessives/contractions (e.g., "prandtl's" -> "prandtl")
    Stage 4: Handle hyphenated compounds (e.g., "high-speed" -> "high", "speed")
    Stage 5: Handle slash-separated terms (e.g., "lift/drag" -> "lift", "drag")
    Stage 6: Split numbers glued to words (e.g., "10degree" -> "10", "degree")
    Stage 7: Extract final tokens (alphanumeric sequences)
    Stage 8: Filter noise (single chars, pure numerics)

    Returns:
        list[str]: A list of cleaned, meaningful token strings.
    """

    # Normalize whitespace before applying token-specific rules.
    text = re.sub(r'\s+', ' ', text).strip()

    # Collapse dotted abbreviations into searchable tokens.
    text = re.sub(
        r'\b([a-zA-Z]\.){2,}',
        lambda m: m.group(0).replace('.', ''),
        text
    )

    # Treat possessive and remaining apostrophes as token boundaries.
    text = re.sub(r"'s\b", '', text)
    text = re.sub(r"s'\b", 's', text)
    text = text.replace("'", ' ')

    # Split compound terms and slash-separated terms into individual tokens.
    text = text.replace('-', ' ')

    text = text.replace('/', ' ')

    # Separate adjacent letters and digits before extracting tokens.
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)

    # Extract alphanumeric tokens, then discard one-character and numeric noise.
    tokens = re.findall(r'[a-zA-Z0-9]+', text)

    tokens = [t for t in tokens if len(t) > 1 and not t.isdigit()]

    return tokens

def normalize(tokens):
    """
    Normalization: Transform tokens into canonical forms to ensure equivelent
    terms map to the same representetion, maximizing recall.

    This is a multi-stage normalizer that makes explicit decisons based on
    analysis of the Cranfield aeronautics corpus:

    Stage 1: Case folding (e.g., "Aerodynamic" -> "aerodynamic")
    Stage 2: British -> American spelling equivalence (e.g., "behaviour" -> "behavior")
    Stage 3: Minimum length filter (remove tokens with len < 2)

    Returns:
        list[str]: Normalized token list.
    """

    normalized = []
    for token in tokens:

        # Apply case folding before spelling normalization.
        token = token.lower()

        # Keep exceptions from being incorrectly converted by each suffix rule.
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

        # Convert only known safe British -re endings.
        RE_SAFE_ENDINGS = ('tre', 'bre')

        if any(token.endswith(ending) for ending in RE_SAFE_ENDINGS):
            if len(token) > 3:
                token = token[:-2] + 'er'

        OGUE_EXCEPTIONS = {'vogue', 'rogue', 'brogue'}

        if token not in OGUE_EXCEPTIONS and token.endswith('ogue'):
            token = token[:-4] + 'og'

        if token.endswith('mme') and len(token) > 4:
            token = token[:-2]

        if len(token) >= 2:
            normalized.append(token)

    return normalized

def remove_stopwords(tokens, stopwords):
    """
    Stop word removal: Filter out tokens that are in the stopwords set.
    """
    return [token for token in tokens if token not in stopwords]

def stem(tokens):
    """
    Stemming: Convert tokens to their stems using the Porter stemmer.
    """
    # Reuse one stemmer instance for the current token sequence.
    stemmer = PorterStemmer()
    stemmed = []
    for token in tokens:

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
                # Collect title and abstract text for each Cranfield record.
                if line_str.startswith('.I'):

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

            if current_doc is not None:
                combined_text = " ".join(title_buffer) + " " + " ".join(abstract_buffer)
                current_doc['text'] = combined_text
                documents.append(current_doc)
    except FileNotFoundError:
        print(f"Error: Cranfield file not found at {filepath}", file=sys.stderr)
        sys.exit(1)

    return documents

def deduplicate(tokens):
    """
    Token deduplication: Remove duplicate tokens within a singel document.

    NOTE: In the Boolean retrieval model, only the presence of a term
        in a document matters, not its frequency. If the system is later extended to TF-IDF or ranked retrieval,
        deduplication would need to be removed since term frequency matters there.

    Returns:
        list[str]: Unique tokens in first-occurrence order.
    """
    # Preserve first-occurrence order while removing duplicate tokens.
    seen = set()
    unique = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            unique.append(token)
    return unique

def main():
    stopwords_file = os.path.join("data", "stopwords.txt")
    dataset_file = os.path.join("data", "cran.all.1400")
    output_file = os.path.join("output", "search-o-SAURS_processed.all")

    # Load input resources before processing the corpus.
    stopwords = load_stopwords(stopwords_file)
    documents = parse_cranfield(dataset_file)

    print(f"Loaded {len(stopwords)} stopwords.")
    print(f"Loaded {len(documents)} documents.")

    total_before = 0
    total_after = 0

    # Apply the document pipeline and write the assignment output format.
    with open(output_file, 'w') as out:
        for doc in documents:
            doc_id = doc['id']
            raw_text = doc['text']

            # Process each document in the required retrieval order.
            tokens = tokenize(raw_text)

            normalized = normalize(tokens)

            no_stopwords = remove_stopwords(normalized, stopwords)

            stemmed = stem(no_stopwords)

            unique = deduplicate(stemmed)

            total_before += len(stemmed)
            total_after += len(unique)

            out.write(f".I {doc_id}\n")

            out.write(".S\n")
            out.write(" ".join(unique) + "\n")

    dedup_pct = (1 - total_after / total_before) * 100 if total_before > 0 else 0
    print(f"Deduplication: {total_before} tokens -> {total_after} unique ({dedup_pct:.1f}% reduction)")
    print(f"Preprocessing completed. Output written to {output_file}")

if __name__ == "__main__":
    main()
