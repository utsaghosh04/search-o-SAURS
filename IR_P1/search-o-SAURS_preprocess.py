import re
import os
import sys
from porter import PorterStemmer

def tokenize(text):
    """
    Tokenization: Convert a document text into a list of meaningful token strings.

    This is a multi-stage tokenizer that makes explicit decisions for edge cases
    commonly found in scientific text (Cranfield aeronautics corpus):

    Stage 1: Normalize whitespace and pre-clean the raw text.
    Stage 2: Handle abbreviations with periods  (e.g., "u.s.a." -> "usa")
    Stage 3: Handle possessives/contractions     (e.g., "prandtl's" -> "prandtl")
    Stage 4: Handle hyphenated compounds         (e.g., "high-speed" -> "high", "speed")
    Stage 5: Handle slash-separated terms        (e.g., "lift/drag" -> "lift", "drag")
    Stage 6: Split numbers glued to words        (e.g., "10degree" -> "10", "degree")
    Stage 7: Extract final tokens                (alphanumeric sequences)
    Stage 8: Filter noise                        (single chars, pure numerics)

    Returns:
        list[str]: A list of cleaned, meaningful token strings.
    """

    # -------------------------------------------------------------------------
    # STAGE 1: Whitespace normalization
    # Collapse all whitespace (tabs, newlines, multiple spaces) into single spaces.
    # -------------------------------------------------------------------------
    text = re.sub(r'\s+', ' ', text).strip()

    # -------------------------------------------------------------------------
    # STAGE 2: Abbreviations with periods
    # Scientific text contains abbreviations like "u.s.a.", "a.i.a.a.", "e.g."
    # DECISION: Collapse them into a single token by removing the periods.
    #   "u.s.a." -> "usa",  "a.i.a.a." -> "aiaa",  "e.g." -> "eg"
    # RATIONALE: Users searching for "NASA" or "AIAA" expect to find these;
    #   splitting on "." would create meaningless single-char tokens.
    # -------------------------------------------------------------------------
    text = re.sub(
        r'\b([a-zA-Z]\.){2,}',
        lambda m: m.group(0).replace('.', ''),
        text
    )

    # -------------------------------------------------------------------------
    # STAGE 3: Possessives and contractions
    # Cranfield has many possessives: "prandtl's", "blasius's", "author's"
    # DECISION: Strip the possessive suffix "'s" and trailing apostrophes.
    #   "prandtl's" -> "prandtl",  "authors'" -> "authors"
    # RATIONALE: The possessive marker carries no retrieval value. A user
    #   searching for "prandtl" should match "prandtl's theory".
    # Also handle French-origin terms like "d'etudes" -> "d", "etudes"
    # (the apostrophe acts as a separator here, handled by later splitting).
    # -------------------------------------------------------------------------
    text = re.sub(r"'s\b", '', text)       # Remove possessive 's
    text = re.sub(r"s'\b", 's', text)      # "authors'" -> "authors"
    text = text.replace("'", ' ')          # Remaining apostrophes become separators
                                           # e.g., "d'etudes" -> "d etudes"

    # -------------------------------------------------------------------------
    # STAGE 4: Hyphenated compound words
    # Very common in scientific text: "two-dimensional", "high-speed",
    # "boundary-layer", "10-foot", "10-percent-thick"
    # DECISION: Split hyphenated words into their components.
    #   "high-speed" -> "high", "speed"
    #   "two-dimensional" -> "two", "dimensional"
    #   "10-foot" -> "10", "foot"
    # RATIONALE: Users may search for "speed" or "dimensional" independently.
    #   Keeping "high-speed" as one token makes it unsearchable by either part.
    #   Splitting maximizes recall — a core IR objective.
    # -------------------------------------------------------------------------
    text = text.replace('-', ' ')

    # -------------------------------------------------------------------------
    # STAGE 5: Slash-separated terms
    # Found in Cranfield: "lift/drag", "aero/space", "and/or", "btu/lb"
    # DECISION: Split on slashes — each side becomes a separate token.
    #   "lift/drag" -> "lift", "drag"
    # RATIONALE: Slashes connect related but distinct concepts. A search for
    #   "drag" should match documents discussing "lift/drag ratio".
    # -------------------------------------------------------------------------
    text = text.replace('/', ' ')

    # -------------------------------------------------------------------------
    # STAGE 6: Split numbers glued to words
    # Cranfield has: "10degree", "000degreek", "100degrees", "10g", "18in"
    # DECISION: Insert a space at digit-to-alpha and alpha-to-digit boundaries.
    #   "10degree" -> "10 degree",  "000degreek" -> "000 degreek",  "re10" -> "re 10"
    # RATIONALE: "10degree" is not a meaningful term. The word "degree" carries
    #   retrieval value while "10" is a numeric qualifier. Splitting them allows
    #   "degree" to be properly stemmed and indexed.
    # EXCEPTION: Ordinals like "10th", "2nd", "1st", "3rd" — the suffix is
    #   not useful after splitting, so they'll become "10" + "th" and "th"
    #   will be filtered as a short token later. This is acceptable.
    # -------------------------------------------------------------------------
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)

    # -------------------------------------------------------------------------
    # STAGE 7: Extract final tokens
    # After all pre-cleaning, extract contiguous alphanumeric sequences.
    # At this point hyphens, slashes, apostrophes, and other punctuation have
    # already been handled. This captures what remains.
    # -------------------------------------------------------------------------
    tokens = re.findall(r'[a-zA-Z0-9]+', text)

    # -------------------------------------------------------------------------
    # STAGE 8: Filter noise tokens
    # DECISION: Remove tokens that are:
    #   (a) Single characters — these are mostly noise from parenthetical labels
    #       like "(a)", "(b)", "(c)" or stray initials. The Cranfield corpus has
    #       ~19,000 single-char occurrences, almost all noise.
    #       Exception: In general IR, single chars like "C" (programming language)
    #       could matter, but in this aeronautics corpus they don't.
    #   (b) Pure numeric tokens — numbers like "0", "100", "1400" carry no
    #       standalone retrieval value in text search. A user won't search for
    #       the number "100" — they'd search for "100 degree" or "mach 2".
    # -------------------------------------------------------------------------
    tokens = [t for t in tokens if len(t) > 1 and not t.isdigit()]

    return tokens

def normalize(tokens):
    """
    Normalization: Transform tokens into canonical forms to ensure equivalent
    terms map to the same representation, maximizing recall.

    This is a multi-stage normalizer that makes explicit decisions based on
    analysis of the Cranfield aeronautics corpus:

    Stage 1: Case folding            (e.g., "Aerodynamic" -> "aerodynamic")
    Stage 2: British → American      (e.g., "behaviour" -> "behavior")
             spelling equivalence
    Stage 3: Minimum length filter   (remove tokens with len < 2)

    WHY NORMALIZATION MATTERS (evidence from Cranfield corpus):
    ─────────────────────────────────────────────────────────────
    Without spelling normalization, the Porter stemmer produces DIFFERENT
    stems for British vs American variants. Empirically verified:

      behaviour  → stem "behaviour"   vs  behavior  → stem "behavior"    (DIVERGE)
      linearised → stem "linearis"    vs  linearized → stem "linear"     (DIVERGE)
      centre     → stem "centr"       vs  center     → stem "center"     (DIVERGE)
      vapour     → stem "vapour"      vs  vapor      → stem "vapor"      (DIVERGE)
      analysed   → stem "analys"      vs  analyzed   → stem "analyz"     (DIVERGE)

    A query for "behavior" would MISS 46 documents using "behaviour".
    Normalization before stemming fixes this by mapping all variants to
    one canonical form.

    Returns:
        list[str]: Normalized token list.
    """

    normalized = []
    for token in tokens:

        # -----------------------------------------------------------------
        # STAGE 1: Case folding
        # DECISION: Convert all tokens to lowercase.
        # RATIONALE: "Aerodynamic", "AERODYNAMIC", and "aerodynamic" are the
        #   same term. Case carries no semantic distinction in this corpus.
        #   Must happen FIRST so that all subsequent rules operate on
        #   consistent lowercase input.
        # -----------------------------------------------------------------
        token = token.lower()

        # -----------------------------------------------------------------
        # STAGE 2: British → American spelling equivalence
        # DECISION: Map British English spellings to their American equivalents.
        # RATIONALE: The Cranfield corpus is a mix of British and American
        #   authored papers. Both spellings coexist for the same concepts:
        #     behaviour (46 occurrences) vs behavior (75 occurrences)
        #     centre    (14 occurrences) vs center   (64 occurrences)
        #     linearised(25 occurrences) vs linearized(70 occurrences)
        #   Without this normalization, the Porter stemmer produces different
        #   stems, fragmenting the index and reducing recall.
        #
        # We apply rule-based transformations covering the four major
        # British/American divergence patterns:
        #
        #   Pattern A: -ise/-ised/-ising/-isation → -ize/-ized/-izing/-ization
        #   Pattern B: -our → -or  (behaviour → behavior)
        #   Pattern C: -re → -er   (centre → center)
        #   Pattern D: -ogue → -og (analogue → analog)
        #
        # ORDER MATTERS: More specific suffixes are checked before shorter
        # ones to prevent partial matches (e.g., check "-isation" before "-ise").
        # -----------------------------------------------------------------

        # Pattern A: -ise variants → -ize variants
        # Handles: linearised→linearized, generalisation→generalization, etc.
        # GUARD: Exclude words where "-ise" is part of the root, not a suffix.
        #   e.g., "noise", "rise", "wise", "cruise", "promise", "otherwise",
        #         "exercise", "comprise", "surprise", "precise", "concise"
        #   These are NOT British spellings — they're base English words.
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

        # Pattern B: -our → -or
        # Handles: behaviour→behavior, vapour→vapor, colour→color, favour→favor
        # GUARD: Exclude words where "-our" is NOT a British suffix.
        #   e.g., "four", "pour", "your", "our", "contour", "detour", "tour"
        OUR_EXCEPTIONS = {
            'four', 'pour', 'your', 'our', 'contour', 'detour', 'tour',
            'glamour', 'velour', 'amour', 'dour', 'scour', 'hour',
        }

        if token not in OUR_EXCEPTIONS and token.endswith('our'):
            # Extra safety: only convert if the root + "or" makes sense
            # i.e., the token has at least 2 chars before "our"
            if len(token) > 4:
                token = token[:-3] + 'or'

        # Pattern C: -re → -er  (only for specific word endings)
        # Handles: centre→center, metre→meter, fibre→fiber, litre→liter
        # GUARD: Many English words legitimately end in "-re" and are NOT
        #   British spellings: "fire", "are", "were", "more", "before",
        #   "where", "there", "structure", "pressure", "ature", "ure", etc.
        # DECISION: Apply only to known safe endings: -tre, -bre
        #   These are the primary -re/-er divergences in scientific text.
        RE_SAFE_ENDINGS = ('tre', 'bre')

        if any(token.endswith(ending) for ending in RE_SAFE_ENDINGS):
            if len(token) > 3:
                token = token[:-2] + 'er'

        # Pattern D: -ogue → -og
        # Handles: analogue→analog, catalogue→catalog, dialogue→dialog
        # GUARD: "vogue" and "rogue" are not spelling variants.
        OGUE_EXCEPTIONS = {'vogue', 'rogue', 'brogue'}

        if token not in OGUE_EXCEPTIONS and token.endswith('ogue'):
            token = token[:-4] + 'og'

        # Pattern E: -mme → -m
        # Handles: programme→program, gramme→gram
        # These are British doubled-consonant endings.
        # GUARD: "flamme" etc. are not common English words in this corpus.
        if token.endswith('mme') and len(token) > 4:
            token = token[:-2]

        # -----------------------------------------------------------------
        # STAGE 3: Minimum length filter
        # DECISION: Remove tokens shorter than 2 characters.
        # RATIONALE: After all normalization transforms, some tokens may have
        #   become very short. Single-character tokens carry no retrieval
        #   value in this corpus (verified: ~19,000 single-char occurrences
        #   in Cranfield, almost all noise from tags/labels).
        # -----------------------------------------------------------------
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

def deduplicate(tokens):
    """
    Token deduplication: Remove duplicate tokens within a single document.

    DECISION: Keep only unique tokens, preserving their first-occurrence order.
    RATIONALE: In the Boolean retrieval model, only the PRESENCE of a term
        in a document matters, not its frequency. A word appearing 5 times
        is identical to appearing once for AND/OR queries.

        Benefits:
        - Reduces the processed file size (fewer tokens to write/read)
        - Produces cleaner postings lists in the index (no duplicate docid checks needed)
        - Speeds up index construction

        Trade-off: If the system is later extended to TF-IDF or ranked retrieval,
        deduplication would need to be removed since term frequency matters there.
        For this Boolean assignment, it is purely beneficial.

    Returns:
        list[str]: Unique tokens in first-occurrence order.
    """
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
    
    # Load resources
    stopwords = load_stopwords(stopwords_file)
    documents = parse_cranfield(dataset_file)
    
    print(f"Loaded {len(stopwords)} stopwords.")
    print(f"Loaded {len(documents)} documents.")
    
    total_before = 0
    total_after = 0
    
    # Process documents and write output
    with open(output_file, 'w') as out:
        for doc in documents:
            doc_id = doc['id']
            raw_text = doc['text']
            
            # ── Preprocessing Pipeline ──
            # Order: Tokenize → Normalize → Stop Word Removal → Stem → Deduplicate
            #
            # NOTE: The assignment lists the four operations as:
            #   i. Tokenization  ii. Stemming  iii. Stop word removal  iv. Normalization
            # Our execution order differs because of hard logical dependencies:
            #   - Normalization (case folding) MUST precede stemming: the Porter
            #     stemmer only recognizes lowercase vowels (a,e,i,o,u). Feeding it
            #     uppercase produces incorrect stems (e.g., stem("BOUNDARY")="BOUNDARY").
            #   - British→American normalization MUST precede stemming: otherwise
            #     "behaviour" stems to "behaviour" instead of unifying with "behavior".
            #   - Stop word removal before stemming: 7 stop words (are→ar, has→ha,
            #     this→thi, etc.) would survive removal after stemming because their
            #     stemmed forms don't match the stop word list.
            #
            # All four functions are implemented as separate, documented functions.

            # 1. Tokenization
            tokens = tokenize(raw_text)
            # 2. Normalization (case folding + British→American spelling)
            normalized = normalize(tokens)
            # 3. Stop Word Removal
            no_stopwords = remove_stopwords(normalized, stopwords)
            # 4. Stemming
            stemmed = stem(no_stopwords)
            # 5. Deduplication (Boolean retrieval: only presence matters)
            unique = deduplicate(stemmed)
            
            total_before += len(stemmed)
            total_after += len(unique)
            
            # Write document header
            out.write(f".I {doc_id}\n")
            # Write preprocessed tokens under .S tag
            out.write(".S\n")
            out.write(" ".join(unique) + "\n")
            
    dedup_pct = (1 - total_after / total_before) * 100 if total_before > 0 else 0
    print(f"Deduplication: {total_before} tokens → {total_after} unique ({dedup_pct:.1f}% reduction)")
    print(f"Preprocessing completed. Output written to {output_file}")

if __name__ == "__main__":
    main()
