"""
ReviewSense - Preprocessing Pipeline
====================================
Single shared text preprocessing module for both model training (`train.py`)
and web application inference (`app.py`).

Ensures identical text normalization between training and production.
"""

import re
from typing import Any
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Negation words that carry critical sentiment meaning and MUST be preserved.
# Removing these would reverse the polarity of reviews (e.g. "not good" -> "good").
NEGATION_WORDS = {
    "not", "no", "nor", "never", "neither", "none", "nothing", 
    "cannot", "cant", "wont", "dont", "didnt", "isnt", "arent", 
    "wasnt", "werent", "havent", "hasnt", "hadnt", "wouldnt", 
    "couldnt", "shouldnt", "mustn't", "needn't", "without"
}

# Scikit-learn default stop words minus critical negation words
# This avoids any external NLTK corpus downloads while strictly preserving sentiment.
CUSTOM_STOP_WORDS = set(ENGLISH_STOP_WORDS) - NEGATION_WORDS

# Regex pattern for HTML tags (IMDb reviews heavily feature <br />, <i>, etc.)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")

# Regex pattern to expand common n't contractions into 'not' before stripping punctuation
CONTRACTIONS = [
    (re.compile(r"\bcan['’]t\b", re.IGNORECASE), "can not"),
    (re.compile(r"\bwon['’]t\b", re.IGNORECASE), "will not"),
    (re.compile(r"\bshan['’]t\b", re.IGNORECASE), "shall not"),
    (re.compile(r"n['’]t\b", re.IGNORECASE), " not"),
]

# Regex pattern for non-alphanumeric characters (keeps letters, numbers, spaces)
NON_ALPHANUMERIC_PATTERN = re.compile(r"[^a-z0-9\s]")

# Regex pattern for excess whitespace
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: Any) -> str:
    """
    Cleans raw review text for sentiment classification.
    
    Processing Steps:
    1. Type safety: Converts input to string; returns empty string for None/non-strings.
    2. Lowercasing: Ensures case-insensitivity.
    3. HTML tag removal: Strips IMDb `<br />` tags and markup.
    4. Contraction handling: Expands `n't` to ` not` so negation polarity is preserved.
    5. Non-alphanumeric removal: Replaces punctuation and special symbols with spaces.
    6. Stop word removal: Filters out non-informative English stop words while
       strictly keeping negation words (e.g., 'not', 'no', 'never', 'nothing').
    7. Whitespace normalization: Trims leading/trailing and collapses multiple spaces.

    Parameters:
        text (Any): The raw review text.

    Returns:
        str: The normalized, cleaned review text.
    """
    # Step 1: Input validation and type casting
    if text is None:
        return ""
    text = str(text).strip()
    if not text:
        return ""

    # Step 2: Convert to lowercase
    text = text.lower()

    # Step 3: Strip HTML tags (e.g. <br />, <b>, <p>)
    text = HTML_TAG_PATTERN.sub(" ", text)

    # Step 4: Expand negative contractions so negation words are preserved
    for pattern, replacement in CONTRACTIONS:
        text = pattern.sub(replacement, text)

    # Step 5: Remove punctuation and special characters
    text = NON_ALPHANUMERIC_PATTERN.sub(" ", text)

    # Step 6: Tokenize and remove stop words (preserving negations)
    words = text.split()
    filtered_words = [w for w in words if w not in CUSTOM_STOP_WORDS and len(w) > 1 or w in {"no"}]

    # Step 7: Collapse redundant whitespace and trim
    cleaned_text = WHITESPACE_PATTERN.sub(" ", " ".join(filtered_words)).strip()

    return cleaned_text


if __name__ == "__main__":
    # Quick sanity check
    sample = "The movie was NOT good at all! <br /><br />Didn't like the ending; complete disaster."
    cleaned = clean_text(sample)
    print("Sample test:")
    print("Raw:    ", sample)
    print("Cleaned:", cleaned)
    assert "not" in cleaned, "Negation 'not' must be retained!"
    assert "not" in clean_text("Didn't enjoy it"), "Contraction didn't must expand to include not!"
    print("Sanity test passed successfully!")
