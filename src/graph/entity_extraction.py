"""
Task 2 — Entity Extraction (Phase 2: improved normalization/deduplication).
Extracts medical/scientific entities from texts using SciSpacy's en_core_sci_sm model.
SciSpacy/spaCy remains the required NLP approach (PDF requirement preserved).

Phase 2 improvements (same approach, better hygiene):
- internal whitespace collapsing ("insulin  resistance" == "insulin resistance")
- possessive "'s" stripping ("patient's" == "patient")
- curly-quote/bracket normalization before punctuation stripping
- filters: pure-numeric tokens and over-long fragments (likely mis-extractions)
- deterministic type resolution: first-seen label wins (was: last wins)
- single shared collection helper (extract_entities and batch_extract no
  longer duplicate the normalization loop)
- public dedupe_entities() helper reused across chunk- and graph-level dedup

Deliberately NOT merged: plurals, synonyms, acronyms ("tumor" vs "tumors"
stay distinct) to preserve useful scientific distinctions.
"""

import re
import string
from typing import Dict, List

from tqdm import tqdm

try:
    import spacy
    import scispacy
except ImportError as e:
    raise ImportError(
        "Missing scispacy or spacy packages. Make sure to run:\n"
        "pip install spacy scispacy"
    ) from e

# Load model, raising clean instructions if the model weights themselves are not present.
try:
    nlp = spacy.load("en_core_sci_sm")
except OSError as e:
    raise OSError(
        "The scispacy model 'en_core_sci_sm' is not installed. Please install the version of "
        "en_core_sci_sm that matches your installed SciSpaCy and spaCy releases (e.g. from "
        "the official SciSpaCy repository or by running: "
        "pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v<VERSION>/en_core_sci_sm-<VERSION>.tar.gz "
        "replacing <VERSION> with your installed scispacy version)."
    ) from e


# Over-long spans are almost always sentence-fragment mis-extractions.
MAX_ENTITY_CHARS = 150

# Curly quotes normalized to ASCII before stripping surrounding punctuation.
_QUOTE_TABLE = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})


def normalize_entity_name(name: str) -> str:
    """Normalize a raw entity surface form to its canonical graph key.

    Steps: strip, lowercase, collapse internal whitespace, strip possessive
    "'s", normalize curly quotes, strip surrounding punctuation/brackets.
    Returns "" when nothing meaningful remains.
    """
    text = name.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"'s\b", "", text).strip()
    text = text.translate(_QUOTE_TABLE)
    # Strip any leading/trailing punctuation characters (e.g. brackets, commas, periods)
    text = text.strip(string.punctuation).strip()
    return re.sub(r"\s+", " ", text)


def is_noise_entity(name: str) -> bool:
    """True for normalized names that should never become graph nodes."""
    if not name or len(name) < 2 or len(name) > MAX_ENTITY_CHARS:
        return True
    # Pure numbers ("22", "1983") carry no entity identity; alphanumeric
    # scientific terms ("il-2", "covid-19") are kept.
    if not any(ch.isalpha() for ch in name):
        return True
    return False


def _collect_from_doc(doc) -> Dict[str, str]:
    """Collect {normalized_name: type} from a parsed spaCy doc.

    First-seen label wins on type conflicts, keeping results deterministic
    regardless of batching or document order.
    """
    collected: Dict[str, str] = {}
    for ent in doc.ents:
        norm_name = normalize_entity_name(ent.text)
        if is_noise_entity(norm_name):
            continue
        ent_type = ent.label_ if ent.label_ else "ENTITY"
        collected.setdefault(norm_name, ent_type)
    return collected


def dedupe_entities(entities: List[dict]) -> List[dict]:
    """Deduplicate ``[{"name": ..., "type": ...}]`` by normalized name.

    First occurrence (name and type) wins; order is preserved.
    """
    seen: Dict[str, dict] = {}
    for ent in entities:
        name = normalize_entity_name(ent.get("name", ""))
        if is_noise_entity(name) or name in seen:
            continue
        seen[name] = {"name": name, "type": ent.get("type") or "ENTITY"}
    return list(seen.values())


def extract_entities(text: str) -> List[dict]:
    """
    Extract and deduplicate entities from a single text block.
    Returns a list of dicts with keys: 'name', 'type'.
    """
    doc = nlp(text)
    return [{"name": name, "type": etype} for name, etype in _collect_from_doc(doc).items()]


def batch_extract(chunks: List[dict], batch_size: int = 128) -> Dict[str, List[dict]]:
    """
    Batch extract entities from a list of chunk dicts using spacy nlp.pipe.
    Returns mapping of chunk_id -> List of deduplicated entity dicts.
    """
    texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]
    results: Dict[str, List[dict]] = {}

    # Use nlp.pipe for efficient batch parsing
    print(f"[entities] Processing {len(chunks)} chunks with SciSpacy ...")
    for doc, chunk_id in zip(
        tqdm(nlp.pipe(texts, batch_size=batch_size), total=len(texts), desc="Extracting entities"),
        chunk_ids
    ):
        collected = _collect_from_doc(doc)
        results[chunk_id] = [
            {"name": name, "type": etype} for name, etype in collected.items()
        ]

    return results
