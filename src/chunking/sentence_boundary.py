"""
Task 2b — Sentence-boundary chunking.
Split into sentences with spaCy's sentencizer (rule-based, no trained model needed).
Greedily pack consecutive sentences into chunks until the next sentence would exceed
the 100-token budget, then start a new chunk. Never split a sentence across chunks.
"""

from typing import List

import spacy
import tiktoken

from src.config import settings

# Lightweight blank English pipeline with only the rule-based sentencizer
_nlp = spacy.blank("en")
_nlp.add_pipe("sentencizer")

ENCODING = tiktoken.get_encoding("cl100k_base")
TOKEN_LIMIT = settings.CHUNK_TOKEN_LIMIT  # default 100


def _token_count(text: str) -> int:
    return len(ENCODING.encode(text))


def chunk(article_id: str, text: str) -> List[dict]:
    """
    Greedily merge consecutive sentences up to TOKEN_LIMIT tokens per chunk.
    """
    doc = _nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    if not sentences:
        return []

    chunks: List[dict] = []
    current_sentences: List[str] = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = _token_count(sent)

        if current_sentences and current_tokens + sent_tokens > TOKEN_LIMIT:
            # Flush the accumulated chunk
            chunks.append({
                "article_id": article_id,
                "chunk_id": f"{article_id}_chunk_{len(chunks)}",
                "text": " ".join(current_sentences),
                "strategy": "sentence_boundary",
            })
            current_sentences = []
            current_tokens = 0

        current_sentences.append(sent)
        current_tokens += sent_tokens

    # Flush remaining sentences
    if current_sentences:
        chunks.append({
            "article_id": article_id,
            "chunk_id": f"{article_id}_chunk_{len(chunks)}",
            "text": " ".join(current_sentences),
            "strategy": "sentence_boundary",
        })

    return chunks
