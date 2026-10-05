"""
Task 2a — Fixed-token chunking (deliberately naive baseline).
Tokenize with tiktoken cl100k_base, slice into fixed 100-token windows
with no overlap and no sentence awareness.
"""

from typing import List

import tiktoken

from src.config import settings

ENCODING = tiktoken.get_encoding("cl100k_base")
TOKEN_LIMIT = settings.CHUNK_TOKEN_LIMIT  # default 100


def chunk(article_id: str, text: str, pmid: int = None) -> List[dict]:
    """
    Split *text* into fixed windows of TOKEN_LIMIT tokens.
    Returns a list of chunk dicts.
    """
    tokens = ENCODING.encode(text)
    chunks: List[dict] = []

    for i in range(0, len(tokens), TOKEN_LIMIT):
        window = tokens[i : i + TOKEN_LIMIT]
        chunk_text = ENCODING.decode(window)
        chunk_dict = {
            "article_id": article_id,
            "chunk_id": f"{article_id}_chunk_{len(chunks)}",
            "text": chunk_text,
            "strategy": "fixed_token",
        }
        if pmid is not None:
            chunk_dict["pmid"] = pmid
        chunks.append(chunk_dict)

    return chunks
