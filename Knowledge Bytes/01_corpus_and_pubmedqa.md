# Byte 2: Corpus and PubMedQA

**Builds on:** Byte 1

## In plain terms

The assignment uses about 5,000 PubMed abstracts. For Phase 4, PubMedQA questions must match articles in that corpus so retrieval can be evaluated against a known gold article. The current corpus preparation preserves real PubMed IDs for the PubMedQA-linked portion and fills the remaining corpus slots with additional abstracts.

## The code

```text
# Conceptual data record
{
  "article_id": "pmid_<PMID>",
  "title": "...",
  "abstract_text": "..."
}

# Evaluation item
{
  "question": "...",
  "pubid": "...",
  "long_answer": "..."
}
```

## What's happening

The data-loading logic was updated so the final corpus is exactly 5,000 abstracts, with 1,000 PubMedQA-linked PubMed abstracts fetched by PMID and 4,000 additional abstracts. A deterministic shuffle is used. The evaluation subset is then generated from matching PubMedQA records.

## Why it matters

Without a matching article ID, Phase 4 cannot fairly say whether the correct paper was retrieved. Matching by PubMed ID makes the gold article relationship explicit rather than relying on fuzzy full-text matching.
