# Byte 7: Biomedical Entity Extraction with SciSpacy

**Builds on:** Byte 6

## In plain terms

SciSpacy is used for scientific/biomedical named-entity extraction. Extracted concepts are represented as Entity nodes in Neo4j.

## The code

```text
import spacy

nlp = spacy.load("en_core_sci_sm")
doc = nlp(chunk_text)

for ent in doc.ents:
    name = ent.text
    label = ent.label_
```

## What's happening

The model processes each chunk and exposes recognized entities through `doc.ents`. The graph builder turns those entities into graph nodes and connects the source chunk using MENTIONS.

## Why it matters

Entity extraction is what gives the graph useful semantic landmarks. Graph expansion can then answer not only 'which chunk is similar?' but also 'which other chunks mention the same scientific concept?'
