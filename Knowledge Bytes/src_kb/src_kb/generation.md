# Knowledge Bytes — `src/generation/`

---

### Byte 1: Prompt Construction Defines the Evidence Boundary
**Builds on:** `retrieval.md`

**In plain terms:**
`prompt_templates.py` converts retrieved chunks into numbered context passages. The prompt tells the LLM to answer from those passages rather than treating the model's general knowledge as the primary evidence source.

**The code:**
```python
for i, chunk in enumerate(chunks, start=1):
    context_lines.append(
        f"[{i}] (from {chunk_id}):\n{chunk_text}\n"
    )
```

The chunk ID is included so the answer can refer back to retrieved evidence.

---

### Byte 2: The RAG Prompt Requests Explicit Citations
**Builds on:** Byte 1

**In plain terms:**
The template asks the model to support claims with the numbered retrieved passages. It also gives the model an explicit fallback when the supplied context is insufficient.

**The code:**
```text
Use only the supplied context.
Cite relevant context passages.
If the context is insufficient, say so.
```

This is a prompt-level guard against unsupported answers.

---

### Byte 3: `LLMClient` Hides Provider Differences
**Builds on:** Byte 1

**In plain terms:**
The generation code exposes one `generate()` interface while supporting different backends such as Groq, OpenAI, and Ollama.

**The code:**
```python
if backend_lower == "groq":
    ...
elif backend_lower == "openai":
    ...
elif backend_lower == "ollama":
    ...
else:
    raise ValueError(...)
```

Retrieval code therefore does not need provider-specific logic.

---

### Byte 4: Generation Configuration Controls Model Behavior
**Builds on:** Byte 3

**In plain terms:**
The client receives backend, model, temperature, and token settings from configuration. The LLM call is therefore a runtime choice rather than a hard-coded implementation detail.

**The code:**
```python
response = client.chat.completions.create(
    model=self.model,
    messages=messages,
    temperature=self.temperature,
    max_tokens=self.max_tokens,
)
```

The exact API wrapper differs by backend, but the surrounding interface stays consistent.

---

### Byte 5: Groq Rate Limits Rotate API Keys
**Builds on:** Byte 3

**In plain terms:**
The Groq client can hold multiple API keys. A recognized HTTP 429 rate-limit response advances to another key and retries.

**The code:**
```python
if is_rate_limit:
    self._current_key_index = (
        self._current_key_index + 1
    ) % len(keys)
    attempts += 1
```

Non-rate-limit failures are not blindly retried.

---

### Byte 6: Query Decomposition Uses the Same LLM Interface
**Builds on:** Bytes 3–5

**In plain terms:**
Complex questions can be sent through `LLMClient` to produce a JSON list of focused sub-questions. This keeps decomposition independent from the retrieval implementation.

**The code:**
```python
response = llm_client.generate(
    prompt,
    system_prompt=system_prompt
)
sub_queries = json.loads(response)
```

If the LLM response is invalid, retrieval falls back to the original question.

---

### PUTTING IT TOGETHER

The generation package sits after retrieval. It receives ranked chunks, formats them as explicit evidence, and sends that context through a provider-independent LLM client. The same client can also support query decomposition before retrieval, while rate-limit handling stays isolated inside the provider layer.
