# Knowledge Bytes — `app/__init__.py`

---

### Byte 1: The App Package Marker
**Builds on:** None — starting point

**In plain terms:**
`app/__init__.py` is empty. Its role is simply to mark `app` as a Python package so modules such as `app.components.search_bar` can be imported cleanly.

**The code:**
```python
# app/__init__.py
```

There is no application logic here. The important point is that package initialization has intentionally been kept empty.

---

### PUTTING IT TOGETHER

The `app` package contains the Streamlit UI rather than the core GraphRAG algorithms. This file establishes the package boundary, while `main.py`, `components/`, and `styles/` contain the actual application behavior.
