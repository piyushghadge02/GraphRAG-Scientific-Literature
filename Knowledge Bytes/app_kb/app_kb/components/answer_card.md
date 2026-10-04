# Knowledge Bytes — `app/components/answer_card.py`

---

### Byte 1: The Answer Card Owns Answer Presentation
**Builds on:** `main.md` Byte 17

**In plain terms:**
`render_answer_card()` is responsible only for displaying the generated answer. It does not generate, modify, or evaluate the answer itself.

**The code:**
```python
def render_answer_card(answer: str):
    st.markdown("### Answer")
    ...
```

Keeping this as a component lets `main.py` stay focused on application flow.

---

### Byte 2: Citation Numbers Are Highlighted
**Builds on:** Byte 1

**In plain terms:**
The answer may contain citation markers such as `[1]` and `[2]`. A regular expression finds those numbers and wraps them in styled HTML so they stand out visually.

**The code:**
```python
answer_html = re.sub(
    r'\[(\d+)\]',
    r'<span style="color: #FF6B4A; font-weight: 600;">[\1]</span>',
    answer
)
```

The underlying citation text is preserved; only its presentation changes.

---

### Byte 3: The Answer Is Rendered Inside a Hero Card
**Builds on:** Byte 2

**In plain terms:**
The processed answer is placed inside the CSS-defined `answer-hero` container. Streamlit renders the HTML so the result appears as a prominent answer block.

**The code:**
```python
st.markdown(
    f"""
    <div class="answer-hero">
        <p>{answer_html}</p>
    </div>
    """,
    unsafe_allow_html=True
)
```

The component therefore depends on the styling defined in `styles/theme.css`.

---

### PUTTING IT TOGETHER

The answer card has one responsibility: make the generated answer readable and visually emphasize citation markers. It receives an already-generated string from `main.py`, applies presentation-only transformation, and renders it using the shared CSS theme.
