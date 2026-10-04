# Knowledge Bytes — `app/styles/theme.css`

---

### Byte 1: The Theme Defines a Shared Design System
**Builds on:** `main.md` and the UI components

**In plain terms:**
`theme.css` centralizes colors, typography, spacing, borders, and component styling for the Streamlit application. Components use class names such as `answer-hero`, `metric-card`, and `graph-frame` rather than repeating all styling logic.

**The code:**
```css
:root {
    --color-bg: #FAFAF7;
    --color-surface: #FFFFFF;
    --color-border: #E2DFD6;
    --color-primary: #0F5257;
    --color-accent: #FF6B4A;
}
```

The variables make the visual system consistent across the application.

---

### Byte 2: The Main Content Area Is Widened
**Builds on:** Byte 1

**In plain terms:**
The CSS overrides Streamlit's default central-column width so the literature interface has enough horizontal space for evidence and graph content.

**The code:**
```css
div[data-testid="stMainBlockContainer"].block-container {
    max-width: 110rem !important;
    width: calc(100% - 6rem) !important;
}
```

The sidebar is intentionally styled separately.

---

### Byte 3: Buttons and Inputs Share the Theme
**Builds on:** Byte 1

**In plain terms:**
Buttons, text inputs, checkboxes, and sliders are restyled to match the application's colors and spacing rather than Streamlit defaults.

**The code:**
```css
.stButton > button {
    background: var(--color-accent) !important;
    color: white !important;
    border-radius: 8px !important;
}

.stTextInput > div > div > input {
    background: var(--color-surface) !important;
    border: 1px solid var(--color-border) !important;
}
```

This keeps the search interface visually consistent.

---

### Byte 4: The Answer Component Depends on a Named CSS Class
**Builds on:** Byte 1 and `answer_card.md`

**In plain terms:**
The answer card uses `.answer-hero` for its prominent presentation. The CSS controls its border, padding, typography, margin, and shadow.

**The code:**
```css
.answer-hero {
    background: var(--color-surface) !important;
    border: 1px solid var(--color-border) !important;
    border-left: 4px solid var(--color-primary) !important;
    border-radius: 12px !important;
    padding: 2rem 2.25rem !important;
}
```

The Python component therefore controls content while CSS controls appearance.

---

### Byte 5: Metric Cards Have a Dedicated Style
**Builds on:** Byte 1 and `metrics_panel.md`

**In plain terms:**
Evaluation metrics are presented as compact cards. The CSS defines the value, label, and delta styling.

**The code:**
```css
.metric-card {
    background: var(--color-surface) !important;
    border: 1px solid var(--color-border) !important;
    border-radius: 8px !important;
}

.metric-card .metric-value {
    color: var(--color-primary) !important;
}
```

This keeps evaluation information visually distinct from the main answer.

---

### Byte 6: Status Badges Communicate Backend Availability
**Builds on:** Byte 1

**In plain terms:**
The header uses badges for Neo4j, vector search, graph retrieval, and LLM. `.status-strip`, `.status-badge`, and `.status-dot` define that presentation.

**The code:**
```css
.status-badge {
    display: inline-flex !important;
    align-items: center !important;
    border: 1px solid var(--color-border) !important;
    border-radius: 999px !important;
}

.status-dot {
    width: 8px !important;
    height: 8px !important;
    border-radius: 50% !important;
    background: var(--color-success) !important;
}
```

These are presentation indicators; they are not the actual connection checks.

---

### Byte 7: Tabs and Graph Containers Get Consistent Styling
**Builds on:** Bytes 1 and 4

**In plain terms:**
The answer/evidence/graph tabs and graph frame receive common spacing, borders, and colors. This makes the different result views feel like parts of one application.

**The code:**
```css
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0 !important;
    padding: 0.6rem 1.25rem !important;
}

.graph-frame {
    background: var(--color-surface) !important;
    border: 1px solid var(--color-border) !important;
    border-radius: 12px !important;
}
```

---

### PUTTING IT TOGETHER

`theme.css` is the presentation layer for the Streamlit app. It establishes a shared visual vocabulary and then gives major UI components dedicated classes. `main.py` loads this stylesheet at startup, while the individual components provide the content that those styles decorate.
