# GraphRAG `app` Knowledge Bytes

This folder mirrors the relevant structure of the uploaded `app/` source:

```text
app_kb/
├── __init__.md
├── main.md
├── components/
│   ├── answer_card.md
│   ├── evidence_panel.md
│   ├── graph_view.md
│   ├── metrics_panel.md
│   └── search_bar.md
└── styles/
    └── theme.md
```

Python `__pycache__` files and `.gitkeep` placeholders are not represented because they contain no application logic relevant to the Knowledge Bytes.

The main architectural split is:
- `main.py`: Streamlit application orchestration
- `components/`: reusable UI components
- `styles/`: visual design system
- `__init__.py`: package marker
