---
### Byte 1: Root entry point
**Builds on:** None — starting point

**In plain terms:**
`app.py` is intentionally a very small root-level entry point for the GraphRAG Streamlit application. Its job is to expose a simple launch target while keeping the actual application logic inside `app/main.py`.

**The code:**
```python
"""Root entry point: `streamlit run app.py` (delegates to app/main.py)."""
```

The comment documents the intended command and makes the architectural boundary clear: this file is only the launcher.

---
### Byte 2: Delegate to the application module
**Builds on:** Byte 1

**In plain terms:**
The file imports the `main` function from `app.main`, which is where the real application entry logic lives. This keeps the root launcher thin and prevents application code from being duplicated here.

**The code:**
```python
from app.main import main
```

The import creates a direct connection from the root-level launch file to the application's main module.

---
### Byte 3: Run the application only when executed directly
**Builds on:** Byte 2

**In plain terms:**
The `__name__` check ensures `main()` runs when `app.py` is executed as the entry point, but not merely when the file is imported by another module. This is the standard Python entry-point pattern.

**The code:**
```python
if __name__ == "__main__":
    main()
```

That distinction makes `app.py` safe to import while still allowing `streamlit run app.py` to trigger the application.

PUTTING IT TOGETHER

`app.py` is a deliberately thin launcher for the GraphRAG UI. It documents the expected Streamlit command, imports the application's `main()` function, and invokes it only when this file is the execution entry point. The actual UI and application behavior therefore remain in `app/main.py`, while the root file provides a clean launch boundary.
