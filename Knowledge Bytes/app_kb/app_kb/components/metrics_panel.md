# Knowledge Bytes — `app/components/metrics_panel.py`

---

### Byte 1: The Metrics Panel Reads Evaluation Results
**Builds on:** `evaluation.md` in `src`

**In plain terms:**
`render_metrics_panel()` reads `data/processed/eval_comparison.csv` and turns stored evaluation results into UI cards. The component does not calculate the underlying retrieval experiment.

**The code:**
```python
csv_path = "data/processed/eval_comparison.csv"

if not os.path.exists(csv_path):
    st.info("Evaluation data not available yet.")
    return
```

The file is treated as an optional application input.

---

### Byte 2: CSV Rows Become a Metric Dictionary
**Builds on:** Byte 1

**In plain terms:**
Each CSV row contains a metric name and values for baseline, full system, and delta. The component converts those rows into a dictionary for easier rendering.

**The code:**
```python
metrics_dict[metric_name] = {
    "baseline": row["Baseline"],
    "full": row["Full_System"],
    "delta": row["Delta"],
}
```

This separates CSV parsing from UI formatting.

---

### Byte 3: Retrieval Metrics Are Rendered Separately
**Builds on:** Byte 2

**In plain terms:**
The panel recognizes Recall@5, Recall@10, and MRR as retrieval metrics. When numeric values are available, it calculates the displayed difference between the full system and baseline.

**The code:**
```python
delta = full_val - baseline_val

if delta > 0:
    arrow = "↑"
elif delta < 0:
    arrow = "↓"
else:
    arrow = "="
```

The UI presents the measured difference rather than generating an interpretation beyond the numbers.

---

### Byte 4: Generation Metrics Are Displayed Separately
**Builds on:** Byte 2

**In plain terms:**
ROUGE-L F1 and BERTScore F1 are shown under a separate Generation section. The panel displays the full-system values when available.

**The code:**
```python
for metric_name in [
    "ROUGE-L F1",
    "BERTScore F1"
]:
    ...
```

This keeps retrieval quality and generated-answer quality conceptually separate.

---

### Byte 5: Missing or Malformed Evaluation Data Fails Gracefully
**Builds on:** Bytes 1–4

**In plain terms:**
Missing files, empty CSVs, or runtime parsing errors do not crash the application. The panel instead tells the user that evaluation data is unavailable.

**The code:**
```python
except Exception as e:
    st.info("Evaluation data not available yet.")
    print(traceback.format_exc())
```

This component is optional to the main Q&A workflow.

---

### PUTTING IT TOGETHER

The metrics panel is a read-only visualization of previously generated evaluation results. It parses the comparison CSV, separates retrieval and generation metrics, and displays numeric differences where applicable. It does not run evaluation or alter the GraphRAG pipeline.
