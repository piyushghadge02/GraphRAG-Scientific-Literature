# Byte 20: Git and GitHub Workflow

**Builds on:** Byte 18

## In plain terms

Git tracks the project history and GitHub stores the repository. The safe workflow used for this project is to inspect changes, stage only intended files, verify the staged diff, commit, then push `main`.

## The code

```text
git status
git diff --cached --check
git diff --cached --name-only
git commit -m "Update Phase 4 evaluation corpus and GraphRAG pipeline"
git push origin main
```

## What's happening

The repository is `piyushghadge02/GraphRAG-Scientific-Literature`. Secrets such as `.env` must never be staged. `git push` only sends commits; staged but uncommitted changes will not be pushed.

## Why it matters

This prevents accidental publication of credentials or unrelated files and keeps the assignment implementation reproducible.
