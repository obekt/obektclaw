---
name: csv-to-database
description: Clean a CSV file and import it into a SQLite (or other) database
---

# Steps
1. Read the first ~20 lines with `read_file` to inspect headers, delimiter, and obvious type clues.
2. Sketch a target schema. Default to TEXT unless a column is clearly numeric or date-like.
3. Decide on a table name (snake_case from the filename unless the user specified one).
4. Check if the table already exists; if it does, ask before overwriting.
5. Use Python (`exec_python`) with the standard `csv` and `sqlite3` modules — no extra dependencies.
6. Validate row counts after import and report a 5-row sample back to the user.

# Watch out for
- BOM in the first column header.
- Mixed encodings — try utf-8 first, fall back to latin-1.
- Quoted commas inside fields (use `csv.reader`, not `str.split`).
- Empty strings vs NULL — coerce empty string → NULL on numeric columns.

# Improvement notes
This skill should be rewritten by the Learning Loop the first time it's used —
it's a starter template, not a final answer.
