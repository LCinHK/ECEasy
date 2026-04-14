# ECEasy FAISS Testing Suite

This directory contains diagnostic and visualization scripts for inspecting the FAISS index quality and retrieval behavior.

## Scripts

### 0. `test_server_split.py`
Smoke tests for the new `eceasy_server/` module split (llm/retrieval/streaming/app).

**What it checks:**
- FastAPI routes are still mounted (`/query`, `/`, `/frontpage`)
- LLM runtime config rejects invalid provider values
- LLM runtime config enforces API key rules for remote providers
- Streaming pipeline emits markers (`__LLM_RESPONSE__`, `__RELATED_QUESTIONS__`, `__SUGGESTED_IMAGES__`)

**Usage:**
```powershell
python test_server_split.py
```

**When to use:**
- After refactoring server modules
- Before committing server architecture changes

---

### 1. `inspect_faiss.py`
Inspect index statistics and metadata distribution.

**What it shows:**
- Total number of vectors in index
- Vector dimensions and metric type
- Vector norm statistics (mean, std)
- Distribution of `doc_type` (e.g., course_syllabus, program_requirement)
- Distribution of `department` (e.g., ELEC, COMP, MATH)
- Top course codes indexed

**Usage:**
```powershell
python inspect_faiss.py
# or for specific model/index:
python inspect_faiss.py --index faiss_index_bge-small-en-v1.5
```

**When to use:**
- After running `ingest_university.py` to verify metadata was extracted
- To quickly check index size and coverage

---

### 2. `plot_faiss_pca.py`
Visualize vector embeddings in 2D using PCA projection, colored by department/doc_type.

**What it shows:**
- Scatter plot of first 3000 vectors (for performance)
- Colors represent top departments/doc types
- Clustering patterns indicate embedding quality

**Usage:**
```powershell
python plot_faiss_pca.py
# or for specific index:
python plot_faiss_pca.py --index faiss_index_bge-small-en-v1.5
```

**Dependencies:**
```powershell
pip install matplotlib scikit-learn
```

**When to use:**
- Verify that semantically similar documents cluster together
- Identify outliers or problematic chunks

---

### 3. `query_debug.py`
Test retrieval quality with sample queries (especially course-code queries).

**What it shows:**
- Top-k retrieved chunks for each test query
- Metadata (course_code, department, doc_type) for each result
- FAISS similarity scores (raw and reranked)
- Whether exact course-code matching boost is working
- Output location/page numbers for each chunk

**Usage:**
```powershell
python query_debug.py
# or with custom queries:
python query_debug.py --queries "ELEC1100" "COMP1021" "What is prerequisite for circuits?"
```

**Sample queries included:**
- `ELEC1100` - exact course code
- `What are prerequisites for ELEC2400?` - natural language + code
- `COMP1001 syllabus` - code + keyword
- `program requirements ELEC` - department + keyword

**When to use:**
- After ingestion, to verify course-code queries return correct syllabus
- Debug why certain queries miss expected documents
- Tune reranking weights if needed

---

### 4. `dump_cached_stream.py`
Dump raw cached backend stream payloads from the local shelve KV store.

**What it shows:**
- Available cached `search_uuid` keys
- Full raw concatenated stream payload for a selected key

**Usage:**
```powershell
python dump_cached_stream.py --list
python dump_cached_stream.py --key <search_uuid>
python dump_cached_stream.py --key <search_uuid> --out raw_stream.txt
```

**When to use:**
- Capture real backend payloads for frontend parser regression tests
- Compare live `/query` output against fixture files in `newUI` debug pages

---

### 5. `check_course_facts.py`
Manual factual regression check for high-risk course identity questions.

This script is **test-only**. It does **not** hardcode answers into the chatbot runtime. Instead, it verifies that the current retrieval path still behaves correctly for a few known-failure cases.

**What it checks right now:**
- `ELEC3130` resolves to the official title `Digital Image Processing`
- `ELEC2200` falls back to a verification note if no current official syllabus source is available
- The retrieval path does not prefer outdated or secondary sources for these direct course-code questions

**Usage:**
```powershell
python check_course_facts.py
```

**When to use:**
- After changing ingestion, reranking, or prompt rules
- After noticing a course-code factual error in chat output
- Before committing changes that affect course identity or prerequisites

---

### 6. `check_course_fact_risks.py`
Broader factual-risk inspection for course-code queries.

**What it shows:**
- Which retrieved contexts are potentially risky because they come from secondary sources or ambiguous course-code mentions
- Whether the official-code guardrail is still working

**Usage:**
```powershell
python check_course_fact_risks.py
```

**When to use:**
- After re-ingesting `ECEknowledge`
- When a query returns a plausible but wrong course identity / prerequisite / offering statement

---

## Quick Start

After ingesting ECEknowledge with `ingest_university.py`:

```powershell
cd testing

# 1. Check index stats
python inspect_faiss.py

# 2. Visualize embeddings
pip install matplotlib scikit-learn
python plot_faiss_pca.py

# 3. Test retrieval quality
python query_debug.py

# 4. Test custom queries
python query_debug.py --queries "ELEC1100" "COMP2011" "Dorm application"

# 5. Check known factual regressions
python check_course_facts.py
python check_course_fact_risks.py
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'faiss'"
```powershell
pip install faiss-cpu
```

### "ModuleNotFoundError: No module named 'langchain_...'"
```powershell
pip install langchain langchain-community langchain-huggingface
```

### "Index file not found"
- Confirm you ran `python ingest_university.py` first
- Check index path is correct in script (default: `../faiss_index_all-MiniLM-L6-v2/`)

---

## Index Interpretation

### Metadata Fields (per chunk)
- `course_code`: e.g., "ELEC1100", "COMP2011" (extracted or inferred)
- `department`: e.g., "ELEC", "COMP", "MATH" (first 4 chars of code)
- `doc_type`: "course_syllabus", "program_requirement", "faq", "general"
- `source_relpath`: Relative path to source file
- `chunk_id`: Sequential ID within index
- `source_name`: Filename

### Reranking Logic (faiss_rag.py)
For course-code queries like "ELEC1100":
- **Strong boost** (-0.45): If chunk's `course_code` matches exactly
- **Medium boost** (-0.25): If query code appears in chunk text/path
- **Penalty** (+0.12): If chunk contains a *different* course code (noise gate)
- **Small boost** (-0.05): If department matches

Lower reranked score = higher rank in final results.

### Factual Regression Checks
If you spot a similar factual error:
1. Add the query to `check_course_facts.py` if it is a direct course-identity question.
2. Add it to `check_course_fact_risks.py` if you want to inspect the retrieved contexts and why they look risky.
3. Re-run the scripts after ingestion or retrieval changes.

Keep the checks narrow and source-based. The goal is to catch regressions, not to hardcode all course facts into runtime code.

---

## Notes

- Scripts use default model: `all-MiniLM-L6-v2`
- To test a different embedding model, update `INDEX_DIR` or use `--index` flag
- Visualization samples first 3000 vectors for speed (configurable in script)
- Query debug shows raw FAISS scores + reranked scores for transparency

