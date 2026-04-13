# ECEasy Runtime Zip Checklist

Use this checklist when updating `ECEasy.7z` for server deployment.

Scope: **Python backend runtime + built frontend assets only**.

## Quick Pack Script (recommended)

Use `pack_runtime.ps1` from project root:

```powershell
# Dry run: show what will be included
.\pack_runtime.ps1 -ArchiveFormat zip -DryRun

# Build zip package (default output: .\ECEasy.zip)
.\pack_runtime.ps1 -ArchiveFormat zip

# Build 7z package (default output: .\ECEasy.7z) - requires 7z in PATH
.\pack_runtime.ps1 -ArchiveFormat 7z
```

## 1) Must Include (always)

- `eceasy_local_server.py`
- `eceasy_server/`
- `ecEasyPrompts.py`
- `requirements-runtime.txt` (or `requirements.txt` if you use only one file)
- `.env` (server version only; keep out of git)
- `newUI/` (built files from `newDesign/AiChatBotInterfaceDesign`)

## 2) Knowledge Base Files (choose by mode)

### If `.env` has `KNOWLEDGE="faiss"`

- `faiss_rag.py`
- FAISS index folder used by your embedding model, for example:
  - `faiss_index_bge-small-en-v1.5/` or
  - `faiss_index_all-MiniLM-L6-v2/`
- `ECEknowledge/` (documents and image resources used for citations/suggested images)
- `image_retrieval.py` (if you use suggested images)
- `models/<your-local-embedding-model>/` only if `EMBEDDING_MODEL_LOCAL_PATH` points to local model files

### If `.env` has `KNOWLEDGE="chroma"`

- `arag/`
- Any required Chroma vector store folders/data under `arag/`

## 3) Optional But Useful

- `docs/deployment_Guide.md`
- `README.md`
- `LICENSE`

## 4) Do Not Pack (exclude)

- `.git/`, `.github/`, `.claude/`, `.agent/`
- `newDesign/` source project (if `newUI/` built output is already included)
- `node_modules/`
- `__pycache__/`, `*.pyc`
- local test/debug folders unless needed in server runtime (for example `testing/`)
- large archives not needed by server (for example old backups)

## 5) Pre-Zip Sanity Checks

- `.env` matches server runtime values:
  - `HOST`, `PORT`
  - `UI_VERSION="newUI"` (if using new UI)
  - `KNOWLEDGE`
  - model/index paths
- `newUI/index.html` exists and asset files are present in `newUI/assets/`
- Selected FAISS/Chroma index folder exists on disk and is readable
- Required API keys exist on server `.env` (or plan to use user-provided keys)

## 6) Post-Upload Quick Check

- Start server and verify home page loads
- Send one query and confirm:
  - response streams
  - sources appear
  - related questions appear
  - (if enabled) suggested images appear

---

Tip: keep this file updated whenever runtime structure changes (new required scripts, renamed index folder, or env keys).
