# ECEasy Project Overview

## What It Is
ECEasy is a **RAG-powered chatbot** designed for HKUST ECE (Electronic & Computer Engineering) students. It answers questions about ECE courses, program requirements, and university life by combining a local knowledge base with web search.

---

## Architecture Overview

### Backend Stack
- **Framework**: FastAPI (ASGI server via Uvicorn) → `http://localhost:8000`
- **Request Handling**: Pydantic for schema validation
- **Logging**: Loguru for structured logging with rotation
- **Environment**: Python 3.10.11, managed via `.env` configuration

### LLM Providers (switchable via `.env`)
The backend supports three LLM providers:

| Provider | Default Model | Use Case |
|----------|---------------|----------|
| **Ollama** | `qwen3:4b` | Local, no API key needed, faster on weak hardware |
| **OpenAI** | `gpt-4o` / `gpt-5-mini` | Remote, powerful, requires API key |
| **DeepSeek** | `deepseek-v3` | Remote alternative, competitive pricing |

Users can select at runtime via the UI (`use_server_key=true` uses backend credentials from `.env`).

### Request Pipeline

For each query:
1. **RAG Context Retrieval** (`faiss_rag.py` or `arag/arag.py`)
   - Retrieves up to 40 candidate chunks from vector store
   - Filters by similarity threshold (FAISS: `MAGIC_NUMBER = 1.5`)
   - Returns top 8 most relevant chunks with metadata
   
2. **Web Search Fallback** (`search_with_duckduckgo()`)
   - If RAG returns < 8 results, queries DuckDuckGo via `ddgs` package
   - Supplements with real-time web results
   
3. **LLM Streaming Response**
   - Sends contexts + query to LLM with citation instructions
   - Streams response chunks as they arrive
   - Stops on configured stop words (e.g., `<|im_end|>`)
   
4. **Related Questions Generation** (optional)
   - Second LLM call to generate 3 follow-up questions
   - Disabled if `RELATED_QUESTIONS=false` in `.env`
   
5. **Image Suggestions** (if available)
   - Queries image manifest from FAISS index
   - Suggests relevant images based on query/response topics
   
6. **Response Caching**
   - Full response cached to `shelve` KV store (`.kv` files)
   - Allows users to retrieve past searches by UUID

### Response Format (Streaming Protocol)
```
[JSON contexts]\n\n__LLM_RESPONSE__\n\n[LLM streaming text]\n\n__RELATED_QUESTIONS__\n\n[JSON questions][\n\n__SUGGESTED_IMAGES__\n\n[JSON images]]
```

---

## Knowledge Base & RAG System

### Two Knowledge Base Options (via `KNOWLEDGE=` in `.env`)

#### 1. **FAISS** (Primary - ECE Content) ⭐ Recommended
- **Location**: `./faiss_index_<model_name>/`
- **Embedding Model**: Configurable (default: `BAAI/bge-small-en-v1.5`)
- **Source Data**: `ECEknowledge/` folder
- **Includes**: Course syllabi, program requirements, FAQs, images
- **Ingestion**: `ingest_FAISS.py`
- **Similarity Metric**: L2 distance (lower = more similar)
- **Threshold**: Chunks with distance ≥ 1.5 filtered out
- **Features**: 
  - Course code extraction & metadata enrichment
  - Image catalog with document type classification
  - Supports PDF, DOCX, TXT, HTML, images

#### 2. **ChromaDB** (Legacy - Networking Content)
- **Location**: `./arag/chromaVectorStore/`
- **Embedding Model**: `sentence-transformers/all-mpnet-base-v2`
- **Collection**: `"nettyRAG"` (legacy name from Netty project)
- **Source Data**: Computer networking documents (old Netty project)
- **Ingestion**: `ingest_Chroma.py` (reads from `localData/`)
- **Similarity Metric**: Cosine distance
- **Threshold**: Documents with score ≥ 1.0 filtered out

**Status**: ⚠️ ChromaDB still contains old Netty content; FAISS is the active ECE knowledge base.

### Embedding Model Management
- **HuggingFace Transformers**: Models auto-downloaded from Hub on first run
- **Caching**: Stored in `./arag/modelCache/` (ChromaDB) or `./models/` (FAISS local path)
- **GPU Support**: Auto-detected; uses CUDA if available, falls back to CPU
- **Offline Mode**: Set `EMBEDDING_MODEL_LOCAL_PATH` in `.env` to use pre-downloaded model

### Knowledge Base Contents (`ECEknowledge/`)
Rich collection ready for ingestion:
- **Program Overviews**: BEng ECE, BEng MEIC, Common Core
- **Course Syllabi**: 
  - ELEC: 34 courses
  - COMP: 22 courses
  - MATH: 19 courses
  - PHYS: 2 courses
- **Program Requirements**: `25-26elec.pdf`, `25-26meic.pdf`, `minor-robo.pdf`
- **FAQ**: `FAQs.docx`
- **Images**: Diagrams, screenshots (cataloged in image manifest)

---

## Frontend

### Two UI Options (via `UI_VERSION=` in `.env`)

#### 1. **newUI** (React/Vite) ⭐ Modern
- **Location**: `./newUI/`
- **Tech**: React, Vite, Tailwind CSS
- **Features**: 
  - Streaming response display
  - Citation badges with popovers
  - Mermaid diagram rendering
  - KaTeX math rendering
  - Syntax highlighting for code blocks
  - Related questions sidebar
  - Image suggestions panel
- **Status**: Active development

#### 2. **oldUI** (Next.js) - Legacy
- **Location**: `./web/` (source), `./ui/` (pre-built output)
- **Tech**: Next.js, React, TypeScript, Tailwind CSS
- **Status**: Serves as fallback, still functional

Both UIs parse the three-part streaming protocol and render:
- **Sources Panel**: Links with favicons, URLs, page numbers
- **Main Response**: Markdown with inline citations
- **Related Questions**: Clickable follow-up suggestions
- **Images**: Suggested relevant images from knowledge base

### Frontend Debug Modes (newUI)
- `.../newUI/chat.html?debugSample=1` → static sample response rendering
- `.../newUI/chat.html?debugFixture=1` → fixture/raw-stream parser test page
- Use `chat.html?debug...` (without extra `/` before `?`) for static file hosting compatibility.

### Raw Response Storage
- Streaming payload is assembled in `eceasy_server/streaming.py` and cached by `search_uuid` into shelve (`KV_NAME`).
- Cache is replayed in `eceasy_server/app.py` (`/query`) if the same UUID is requested.
- This makes it easy to copy raw payloads for parser regression tests.

---

## Configuration (`.env`)

Key environment variables:

```env
# Knowledge Base
KNOWLEDGE="faiss"                           # "faiss" or "chroma"
EMBEDDING_MODEL_HUB_NAME="BAAI/bge-small-en-v1.5"
EMBEDDING_MODEL_LOCAL_PATH="./models/bge-small-en-v1.5"  # Optional: offline mode

# UI
UI_VERSION="newUI"                          # "newUI" or "oldUI"

# Server
HOST="0.0.0.0"
PORT=8000
KV_NAME="eceasy-chat-local.kv"

# LLM Provider (backend fallback)
LLM_PROVIDER="openai"                       # "ollama", "openai", or "deepseek"

# LLM Model Selection
OLLAMA_BASE_URL="http://localhost:11434/v1"
OLLAMA_MODEL="qwen3:4b"
OPENAI_API_KEY="sk-..."
OPENAI_MODEL="gpt-5-mini"
DEEPSEEK_API_KEY="sk-..."
DEEPSEEK_MODEL="deepseek-v3"

# Features
RELATED_QUESTIONS="true"                    # Generate follow-up questions
```

See `.env.example` for full reference.

---

## Code Structure

### Key Modules

| Module | Purpose |
|--------|---------|
| `eceasy_server/app.py` | FastAPI application setup, CORS, static file serving |
| `eceasy_server/config.py` | Environment loading, configuration constants |
| `eceasy_server/schemas.py` | Pydantic models for request/response validation |
| `eceasy_server/llm.py` | LLM provider resolution, model selection logic |
| `eceasy_server/retrieval.py` | RAG context retrieval, DuckDuckGo search, related questions |
| `eceasy_server/streaming.py` | Response streaming, caching, image suggestion orchestration |
| `faiss_rag.py` | FAISS vector store querying, similarity reranking |
| `arag/arag.py` | ChromaDB vector store interface (legacy) |
| `image_retrieval.py` | Image manifest loading, search by course code/dept/keywords |
| `ingest_FAISS.py` | Build FAISS index from `ECEknowledge/`, extract metadata, catalog images |
| `ingest_Chroma.py` | Build ChromaDB from `localData/` (legacy) |
| `ecEasyPrompts.py` | System prompts for RAG answering and related question generation |

### Frontend

| Folder | Purpose |
|--------|---------|
| `web/src/` | React/Vite source (Next.js-style layout, `page.tsx`) |
| `ui/` | Pre-built static output served by FastAPI |
| `public/` | Static assets (favicons, images) |

---

## Dependencies Overview

### Core Dependencies (Python 3.10.11)

**Web Framework**
- `fastapi==0.128.0` - Fast web framework
- `uvicorn==0.40.0` - ASGI server
- `pydantic==2.12.5` - Request validation via type hints

**LLM Integration**
- `openai==1.109.1` - OpenAI API client
- `ddgs==9.10.0` - DuckDuckGo search (web fallback)

**RAG & Vector Search**
- `faiss-cpu==1.13.2` - Vector indexing (ECE knowledge)
- `langchain-community==0.3.14` - Integration framework
- `langchain-chroma==0.2.0` - ChromaDB adapter
- `langchain-huggingface==0.1.2` - HuggingFace embeddings
- `langchain-openai==0.3.1` - OpenAI LangChain integration
- `langchain-text-splitters==0.3.8` - Document chunking
- `sentence-transformers==5.2.0` - Pre-trained embeddings
- `torch==2.10.0` - ML backend for embeddings (GPU-capable)

**Document Processing**
- `pypdf==6.7.5` - PDF text extraction
- `docx2txt==0.9` - Word document parsing
- `beautifulsoup4==4.14.3` - HTML parsing

**Utilities**
- `httpx==0.28.1` - Modern HTTP client
- `python-dotenv==1.2.1` - Environment loading
- `loguru==0.7.3` - Structured logging
- `pillow==11.3.0` - Image processing

**Special**
- `hf-xet==1.3.2` - HuggingFace extensible embeddings toolkit (used for embedding model management)

---

## Current State & Known Issues

| Status | Issue | Detail |
|--------|-------|--------|
| ✅ | FAISS Active | ECEasy now uses FAISS for ECE knowledge (primary) |
| ✅ | Ingestion Ready | `ingest_FAISS.py` successfully builds indexes from ECEknowledge/ |
| ✅ | Image Support | Image manifest generated and integrated into responses |
| ⚠️ | ChromaDB Legacy | Old Netty content still in `arag/chromaVectorStore/` — kept for reference |
| ⚠️ | UI Migration | Both `newUI` and `oldUI` functional; `newUI` recommended |
| 🔧 | LLM Integration | Supports Ollama (local), OpenAI, DeepSeek (runtime switching) |
| 📝 | Prompt Quality | System prompts tuned for ECE student context |

---

## How to Run

### Quick Start
```powershell
# Windows batch file
.\run_local_server.bat

# Or manual Python
pip install -r requirements.txt
python eceasy_local_server.py
```

Access UI: `http://localhost:8000/ui/index.html` (newUI) or `http://localhost:8000/` (oldUI)

### Ingest ECE Knowledge
```powershell
# Build FAISS index from ECEknowledge/
python ingest_FAISS.py

# (Legacy) Build ChromaDB from localData/
python ingest_Chroma.py
```

### Development
```powershell
# Rebuild frontend (Next.js)
cd web
npm run build
# Output goes to ./ui/

# Or use Vite (newUI)
npm run dev
```

---

## Dependencies & Python Version

**Python**: 3.10.11 (required for compatibility)  
**Platform**: Windows (tested), Linux/macOS (should work)  
**GPU**: Optional (torch auto-detects CUDA; falls back to CPU)

All pinned versions in `requirements.txt` match the tested working environment.

