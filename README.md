

![ECEasy Logo](web/src/app/logo_name.svg)

An AI chatbot system designed to accommodate inter-department queries, in applications such as ECE students.
Part of this README and most of the code was done with the help of AI.

# How to Run ECEasy

This setup allows you to run the ECEasy backend locally using **Ollama**, **OpenAI**, or **DeepSeek** as the LLM provider, with support for multiple knowledge bases (FAISS or ChromaDB) and flexible UI selection.

## 1. Prerequisites

- **Python 3.10+** (Tested on Python 3.10; newer versions may work but may require dependency adjustments)
- **Node 18+** (preferably with npm 9+) — required to build the frontend UI
- (Optional) Use a virtual environment:
  ```bash
  python -m venv .venv
  .venv\Scripts\activate  # Windows
  # source .venv/bin/activate  # Linux/Mac
  ```

## 2. Configuration (`.env` file)

Copy `.env.example` to `.env` and configure the following options:

### LLM Provider Selection

For `UI_VERSION="newUI"`, users can choose provider (`OpenAI`/`DeepSeek`) and enter their own API key in the chat UI before sending messages.
If they skip, ECEasy uses server keys from `.env` (this may incur shared project costs and can be rate-limited).

#### OpenAI:
```dotenv
LLM_PROVIDER="openai"
OPENAI_API_KEY="sk-your-openai-api-key"
OPENAI_MODEL="gpt-4o"
OPENAI_BASE_URL="https://api.openai.com/v1"  # Optional: for custom endpoints
```

#### DeepSeek:
```dotenv
LLM_PROVIDER="deepseek"
DEEPSEEK_API_KEY="sk-your-deepseek-api-key"
DEEPSEEK_BASE_URL="https://api.deepseek.com"
DEEPSEEK_MODEL="deepseek-chat"
```

#### Ollama (Local, recommended for development):
```dotenv
LLM_PROVIDER="ollama"
OLLAMA_BASE_URL="http://localhost:11434/v1"
OLLAMA_MODEL="qwen3:4b" 
# Pull the model first: ollama pull qwen3:4b
```

### Knowledge Base Selection

```dotenv
# Choose knowledge source:
#   "faiss"  → FAISS vector store (ECE/course knowledge from ./ECEknowledge/)
#   "chroma" → ChromaDB (legacy network knowledge from ./arag/chromaVectorStore/)
KNOWLEDGE="faiss"
```

**Note:** To use FAISS, first run the ingestion script (see **Section 3**).

### Embedding Model Configuration (FAISS only)

```dotenv
# HuggingFace Hub model ID (used during ingestion and retrieval)
EMBEDDING_MODEL_HUB_NAME="all-MiniLM-L6-v2"

# Optional: path to local embedding model (for offline use)
# EMBEDDING_MODEL_LOCAL_PATH="models/all-MiniLM-L6-v2"
```

### UI Version Selection

```dotenv
# Choose UI frontend:
#   "newUI" → Modern React/Vite interface (recommended, located at ./newUI/)
#   "oldUI" → Legacy Next.js interface (located at ./ui/)
UI_VERSION="newUI"
```

### Server Configuration

```dotenv
HOST="0.0.0.0"
PORT=8000
```

### Feature Flags

```dotenv
# Enable/disable related questions generation
RELATED_QUESTIONS="true"

# Chat history storage (database file)
KV_NAME="eceasy-chat-local.kv"
```

## 3. Knowledge Base Setup (FAISS)

If using `KNOWLEDGE="faiss"`, you must first ingest documents from `./ECEknowledge/`:

```bash
pip install -r requirements_local.txt
python ingest_university.py
```

This will:
- Read all `.pdf`, `.docx`, `.txt` files from `./ECEknowledge/` (recursively)
- Extract structured metadata: course codes, departments, document type
- Build a FAISS vector index at `./faiss_index_all-MiniLM-L6-v2/` (or custom model folder)
- Use the embedding model specified in `EMBEDDING_MODEL_HUB_NAME`

**Note:** Windows users may need to stop the server and close all Python processes before re-ingesting to avoid file lock errors.

## 4. Build the Frontend UI

ECEasy has two UI options; each must be built separately.

### Option A: Build New UI (React/Vite) — Recommended

```bash
cd newDesign/AiChatBotInterfaceDesign
npm install
npm run build
```

Output: `./newDesign/Aichatbotinterfacedesign/dist/` → served at `/newUI/`

### Option B: Build Old UI (Next.js)

```bash
cd web
npm install
npm run build
```

Output: `./web/.next/` → served at `/ui/`

## 5. Start the Backend Server

### Windows (using provided script):
```bash
.\run_local_server.bat
```

### Manual (all platforms):
```bash
pip install -r requirements_local.txt
python eceasy_local_server.py
```

The server will start at `http://0.0.0.0:8000` (or configured `HOST:PORT`).

## 6. Access the UI

Open your browser and navigate to the selected UI:

- **New UI (React/Vite):** [http://localhost:8000/](http://localhost:8000/) or [http://localhost:8000/newUI/index.html](http://localhost:8000/newUI/index.html)
- **Old UI (Next.js):** [http://localhost:8000/ui/index.html](http://localhost:8000/ui/index.html)

The server auto-redirects based on `UI_VERSION` setting in `.env`.

## 7. Testing & Debugging

Use the diagnostic tools in `./testing/` to inspect index quality and retrieval behavior:

```bash
cd testing

# Check FAISS index statistics and metadata distribution
python inspect_faiss.py

# Visualize vector embeddings in 2D (requires matplotlib, scikit-learn)
pip install matplotlib scikit-learn
python plot_faiss_pca.py

# Debug retrieval quality for test queries
python query_debug.py --queries "ELEC1100" "COMP2011"
```

See `./testing/README.md` for detailed documentation.

## 8. Features

### Supported LLM Providers
- ✅ Ollama (local, no API keys needed)
- ✅ OpenAI (requires API key)
- ✅ DeepSeek (requires API key)

### API Key Mode (New UI)
- ✅ Prompt users to select `OpenAI` or `DeepSeek`
- ✅ Allow users to provide their own API key per chat session
- ✅ Allow explicit skip to use server-side key with cost warning
- ✅ Keep backward compatibility for old UI / legacy clients (server key path)

### Knowledge Bases
- ✅ **FAISS** — Fast vector search over structured course/department knowledge
  - Extracts and prioritizes: course codes, departments, document type
  - Metadata-aware reranking for exact course-code queries
  - Suitable for: course information, prerequisites, program requirements
- ✅ **ChromaDB** — Flexible vector store for domain-specific knowledge
  - Suitable for: general Q&A, legacy network knowledge

### UI Features (New UI)
- 💬 Real-time streaming responses
- 🛑 Stop generation button to interrupt long responses
- 📋 Source citations with metadata
- 🔗 Related questions for follow-ups
- 📱 Responsive design with sidebar navigation

## 9. Troubleshooting

### Server fails to start
- Ensure Python 3.10+ is installed: `python --version`
- Install dependencies: `pip install -r requirements_local.txt`
- Check `.env` file exists and contains valid `LLM_PROVIDER` setting

### UI not loading
- Verify frontend was built: check for `./ui/` or `./newUI/` directories
- Confirm `UI_VERSION` in `.env` matches built UI folder
- Clear browser cache and hard refresh (Ctrl+Shift+R / Cmd+Shift+R)

### FAISS retrieval not working
- Verify ingestion completed: `python ingest_university.py`
- Check `./ECEknowledge/` contains documents (`.pdf`, `.docx`, `.txt`)
- Confirm `KNOWLEDGE="faiss"` in `.env`
- Test with: `cd testing && python query_debug.py`

### Ollama not responding
- Ensure Ollama is running: `ollama serve` (in another terminal)
- Verify model is installed: `ollama list` should show your model
- Pull model if missing: `ollama pull qwen3:4b`
- Check `OLLAMA_BASE_URL` matches Ollama's listen address

### File permission errors on Windows
- Close all Python processes and the server
- Delete any `faiss_index_*.__bak__` or `__tmp__` folders manually if they exist
- Re-run ingestion: `python ingest_university.py`

## 10. Architecture Notes

- **Backend:** FastAPI (Python) with streaming responses
- **Frontend (New):** React + Vite + Tailwind CSS
- **Frontend (Old):** Next.js
- **Knowledge Base:** FAISS (vector search) or ChromaDB (hybrid search)
- **Embedding Model:** HuggingFace (configurable, default: `all-MiniLM-L6-v2`)
- **Chat History:** Local shelve database (`.kv` files)

## 11. Environment Variables Summary

| Variable | Options | Default | Purpose |
|----------|---------|---------|---------|
| `LLM_PROVIDER` | `ollama`, `openai`, `deepseek` | `ollama` | LLM backend |
| `KNOWLEDGE` | `faiss`, `chroma` | `faiss` | Knowledge base |
| `UI_VERSION` | `newUI`, `oldUI` | `newUI` | Frontend interface |
| `EMBEDDING_MODEL_HUB_NAME` | HuggingFace model ID | `all-MiniLM-L6-v2` | Embedding model |
| `HOST` | IP address | `0.0.0.0` | Server listen address |
| `PORT` | Port number | `8000` | Server port |
| `RELATED_QUESTIONS` | `true`, `false` | `true` | Generate follow-up questions |

---

**For more information**, see the diagnostic tools in `./testing/` or check individual module docstrings.

