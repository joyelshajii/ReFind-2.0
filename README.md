# ReFind

> **Find what you remember.**  
> Built by team **BitByBit**

ReFind is a privacy-focused, AI-powered contextual memory and personal search engine. It bridges the gap between human memory (fuzzy, contextual clues like person, rough time, visual appearance, subject keywords) and digital file retrieval.

Instead of requiring exact filenames or folders, ReFind understands natural queries like:
> *“Find the architecture PDF Rahul sent me around February. It had PostgreSQL and BLE.”*  
> *“Find the file containing the word sustainability.”*  
> *“Find that document where I wrote about databases.”*

The default pipeline uses **local EmbeddingGemma 2** for text, query and image embeddings, local OCR, exact vector retrieval, keyword/metadata signals, and optional Jina reranking. Gemini explanations are generated when you open a result’s context. See [performance behavior and measurements](backend/docs/performance.md).

---

## 👥 Setup & Quick Start

Welcome to ReFind! Do NOT commit real API keys to Git.

### 1. Clone the Repository
```bash
git clone https://github.com/joyelshajii/ReFind.git
cd ReFind
```

### 2. Configure Your Local Environment Variables
Create your own local `.env` from the provided `.env.example`:

**Windows:**
```powershell
copy .env.example .env
```

**Linux / macOS:**
```bash
cp .env.example .env
```

Open `.env` and fill in your keys:
```env
# AI & Embedding Providers (Free Tiers)
GEMINI_API_KEY=your_gemini_api_key_here
JINA_API_KEY=your_jina_api_key_here
GEMINI_MODEL=gemini-2.5-flash-lite
GEMINI_EMBEDDING_MODEL=gemini-embedding-2

# Database Configuration (Optional: Leave empty for zero-config SQLite + vector store)
DATABASE_URL=
```
*(The `.env` file is ignored by `.gitignore` and must never be committed).*

---

### 3. Install Backend Dependencies & Start FastAPI
Open your first terminal:
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
*The backend starts at `http://127.0.0.1:8000` with interactive API docs at `http://127.0.0.1:8000/docs`.*

---

### 4. Install Frontend Dependencies & Start Next.js
Open a second terminal:
```bash
cd frontend
npm install
npm run dev
```
*(On Windows PowerShell, use `npm.cmd install` and `npm.cmd run dev` if script execution policies apply).*

Open your browser at:
```
http://localhost:3000
```

---

## 🏗️ Default Search Pipeline

```text
Query → local clue parsing → EmbeddingGemma 2 query vector
      → exact cached matrix retrieval + keyword/filename/metadata matching
      → Jina reranking (when configured) → ranked results and evidence

Open context → relevant passages → optional grounded Gemini explanation
```

`LOCAL_FAST_MODE=true` takes precedence over the legacy `EMBEDDING_PROVIDER`
setting. Text and image embeddings stay local. One process keeps one model on
the GPU. The backend warms it at startup and lets searches run between indexing
batches. On Windows, `backend/start-local.ps1` starts the configured environment.


### 5-Signal Scoring Formula
$$\text{Score} = w_{\text{sem}} \cdot S_{\text{semantic}} + w_{\text{rerank}} \cdot S_{\text{reranker}} + w_{\text{kw}} \cdot S_{\text{keyword}} + w_{\text{meta}} \cdot S_{\text{metadata}} + w_{\text{fn}} \cdot S_{\text{filename}}$$

Configured in `backend/app/core/config.py`:
- `WEIGHT_SEMANTIC = 0.45`
- `WEIGHT_RERANKER = 0.20`
- `WEIGHT_KEYWORD = 0.15`
- `WEIGHT_METADATA = 0.10`
- `WEIGHT_FILENAME = 0.10`

---

## 🛡️ Privacy & Local-First Ingestion
- **Local Text Extraction**: PDFs processed locally using **PyMuPDF**, Word documents via **python-docx**, presentations via **python-pptx**, and text files natively.
- **Local OCR**: Scanned PDFs and images (PNG, JPG, JPEG) are OCR-processed locally via **RapidOCR** with **Tesseract** fallback in local-fast mode.
- **Local indexing**: Local-fast mode does not send documents or images to external AI services. The explicitly selected legacy mode can use remote image analysis and embeddings.
- **Minimal API Payloads**: Only compact search queries, document chunks, and retrieved context snippets are passed to Gemini / Jina.
- **Change Tracking & Caching**: SHA-256 file hashing prevents regenerating embeddings for unchanged documents.
- **Zero-Downtime Fallback**: If Gemini or Jina is temporarily unavailable or rate-limited, ReFind gracefully degrades to local regex query understanding and hybrid retrieval without throwing runtime errors.

---

## 🗄️ Database & Vector Search Engine

ReFind supports two database setups seamlessly:

1. **Zero-Configuration Mode (Default)**:
   - Uses local SQLite with 768-dimensional dense vector similarity. Runs out of the box with zero external database configuration.

2. **Optional PostgreSQL storage**:
   - Start the container using Docker:
     ```bash
     docker compose up -d
     ```
   - Set in `.env`:
     ```env
     DATABASE_URL=postgresql://refind_user:refind_password@localhost:5432/refind_db
     ```
   - ReFind enables the `vector` extension, but the current schema stores vectors as JSON and uses exact application-side scoring. Switching the URL alone does not create a native vector index. The revision-cached search snapshot currently targets SQLite.

---

## 🧪 Running Automated Tests

Run the full pytest suite from the `backend/` directory:
```bash
cd backend
python -m pytest tests/test_refind.py -v
python -m pytest tests/test_ai_pipeline.py -v
python -m pytest tests/test_image_pipeline.py -v
```

### Performance and local model checks

See [current performance measurements and reproduction commands](backend/docs/performance.md)
and [local indexing documentation](backend/docs/local_fast_indexing.md). The older
[Phase 3 image benchmark](backend/docs/phase3_image_benchmark.md) documents the
historical remote-vision comparison, not the current default pipeline.
