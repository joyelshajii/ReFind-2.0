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
On Windows, open PowerShell and use the backend virtual environment:
```powershell
cd backend
# Create only if .venv does not already exist:
if (-not (Test-Path .venv)) { python -m venv .venv }
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./start-local.ps1
```
Use `start-local.ps1` for subsequent starts. A bare `python` or `uvicorn`
command can select a different Python installation without PyAV or the configured
embedding dependencies. For development reload, use
`./.venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload`.

On macOS/Linux:
```bash
cd backend
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
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

### Windows tray search

Run `desktop/install-tray.ps1` from PowerShell once. It builds the native tray
launcher, starts it immediately, and registers it for your Windows sign-in.
The launcher appears in the notification area (the arrow at the right of the
taskbar). Press **Alt+Space** from any app to show the borderless search overlay.
Type a file name or description, use **Up/Down** to choose a result, and press
**Enter** or click to open the original in its default application. **Escape**
closes the overlay. Double-click the tray icon to open search, or right-click it
to open ReFind in a browser or exit.

The tray launcher starts the local EmbeddingGemma backend if needed. Run
`desktop/uninstall-tray.ps1` to stop the tray launcher and remove sign-in startup.

---

## 🏗️ Default Search Pipeline

```text
Query → local clue parsing → EmbeddingGemma 2 query vector
      → exact cached matrix retrieval + keyword/filename/metadata matching
      → Jina reranking (when configured) → ranked results and evidence

Open context → relevant passages → optional grounded Gemini explanation
```

`LOCAL_FAST_MODE=true` takes precedence over the legacy `EMBEDDING_PROVIDER`
setting. Text, image, and audio embeddings stay local. One process keeps one model on
the GPU. The backend warms it at startup and lets searches run between indexing
batches. On Windows, `backend/start-local.ps1` starts the configured environment.

Media indexing first checks embedded text subtitles, lyrics, and matching companion
transcripts (`.srt`, `.vtt`, `.lrc`, `.txt`, `.ass`, `.ssa`, `.sub`, and `.meta.json` transcript
fields). Usable text is embedded directly: no full video/audio decoding or media
inference runs on that path. Titles, comments, and empty or bitmap subtitle tracks
do not substitute for a transcript. Upload companion files with the recording or
later using the same base filename; language suffixes such as `movie.en.srt` work.
To apply a later transcript, explicitly select the recording again. Uploading a
companion alone does not reindex older recordings. Folder indexing requires an
explicit absolute folder path and includes its subfolders. A blank path never
scans previous uploads or demo files; uploads index only the files selected.

**Remove** deletes an indexed item and its saved copy inside ReFind's upload
storage, along with associated unshared companions and queued jobs. **Clear Index**
deletes all indexed data, saved upload copies (including failed/previous uploads),
and queued jobs. Source files outside upload storage are preserved. Clear remains
available when the index is empty so saved copies can still be removed. Removal
is blocked during active indexing; it cannot race an indexing run.

Without usable text, audio recordings are decoded locally with PyAV and indexed with EmbeddingGemma 2's
audio encoder in 25-second sections with 5 seconds of overlap, covering the entire
recording. Search for words or ideas you remember hearing to retrieve the original
audio with the matching time range. These are semantic audio matches, not verified
verbatim transcripts; supplied `.txt`, `.srt`, and `.lrc` transcripts also support
literal text search. EmbeddingGemma produces vectors, not written transcriptions.
Install the updated backend requirements, restart the backend, and reindex audio
folders once to replace older entries that only indexed metadata. Unchanged audio
will then reuse its index. Missing decoders or failed audio embeddings are reported
as indexing errors instead of silently creating metadata-only entries.

Videos without usable transcripts use EmbeddingGemma 2 video embeddings from 8-frame sequences sampled at
one frame per second, with 2 frames of overlap. The audio track is indexed with
the same audio pipeline when usable transcript text is unavailable. Search descriptions of
scenes or remembered dialogue to retrieve the original video and a matching time
range. Silent videos remain searchable by their visuals. Very brief events between
sampled frames can be missed. Reindex existing video folders after restarting to
apply the new inspection policy. Existing compatible direct media indexes are reused.

For media **longer than 15 minutes** without usable text (or with unknown duration),
the indexing page offers **Continue now**, **Schedule later**, and **Skip for now**.
No expensive media inference starts until a choice authorizes it. Dates are entered
in the user's local time and stored in UTC. Scheduled jobs and pending choices are
saved in the database and survive backend restarts. ReFind must be running to process
a job; if it was closed at the scheduled time, processing starts on its next run.
Changed recordings require a fresh choice before a scheduled full-media job runs.


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
