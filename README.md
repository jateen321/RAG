<img width="1318" height="641" alt="image" src="https://github.com/user-attachments/assets/f4ff5b58-b10c-4700-99d8-a9b6acfa7b51" />

# 📚 Hindi Textbook RAG

Query your scanned Hindi textbook PDFs using AI — **completely free!**

Uses **Google Gemini (free API)** for smart answers and **EasyOCR** for Hindi text extraction.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> ⏳ First install takes a few minutes (EasyOCR downloads ~100MB model).

### 2. Set Up Your Free API Key

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Click **"Get API Key"** → **"Create API Key"**
3. Create your `.env` file:

```bash
cp .env.example .env
```

4. Edit `.env` and paste your key:

```
GEMINI_API_KEY=your_actual_key_here
```

### 3. Index a PDF

Drop your Hindi textbook PDF into the `data/` folder, then:

```bash
python app.py index data/your_textbook.pdf
```

This will:
- Extract Hindi text from every page (OCR)
- Split into searchable chunks
- Create embeddings and store them

### 4. Ask Questions!

**One-shot question:**
```bash
python app.py ask "मुगल साम्राज्य की स्थापना कब हुई?"
python app.py ask "What are the main topics in chapter 1?"
```

**Interactive chat:**
```bash
python app.py chat
```

## 📋 All Commands

| Command | Description |
|---|---|
| `python app.py index <pdf>` | Index a PDF for searching |
| `python app.py ask "question"` | Ask a one-shot question |
| `python app.py chat` | Start interactive chat |
| `python app.py status` | Show database statistics |
| `python app.py reset` | Clear all indexed data |

## 🏗️ How It Works

```
Scanned PDF → EasyOCR (Hindi) → Text Chunks → Gemini Embeddings → ChromaDB
                                                                      ↓
Your Question → Gemini Embedding → Similarity Search → Top 5 Chunks
                                                                      ↓
                                          Gemini Flash + Context → Answer!
```

## 📁 Project Structure

```
RAG/
├── app.py              # CLI interface (main entry point)
├── config.py           # Configuration & constants
├── ocr_engine.py       # PDF → text extraction (EasyOCR)
├── indexer.py          # Text → chunks → embeddings → ChromaDB
├── retriever.py        # Semantic search in ChromaDB
├── rag_engine.py       # Retrieve + Generate answers
├── requirements.txt    # Python dependencies
├── .env                # Your API key (private, not in git)
├── .env.example        # Template for .env
├── data/               # Drop your PDFs here
└── chroma_db/          # Vector database (auto-created)
```

## ⚙️ Configuration

Edit `config.py` to tune these settings:

| Setting | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 500 | Characters per chunk |
| `CHUNK_OVERLAP` | 100 | Overlap between chunks |
| `TOP_K` | 5 | Number of chunks to retrieve |
| `PDF_DPI` | 200 | OCR scan resolution |
| `LLM_MODEL` | gemini-2.0-flash | Gemini model to use |

## 💡 Tips

- **Better OCR quality**: Use higher `PDF_DPI` (300) for low-quality scans, but it's slower.
- **Index multiple books**: Run `index` on multiple PDFs — they all go into the same database.
- **Ask in any language**: Questions can be in Hindi, English, or mixed.
- **Rate limits**: Free Gemini API allows ~15 requests/minute. If you hit limits, wait a minute.

## 🔧 Troubleshooting

| Issue | Solution |
|---|---|
| `GEMINI_API_KEY not set` | Create `.env` file with your key |
| OCR gives poor results | Increase `PDF_DPI` to 300 in `config.py` |
| Rate limit errors | Wait 1-2 minutes, free API has limits |
| `No indexed documents` | Run `python app.py index <pdf>` first |
