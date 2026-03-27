# 🤖 GenAI Telegram Bot — RAG + Vision

A production-ready Telegram bot with **Mini-RAG** and **Image Description** capabilities. Built with Python, `python-telegram-bot`, `sentence-transformers`, and the Anthropic Claude API (with local BLIP + Ollama fallbacks).

---

## 📐 System Architecture

```
User (Telegram)
      │
      ▼
┌─────────────────────────────────────────────────────┐
│                   bot.py (Handler Layer)             │
│   /ask  /image  /history  /clear  /summarize  /help │
└───────────┬──────────────────────┬──────────────────┘
            │                      │
            ▼                      ▼
┌─────────────────────┐  ┌────────────────────────────┐
│   rag_engine.py     │  │    vision_engine.py         │
│                     │  │                             │
│  1. Chunk documents │  │  1. Receive image bytes     │
│  2. Embed (SBERT)   │  │  2. Call Claude Vision API  │
│  3. Store in SQLite │  │     (or local BLIP fallback)│
│  4. Retrieve top-k  │  │  3. Return caption + tags   │
│  5. Build prompt    │  └─────────────┬───────────────┘
│  6. Call LLM        │                │
└──────────┬──────────┘                │
           │                           │
           ▼                           ▼
┌──────────────────────────────────────────────────────┐
│                    LLM / Vision Model                 │
│                                                      │
│  ┌──────────────────┐     ┌─────────────────────┐   │
│  │ Anthropic Claude │ OR  │  Ollama (local)      │   │
│  │ (API, fast)      │     │  phi3/mistral/llama3 │   │
│  └──────────────────┘     └─────────────────────┘   │
└──────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│                  SQLite Database                      │
│                                                      │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │   chunks    │  │ query_cache  │  │user_history│  │
│  │ (embeddings)│  │ (answers)    │  │(last 3 msg)│  │
│  └─────────────┘  └──────────────┘  └────────────┘  │
└──────────────────────────────────────────────────────┘
```

### Data Flow

```
/ask "What is RAG?"
  └─► embed query ──► cosine similarity ──► top-3 chunks
       └─► build prompt with context ──► LLM
            └─► reply with answer + sources ──► cache in SQLite

/image (send photo)
  └─► download image bytes
       └─► Claude Vision API / BLIP local
            └─► caption + 3 tags ──► reply to user
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- A Telegram bot token from [@BotFather](https://t.me/BotFather)
- Anthropic API key (optional but recommended) from [console.anthropic.com](https://console.anthropic.com)

### 1. Clone & Install

```bash
git clone <your-repo-url>
cd telebot

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your values
```

Minimum required in `.env`:
```env
TELEGRAM_BOT_TOKEN=your_bot_token
ANTHROPIC_API_KEY=your_api_key   # optional but recommended
```

### 3. Run the Bot

```bash
cd src
python bot.py
```

---

## 🐳 Docker Compose

```bash
# Copy and fill in .env
cp .env.example .env

# Build and run
docker compose up --build -d

# View logs
docker compose logs -f
```

---

## 🧪 Test RAG Without Telegram

```bash
# Test the full RAG pipeline from the command line
python test_rag.py "What is a vector embedding?"

# See raw retrieved chunks (before LLM)
python test_rag.py --chunks "How do I make masala chai?"
```

---

## 🤖 Bot Commands

| Command | Description |
|---|---|
| `/start` | Welcome message |
| `/help` | Show usage instructions |
| `/ask <question>` | Query the knowledge base (RAG) |
| `/image` | Prompt the bot to expect an image, then send a photo |
| `/history` | View your last 3 interactions |
| `/clear` | Clear your conversation history |
| `/summarize` | Summarize the last bot response |

**Pro tip:** You can also just send a plain text message and it will be treated as an `/ask` query. Just send a photo directly for image description.

---

## 📚 Knowledge Base

The bot comes with 4 sample documents in `knowledge_base/`:

| File | Contents |
|---|---|
| `tech_faq.md` | RAG, embeddings, SQLite, cosine similarity, sentence-transformers |
| `company_policy.md` | Remote work, leave, code of conduct, expenses, equipment |
| `recipes.md` | Masala chai, avocado toast, overnight oats, dal tadka |
| `ai_concepts.md` | LLMs, fine-tuning, prompt engineering, Ollama, quantization |

### Adding Your Own Documents

Just drop `.md` or `.txt` files into the `knowledge_base/` folder and restart the bot. They will be automatically chunked, embedded, and indexed on startup.

---

## 🧠 Models Used

### Embeddings (RAG)
| Model | Size | Speed | Quality |
|---|---|---|---|
| `all-MiniLM-L6-v2` *(default)* | 80MB | ⚡ Fast | ✅ Good |
| `snowflake-arctic-embed-s` | 130MB | Medium | ✅✅ Better |

Set with `EMBED_MODEL` env var. Runs **100% locally**, no API needed.

### LLM (Answer Generation)
| Option | How to Enable | Pros | Cons |
|---|---|---|---|
| **Anthropic Claude Haiku** *(default)* | Set `ANTHROPIC_API_KEY` | Fast, accurate, cheap | Requires API key |
| **Ollama (phi3/mistral)** | Set `OLLAMA_URL` | Free, private | Slower, needs GPU/RAM |
| **Extractive Fallback** | No config needed | Always works | No generation, shows snippets |

### Vision (Image Description)
| Option | How to Enable | Pros | Cons |
|---|---|---|---|
| **Claude Vision** *(default)* | Set `ANTHROPIC_API_KEY` | High quality, fast | Requires API key |
| **Local BLIP** | Uncomment in `requirements.txt` + `VISION_MODEL` | Free, private | ~1.5GB download, slower |

---

## ✨ Features

- ✅ **Dual-mode**: RAG (text Q&A) + Vision (image description) in one bot
- ✅ **Smart caching**: Identical queries use cached answers (marked ⚡)
- ✅ **Source attribution**: Shows which document was used for each answer
- ✅ **Message history**: Per-user conversation history (last 3 interactions)
- ✅ **Graceful fallbacks**: Claude API → Ollama → Extractive snippets
- ✅ **Direct photo support**: Send photos without needing `/image` command
- ✅ **Uncompressed image support**: Handles images sent as documents too
- ✅ **Docker-ready**: Full Docker Compose setup included

---

## 🗂️ Project Structure

```
telebot/
├── src/
│   ├── bot.py              # Telegram handlers (entry point)
│   ├── rag_engine.py       # Chunking, embedding, retrieval, LLM call
│   ├── vision_engine.py    # Image description (Claude Vision / BLIP)
│   └── history.py          # Per-user session history
├── knowledge_base/
│   ├── tech_faq.md
│   ├── company_policy.md
│   ├── recipes.md
│   └── ai_concepts.md
├── test_rag.py             # CLI test tool (no bot needed)
├── requirements.txt
├── .env.example
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 💡 Design Decisions

**Why SQLite for embeddings?**
Zero setup, single file, works everywhere. For >100k chunks, swap to ChromaDB or Qdrant.

**Why `all-MiniLM-L6-v2`?**
80MB, runs on CPU in <100ms per query, good quality for most use cases. No GPU required.

**Why Claude Haiku over Ollama as default?**
Faster (1–2s vs 5–30s), reliable quality, very affordable (~$0.001 per query), and no local GPU requirement. Ollama is a great alternative for privacy-sensitive deployments.

**Why chunk by characters with overlap?**
Sentence-level chunking with 80-char overlap ensures context isn't lost at boundaries, important for coherent retrieval.

**Caching strategy:**
Queries are hashed (MD5) and answers stored in SQLite. Cache is per-query-string, not semantic — fast and simple.
