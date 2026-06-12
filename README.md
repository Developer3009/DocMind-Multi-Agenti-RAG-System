# DocMind-Multi-Agenti-RAG-System
Multi-Agent RAG System powered by Ollama, LangChain, ChromaDB, and Streamlit with intelligent query routing, specialist agents, citation-based retrieval, and reasoning synthesis.
# 🧠 DocMind — Multi-Agent RAG System

A production-quality, fully local RAG chatbot powered by **5 specialist AI agents**,
an **orchestrator router**, Ollama (Gemma 3 / LLaMA / Mistral), LangChain, ChromaDB, and Streamlit.

**No API key. No cloud. Runs 100% on your machine.**

---

## ✨ Features

| Feature | Details |
|---|---|
| 🎯 Intelligent routing | Orchestrator agent reads the query and picks the best specialist |
| 🔍 RAG Agent | Precise document QA with source citations |
| 📋 Summary Agent | Structured summaries with key topics & conclusions |
| ⚖️ Comparison Agent | Markdown comparison tables across documents |
| 🧠 Analyst Agent | Deep reasoning, pros/cons, implications |
| 💡 General Agent | Fallback for non-document questions |
| 🔀 Multi-agent synthesis | When multiple agents run, outputs are merged into one answer |
| 🌊 Streaming | Real-time token streaming for every agent |
| 📑 Source citations | Exact file + page references shown per answer |
| ⚙️ MMR retrieval | Maximal Marginal Relevance for diverse, non-redundant chunks |
| 🎛️ Manual override | Pick any agent manually from the sidebar |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│  Orchestrator Agent  │  ← reads query, picks agent(s)
│  (gemma3 / llama3)  │
└──────────┬──────────┘
           │  routes to one or more of:
    ┌──────┼──────────────────────────┐
    │      │                          │
    ▼      ▼           ▼             ▼
 RAG    Summary    Comparison    Analyst    General
Agent    Agent       Agent        Agent     Agent
    │      │           │             │        │
    └──────┴───────────┴─────────────┴────────┘
                       │
                       ▼
           ┌───────────────────┐
           │  (if multi-agent) │
           │  Synthesiser LLM  │  ← merges outputs
           └────────┬──────────┘
                    │
                    ▼
              Final Answer + Sources
```

---

## 📂 Project Structure

```
docmind/
├── app.py                        # Main Streamlit UI
├── requirements.txt
├── .streamlit/
│   └── config.toml               # Theme & server settings
├── agents/
│   ├── __init__.py
│   ├── orchestrator.py           # Routing + synthesis LLM
│   ├── rag_agent.py              # Document QA specialist
│   ├── specialist_agents.py      # Summary / Comparison / Analyst / General
│   └── registry.py              # Central agent factory
└── utils/
    ├── __init__.py
    ├── indexer.py                # PDF loading, chunking, embedding
    └── history.py                # Chat session state
```

---

## 🖥️ Prerequisites

### 1. Install Ollama
```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh
# Windows: https://ollama.com/download
```

### 2. Pull models
```bash
# LLM (pick at least one)
ollama pull gemma3          # recommended — ~3 GB
ollama pull gemma3:12b      # better quality — ~8 GB

# Embedding (required)
ollama pull nomic-embed-text
```

### 3. Start Ollama
```bash
ollama serve
```

---

## 🚀 Run Locally

```bash
# Clone
git clone https://github.com/<your-username>/docmind.git
cd docmind

# Virtual environment
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# Install
pip install -r requirements.txt

# Run
streamlit run app.py
```

Open **http://localhost:8501**

---

## 🌐 Deploy on Streamlit Cloud

Streamlit Cloud can't reach your local Ollama. You need a public Ollama endpoint.

### Option A — ngrok tunnel (quickest for demos)
```bash
# On your local machine:
ngrok http 11434
# Paste the https://xxxx.ngrok-free.app URL into the "Ollama Host" field
```

### Option B — Cloud VM (permanent)
1. Spin up any VPS (DigitalOcean, Hetzner, Railway, Fly.io, etc.)
2. Install Ollama: `curl -fsSL https://ollama.com/install.sh | sh`
3. Pull models: `ollama pull gemma3 && ollama pull nomic-embed-text`
4. Allow port 11434 in firewall: `ufw allow 11434`
5. Start with public binding:
   ```bash
   OLLAMA_HOST=0.0.0.0 ollama serve
   ```
6. Use `http://<VM_IP>:11434` as the Ollama host in the app

### Deploy steps
1. Push repo to GitHub (the `.gitignore` already excludes `chroma_db/`)
2. Go to **https://share.streamlit.io** → New App
3. Select repo, branch `main`, main file `app.py`
4. Deploy

---

## 🛠️ How to Use

1. **Upload PDFs** via the sidebar file uploader
2. Click **⚡ Index Documents** (only needed when files change)
3. **Ask anything** — the orchestrator picks the right agent automatically
4. Optionally switch to **Manual mode** to force a specific agent
5. Click **📑 Sources** under any answer to see where it came from

### Example prompts by agent

| Agent | Example prompt |
|---|---|
| 🔍 RAG | "What does the paper say about transformer attention?" |
| 📋 Summary | "Give me a structured summary of this report" |
| ⚖️ Comparison | "Compare the methods described in both documents" |
| 🧠 Analyst | "What are the strengths and weaknesses of this approach?" |
| 💡 General | "What is gradient descent?" |

---

## ⚙️ Configuration Options

| Setting | Default | Description |
|---|---|---|
| Ollama Host | `http://localhost:11434` | Where Ollama runs |
| LLM Model | `gemma3:latest` | Answering model |
| Embedding Model | `nomic-embed-text` | Indexing model |
| Chunk size | 1200 chars | Size of each text chunk |
| Chunk overlap | 200 chars | Overlap between chunks |
| Top-K | 5 | Number of chunks retrieved |

---

## 🛠️ Troubleshooting

| Issue | Fix |
|---|---|
| `Connection refused` | Run `ollama serve` |
| `model not found` | Run `ollama pull <model-name>` |
| Routing always picks RAG | Try a clearer query or use Manual mode |
| Slow first query | First run indexes — subsequent queries are cached |
| Empty sources | Check the PDF has text (not scanned images) |

---

## 📦 Tech Stack

| Layer | Library |
|---|---|
| UI | Streamlit 1.45 |
| LLM + Embeddings | Ollama |
| Orchestration | LangChain |
| Vector Store | ChromaDB (MMR search) |
| PDF Parsing | PyMuPDF |
| Agent Framework | Custom multi-agent registry |

---

## 📄 License

MIT — free to use, modify, and deploy.
