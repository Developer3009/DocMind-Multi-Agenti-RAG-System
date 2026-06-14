"""
DocMind — Multi-Agent RAG System
Powered by Ollama · LangChain · ChromaDB · Streamlit
"""

import os
import sys

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st

# 1. Core Configuration (Must be the first Streamlit command)
st.set_page_config(
    page_title="DocMind Multi-Agent Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. UI Override: Injecting Custom CSS for a High-End Look
st.markdown("""
    <style>
    /* Dark tactical background */
    .stApp {
        background-color: #0b0f19;
    }
    /* Sleek chat message containers */
    [data-testid="stChatMessage"] {
        background-color: #151a28;
        border: 1px solid #2a3441;
        border-radius: 6px;
        padding: 1rem;
    }
    /* Emphasizing metrics */
    [data-testid="stMetricValue"] {
        color: #00ffcc;
    }
    </style>
""", unsafe_allow_html=True)

# 3. The Sidebar: System Controls & Telemetry
with st.sidebar:
    st.title("⚙️ System Telemetry")
    
    # Simulating backend status checks
    col1, col2 = st.columns(2)
    col1.metric("Vector DB", "Online", delta="Chroma", delta_color="normal")
    col2.metric("Agent State", "Idle", delta="Awaiting Input", delta_color="off")
    
    st.divider()
    
    st.markdown("### Inference Parameters")
    temperature = st.slider("Creativity (Temperature)", 0.0, 1.0, 0.2)
    search_k = st.number_input("RAG Context Windows (k)", min_value=1, max_value=10, value=4)
    
    if st.button("Purge System Memory", type="primary"):
        st.session_state.messages = []
        st.rerun()

# 4. Main Interface: The Chat Terminal
st.title("🧠 DocMind Intel Interface")
st.caption("Multi-Agent Document Retrieval and Synthesis")

# Initialize conversation history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "System initialized. Vector space loaded. How can the agents assist you?"}]

# Render existing conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 5. The Input & Execution Loop
if prompt := st.chat_input("Enter query for the agent swarm..."):
    # Append and display user input
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Trigger the multi-agent pipeline
    with st.chat_message("assistant"):
        with st.spinner("Agents are analyzing the vector space..."):
            
            # --- INTEGRATION POINT ---
            # This is where you pass `prompt`, `temperature`, and `search_k` 
            # into your Langchain/Chroma multi-agent pipeline.
            # response = my_rag_pipeline.run(prompt)
            
            # Placeholder response
            simulated_response = f"**Agent Synthesis:** Based on the {search_k} retrieved documents, the process is successfully executing."
            
            st.markdown(simulated_response)
            st.session_state.messages.append({"role": "assistant", "content": simulated_response})

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from agents.registry import AgentRegistry, AGENT_ICONS, AGENT_DESCRIPTIONS
from utils.indexer import build_retriever
from utils.history import init_history, add_message, get_history, clear_history

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind — Multi-Agent RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    font-family: 'Inter', sans-serif;
    background: #07090f;
    color: #e8eaf0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1e2433;
}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: #8b94a8 !important;
    font-size: 0.82rem;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #e8eaf0 !important;
}

/* ── Sidebar inputs ── */
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stSelectbox select,
[data-testid="stSidebar"] [data-baseweb="select"] {
    background: #161b27 !important;
    border: 1px solid #2a3347 !important;
    color: #e8eaf0 !important;
    border-radius: 6px !important;
    font-size: 0.85rem !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #161b27;
    border: 1.5px dashed #2a3347;
    border-radius: 10px;
    padding: 0.75rem;
}
[data-testid="stFileUploader"]:hover {
    border-color: #4f6ef7;
}

/* ── Main area ── */
.main .block-container {
    padding: 1.5rem 2rem 3rem 2rem;
    max-width: 960px;
    margin: 0 auto;
}

/* ── Logo / header ── */
.docmind-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 0.2rem;
}
.docmind-logo {
    font-size: 2rem;
    line-height: 1;
}
.docmind-title {
    font-size: 1.7rem;
    font-weight: 700;
    color: #e8eaf0;
    letter-spacing: -0.03em;
    margin: 0;
}
.docmind-sub {
    font-size: 0.875rem;
    color: #4f5a72;
    margin: 0 0 1.5rem 0;
}

/* ── Divider ── */
.section-divider {
    border: none;
    border-top: 1px solid #1e2433;
    margin: 1rem 0;
}

/* ── Agent badge ── */
.agent-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    margin-right: 4px;
}
.badge-rag        { background: #1a2e4a; color: #60a5fa; border: 1px solid #2563eb44; }
.badge-summary    { background: #1a3a2a; color: #4ade80; border: 1px solid #16a34a44; }
.badge-comparison { background: #2d1a3a; color: #c084fc; border: 1px solid #9333ea44; }
.badge-analyst    { background: #3a2a1a; color: #fb923c; border: 1px solid #ea580c44; }
.badge-general    { background: #1a2d3a; color: #38bdf8; border: 1px solid #0284c744; }

/* ── Chat messages ── */
.stChatMessage {
    background: transparent !important;
    border: none !important;
}
[data-testid="stChatMessageContent"] {
    font-size: 0.9rem;
    line-height: 1.65;
    color: #d4d8e8;
}

/* ── User bubble ── */
[data-testid="stChatMessage"][data-testid*="user"] [data-testid="stChatMessageContent"],
.stChatMessage:has([aria-label="user avatar"]) [data-testid="stChatMessageContent"] {
    background: #161e31 !important;
    border: 1px solid #1e2a45 !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
}

/* ── Assistant bubble ── */
.stChatMessage:has([aria-label="assistant avatar"]) [data-testid="stChatMessageContent"] {
    background: #0f1420 !important;
    border: 1px solid #1a2035 !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: #0d1117 !important;
    border: 1.5px solid #2a3347 !important;
    border-radius: 12px !important;
    color: #e8eaf0 !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #4f6ef7 !important;
    box-shadow: 0 0 0 3px #4f6ef722 !important;
}

/* ── Source dataframe ── */
.stDataFrame {
    border: 1px solid #1e2433 !important;
    border-radius: 8px !important;
    overflow: hidden;
    font-size: 0.8rem;
}

/* ── Stat card ── */
.stat-row {
    display: flex;
    gap: 10px;
    margin: 0.5rem 0;
}
.stat-card {
    flex: 1;
    background: #0f1520;
    border: 1px solid #1e2a3a;
    border-radius: 8px;
    padding: 10px 14px;
    text-align: center;
}
.stat-card .stat-value {
    font-size: 1.3rem;
    font-weight: 700;
    color: #4f6ef7;
    font-family: 'JetBrains Mono', monospace;
}
.stat-card .stat-label {
    font-size: 0.7rem;
    color: #4f5a72;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Routing info box ── */
.routing-box {
    background: #0d1520;
    border: 1px solid #1e2d45;
    border-left: 3px solid #4f6ef7;
    border-radius: 6px;
    padding: 8px 12px;
    margin-bottom: 10px;
    font-size: 0.78rem;
    color: #6b7fa8;
}

/* ── Spinner override ── */
.stSpinner > div {
    border-top-color: #4f6ef7 !important;
}

/* ── Agent panel ── */
.agent-panel {
    background: #0d1117;
    border: 1px solid #1e2433;
    border-radius: 10px;
    padding: 12px;
    margin-bottom: 8px;
}
.agent-panel-title {
    font-size: 0.8rem;
    font-weight: 600;
    color: #e8eaf0;
    margin-bottom: 6px;
}

/* ── Buttons ── */
.stButton > button {
    background: #161e31 !important;
    color: #8b94a8 !important;
    border: 1px solid #2a3347 !important;
    border-radius: 7px !important;
    font-size: 0.8rem !important;
    padding: 4px 12px !important;
    transition: all 0.15s;
}
.stButton > button:hover {
    border-color: #4f6ef7 !important;
    color: #e8eaf0 !important;
}

/* ── Welcome card ── */
.welcome-card {
    background: #0d1520;
    border: 1px solid #1e2d45;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    margin: 2rem 0;
}
.welcome-card h3 { color: #e8eaf0; font-size: 1.1rem; margin: 0.5rem 0; }
.welcome-card p  { color: #4f5a72; font-size: 0.85rem; margin: 0; }
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
init_history()
if "registry" not in st.session_state:
    st.session_state.registry = None
if "index_stats" not in st.session_state:
    st.session_state.index_stats = None
if "indexed_key" not in st.session_state:
    st.session_state.indexed_key = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 DocMind")
    st.markdown("---")

    # ── Model config ──
    st.markdown("### ⚙️ Model Configuration")
    ollama_host = st.text_input(
        "Ollama Host",
        value=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        help="URL where your Ollama server is running.",
    )

    llm_model = st.selectbox(
        "LLM (for answers)",
        ["gemma3:latest", "gemma3:4b", "gemma3:12b", "gemma3:27b",
         "llama3.2", "llama3.2:1b", "mistral", "phi3", "qwen2.5"],
        index=0,
    )

    embed_model = st.selectbox(
        "Embedding Model (for indexing)",
        ["nomic-embed-text", "mxbai-embed-large", "all-minilm"],
        index=0,
    )

    st.markdown("---")

    # ── Agent mode ──
    st.markdown("### 🤖 Agent Mode")
    agent_mode = st.radio(
        "Routing",
        ["🎯 Auto (Orchestrator decides)", "🔧 Manual (pick agent)"],
        label_visibility="collapsed",
    )

    manual_agent = None
    if "Manual" in agent_mode:
        manual_agent = st.selectbox(
            "Select agent:",
            list(AGENT_ICONS.keys())[:-1],  # exclude 'orchestrator'
            format_func=lambda x: f"{AGENT_ICONS[x]}  {x.replace('_', ' ').title()}",
        )

    st.markdown("---")

    # ── RAG settings ──
    with st.expander("🔬 Advanced RAG Settings"):
        chunk_size    = st.slider("Chunk size (chars)", 500, 3000, 1200, 100)
        chunk_overlap = st.slider("Chunk overlap",       0,  500,  200,  50)
        top_k         = st.slider("Top-K retrieved",     1,   10,    5,   1)

    st.markdown("---")

    # ── Document upload ──
    st.markdown("### 📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Drop PDF files here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    # Index button
    index_key = tuple((f.name, f.size) for f in uploaded_files) if uploaded_files else ()
    needs_index = index_key != st.session_state.indexed_key

    if uploaded_files:
        if st.button("⚡ Index Documents", use_container_width=True, disabled=not needs_index):
            with st.spinner("Embedding documents…"):
                try:
                    retriever, stats = build_retriever(
                        uploaded_files,
                        ollama_host,
                        embed_model,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        k=top_k,
                    )
                    # Rebuild registry with current model
                    registry = AgentRegistry(model=llm_model, base_url=ollama_host)
                    registry.set_retriever(retriever)
                    st.session_state.registry = registry
                    st.session_state.index_stats = stats
                    st.session_state.indexed_key = index_key
                    clear_history()
                    st.success(f"✓ Indexed {stats['total_chunks']} chunks")
                except Exception as e:
                    st.error(f"Indexing failed: {e}")

    # ── Index stats ──
    if st.session_state.index_stats:
        stats = st.session_state.index_stats
        st.markdown("---")
        st.markdown("### 📊 Index Stats")
        st.markdown(f"""
<div class="stat-row">
  <div class="stat-card">
    <div class="stat-value">{len(stats['files'])}</div>
    <div class="stat-label">Files</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{stats['total_pages']}</div>
    <div class="stat-label">Pages</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{stats['total_chunks']}</div>
    <div class="stat-label">Chunks</div>
  </div>
</div>
""", unsafe_allow_html=True)
        for f in stats["files"]:
            st.markdown(f"📄 `{f['name']}` — {f['pages']}p · {f['size_kb']} KB")

    # ── Controls ──
    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        clear_history()
        st.rerun()

    st.markdown("---")
    st.markdown("### 🤖 Agents")
    for name, desc in AGENT_DESCRIPTIONS.items():
        icon = AGENT_ICONS[name]
        st.markdown(f"{icon} **{name.replace('_', ' ').title()}**")
        st.caption(desc)

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="docmind-header">
  <span class="docmind-logo">🧠</span>
  <h1 class="docmind-title">DocMind</h1>
</div>
<p class="docmind-sub">Multi-agent RAG · Powered by Ollama · No API key required</p>
<hr class="section-divider">
""", unsafe_allow_html=True)

# ── Welcome state ─────────────────────────────────────────────────────────────
if not st.session_state.registry:
    st.markdown("""
<div class="welcome-card">
  <div style="font-size:3rem">📄</div>
  <h3>Upload documents to begin</h3>
  <p>Use the sidebar to upload PDFs and click <strong>Index Documents</strong>.<br>
  Then ask anything — the right agent will be selected automatically.</p>
</div>
""", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    examples = [
        ("🔍", "What are the key findings?"),
        ("📋", "Summarise this document"),
        ("⚖️", "Compare the methodologies"),
        ("🧠", "What are the implications?"),
    ]
    for col, (icon, ex) in zip([col1, col2, col3, col4], examples):
        with col:
            st.markdown(f"""
<div style="background:#0d1520;border:1px solid #1e2d45;border-radius:10px;
            padding:12px;text-align:center;font-size:0.8rem;color:#6b7fa8;">
  <div style="font-size:1.5rem">{icon}</div>
  <div style="margin-top:6px;color:#8b94a8">{ex}</div>
</div>""", unsafe_allow_html=True)

# ── Render chat history ───────────────────────────────────────────────────────
def badge(agent_name: str) -> str:
    cls_map = {
        "rag_agent": "badge-rag", "summary_agent": "badge-summary",
        "comparison_agent": "badge-comparison", "analyst_agent": "badge-analyst",
        "general_agent": "badge-general",
    }
    icon = AGENT_ICONS.get(agent_name, "🤖")
    cls  = cls_map.get(agent_name, "badge-rag")
    label = agent_name.replace("_agent", "").title()
    return f'<span class="agent-badge {cls}">{icon} {label}</span>'


for msg in get_history():
    with st.chat_message(msg.role):
        if msg.role == "assistant" and msg.agent_used:
            badges = "".join(badge(a) for a in msg.agent_used)
            if msg.routing_reason:
                st.markdown(
                    f'<div class="routing-box">🎯 Routed to: {badges} — {msg.routing_reason}</div>',
                    unsafe_allow_html=True,
                )
        st.markdown(msg.content)
        if msg.sources:
            with st.expander(f"📑 Sources ({len(msg.sources)})"):
                import pandas as pd
                st.dataframe(pd.DataFrame(msg.sources), use_container_width=True)

# ── Chat input ────────────────────────────────────────────────────────────────
registry: AgentRegistry | None = st.session_state.registry

if user_input := st.chat_input(
    "Ask anything about your documents…" if registry else "Index documents first to start chatting",
    disabled=not registry,
):
    add_message("user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        # ── Route ──
        if "Manual" in agent_mode and manual_agent:
            routing = {"agents": [manual_agent], "reason": "Manually selected."}
        else:
            with st.spinner("🎯 Routing query…"):
                routing = registry.route(user_input)

        agents_to_use = routing.get("agents", ["rag_agent"])
        reason = routing.get("reason", "")

        # Show routing info
        badges_html = "".join(badge(a) for a in agents_to_use)
        st.markdown(
            f'<div class="routing-box">🎯 Routed to: {badges_html} — {reason}</div>',
            unsafe_allow_html=True,
        )

        all_sources = []

        if len(agents_to_use) == 1:
            # ── Single agent — stream directly ──
            agent_name = agents_to_use[0]
            stream_box = st.empty()
            full_text  = ""

            for token, sources in registry.stream_agent(agent_name, user_input):
                if sources is not None:
                    all_sources.extend(sources)
                else:
                    full_text += token
                    stream_box.markdown(full_text + "▌")

            stream_box.markdown(full_text)

        else:
            # ── Multi-agent — run each, then synthesise ──
            agent_outputs = {}
            for agent_name in agents_to_use:
                with st.expander(f"{AGENT_ICONS.get(agent_name, '🤖')} {agent_name.replace('_', ' ').title()}", expanded=True):
                    box = st.empty()
                    text = ""
                    for token, sources in registry.stream_agent(agent_name, user_input):
                        if sources is not None:
                            all_sources.extend(sources)
                        else:
                            text += token
                            box.markdown(text + "▌")
                    box.markdown(text)
                    agent_outputs[agent_name] = text

            st.markdown("---")
            st.markdown("**🔀 Synthesised Answer:**")
            with st.spinner("Combining agent outputs…"):
                full_text = registry.synthesise(user_input, agent_outputs)
            st.markdown(full_text)

        # ── Show sources ──
        if all_sources:
            with st.expander(f"📑 Sources ({len(all_sources)})"):
                import pandas as pd
                st.dataframe(pd.DataFrame(all_sources), use_container_width=True)

        add_message(
            "assistant",
            full_text,
            agent_used=agents_to_use,
            sources=all_sources,
            routing_reason=reason,
        )
