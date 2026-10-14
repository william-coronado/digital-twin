import logging
# Silence noisy Streamlit watcher warnings about missing torchvision
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

import os
import random
import html
import streamlit as st
import ollama
from ingest import sync_vector_store, SOURCE_DIR, VECTOR_DB_DIR, load_manifest, start_folder_watcher
from query_engine import QueryEngine, finalize_answer
from config_loader import load_config

config = load_config()
app_name = config["app_name"]

# Page configuration
st.set_page_config(
    page_title=f"{app_name} - Local RAG Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for modern look
st.markdown("""
<style>
    /* Gradient Main Title */
    .title-text {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(135deg, #6366F1, #3B82F6, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
    }
    .subtitle-text {
        color: #94A3B8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    /* Sidebar styling tweaks */
    .sidebar-section {
        background-color: #1E293B;
        color: #F8FAFC !important;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        border: 1px solid #334155;
    }
    .sidebar-section b, .sidebar-section span, .sidebar-section p, .sidebar-section div {
        color: #F8FAFC !important;
    }
    /* Pulse indicator */
    .pulse-green {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #10B981;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 1);
        transform: scale(1);
        animation: pulse 2s infinite;
        margin-right: 8px;
    }
    .pulse-red {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #EF4444;
        margin-right: 8px;
    }
    @keyframes pulse {
        0% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        }
        70% {
            transform: scale(1);
            box-shadow: 0 0 0 10px rgba(16, 185, 129, 0);
        }
        100% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
        }
    }
    /* Citation block */
    .citation-card {
        background-color: #0F172A;
        border-left: 4px solid #6366F1;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 4px;
        font-size: 0.9rem;
        color: #E2E8F0;
    }
    .citation-header {
        font-weight: 700;
        color: #818CF8;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)


# --- System Status & Connection Check ---
@st.cache_data(ttl=5)
def check_ollama_status():
    """Checks if Ollama is running and returns installed model names."""
    try:
        models_data = ollama.list()
        if hasattr(models_data, 'models'):
            models = [m.model for m in models_data.models]
        elif isinstance(models_data, dict):
            models = [m['name'] for m in models_data.get('models', [])]
        else:
            models = []
        return True, models
    except Exception as e:
        return False, []


ollama_online, available_models = check_ollama_status()


def _split_model_name(name: str) -> tuple:
    if ":" in name:
        repo, tag = name.split(":", 1)
    else:
        repo, tag = name, "latest"
    return repo, tag


def is_model_installed(config_model: str, installed_models: list) -> bool:
    """Exact (repository, tag) match, treating an absent tag as 'latest'
    (Ollama's own convention). Avoids fuzzy substring matching, which could
    false-positive e.g. 'command-r' matching an installed 'command-r-plus'
    (a different, larger model)."""
    cfg_repo, cfg_tag = _split_model_name(config_model)
    for m in installed_models:
        m_repo, m_tag = _split_model_name(m)
        if cfg_repo == m_repo and cfg_tag == m_tag:
            return True
    return False


# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Watcher Initialization
@st.cache_resource(show_spinner="Indexing documents (the first run on a large folder can take several minutes)...")
def _init_watcher():
    """Runs once per server process (not per session) — st.cache_resource
    prevents one leaked Observer thread per browser tab/session, which is
    what a st.session_state gate here would otherwise cause."""
    sync_vector_store(verbose=False)

    def on_folder_change():
        sync_vector_store(verbose=False)

    return start_folder_watcher(on_folder_change)


try:
    _init_watcher()
except Exception as e:
    st.sidebar.error(f"Watcher error: {e}")

# Load active config models from config.json
config = load_config()
target_models = config["available_llm_models"]
default_llm = config["default_llm_model"]

installed = {m: is_model_installed(m, available_models) for m in target_models}

# Default to default_llm if pulled, else the first pulled model
if installed.get(default_llm):
    default_idx = target_models.index(default_llm)
else:
    default_idx = next((i for i, m in enumerate(target_models) if installed[m]), 0)

# Initialize Query Engine
if "query_engine" not in st.session_state:
    selected_model_clean = target_models[default_idx] if target_models else default_llm
    st.session_state.query_engine = QueryEngine(model_name=selected_model_clean)
    st.session_state.active_model = selected_model_clean


# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.markdown(f"### 🧠 {app_name} Console")
    
    # 1. Connection Status
    if ollama_online:
        st.markdown('<div class="sidebar-section"><span class="pulse-green"></span><b>Ollama Server:</b> Online</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="sidebar-section"><span class="pulse-red"></span><b>Ollama Server:</b> Offline / Connecting</div>', unsafe_allow_html=True)
        st.error("Please ensure Ollama app is running locally on your Mac.")
        
    # 2. Vector DB Status & Sync
    manifest = load_manifest()
    indexed_files = list(manifest.get("files", {}).keys())
    
    st.markdown("#### 📦 Vector Database")
    st.caption(f"Storage Path: `{os.path.abspath(VECTOR_DB_DIR)}`")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Documents", len(indexed_files))
    with col2:
        st.metric("Chunks", manifest.get("total_chunks", 0))
        
    failed_files = manifest.get("failed", {})
    if failed_files:
        with st.expander(f"⚠️ {len(failed_files)} file(s) failed to index"):
            for file, error in failed_files.items():
                st.markdown(f"**{file}**: {error}")
            st.caption("They're retried on the next sync.")

    if st.button("🔄 Sync Documents Now", width="stretch"):
        with st.spinner(f"Syncing files in `{SOURCE_DIR}`..."):
            manifest = sync_vector_store()
        failed = manifest.get("failed", {})
        # Shown after the rerun below, which refreshes the counts above
        if failed:
            details = "\n".join(f"- **{f}**: {err}" for f, err in failed.items())
            st.session_state.sync_notice = ("warning", f"Synced, but {len(failed)} item(s) failed (retried on the next sync):\n{details}")
        else:
            st.session_state.sync_notice = ("success", "Synchronized successfully!")
        st.rerun()
    if "sync_notice" in st.session_state:
        kind, text = st.session_state.pop("sync_notice")
        getattr(st, kind)(text)

    # 3. Model Configuration
    st.markdown("#### 🤖 Model Configuration")
    # The key keeps the selection when a model is pulled while the app runs:
    # labels and default_idx change then, which resets an unkeyed widget
    # (key-based identity needs Streamlit 1.50+)
    selected_model = st.selectbox(
        "Select Local LLM:",
        target_models,
        index=default_idx,
        key="llm_model",
        format_func=lambda m: f"{m} ({'✅ Installed' if installed[m] else '📥 Needs Pulling'})",
    )

    # Update model in Query Engine if changed
    if selected_model != st.session_state.active_model:
        st.session_state.query_engine.change_model(selected_model)
        st.session_state.active_model = selected_model
        st.toast(f"Switched model to: {selected_model}")
        
    # Help message if the model needs pulling
    if not installed[selected_model]:
        st.warning(f"Model `{selected_model}` is not pulled. Run `ollama pull {selected_model}` in your terminal to download it locally.")

    # 4. Source Documents List
    st.markdown("#### 📄 Document Library")
    st.caption(f"Drop files into: `{os.path.abspath(SOURCE_DIR)}`")
    if indexed_files:
        for file in indexed_files:
            file_chunks = manifest["files"][file].get("chunks_count", 0)
            st.markdown(f"📄 **{file}** *({file_chunks} chunks)*")
    else:
        st.info(f"No documents indexed. Drop `.txt`, `.md`, `.pdf`, `.docx`, `.xlsx`, `.csv`, or `.html` into `{SOURCE_DIR}` to start.")


# --- MAIN CHAT PANEL ---
st.markdown(f'<div class="title-text">🧠 {app_name}</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Privacy-first Local RAG Knowledge Engine running on Apple Silicon</div>', unsafe_allow_html=True)

# Welcome instructions if history is empty
if not st.session_state.messages:
    st.markdown(f"""
    ### Welcome to your {app_name}! 👋
    This interface connects directly to your local file repository and serves a Retrieval-Augmented Generation (RAG) loop using models running 100% on your device.
    
    **How to use:**
    1. Place files inside `{os.path.abspath(SOURCE_DIR)}`.
    2. Click **Sync Documents Now** in the sidebar (or let the automatic background watcher sync it).
    3. Ask questions in the chat bar below. The system will retrieve matching facts and reply.
    
    *Privacy Guarantee: No internet request is made. All file parsing, vector embeddings, and LLM completions are kept local.*
    """)
    
    def draw_suggestions():
        prompts = config["suggested_prompts"]
        st.session_state.active_suggestions = random.sample(prompts, min(2, len(prompts)))

    if "active_suggestions" not in st.session_state:
        draw_suggestions()

    # Suggested queries cards
    st.markdown("#### Try these sample queries:")
    for col, (i, suggestion) in zip(st.columns(2), enumerate(st.session_state.active_suggestions)):
        with col:
            if st.button(suggestion, key=f"sug_{i}", width="stretch"):
                st.session_state.messages.append({"role": "user", "content": suggestion})
                st.rerun()

    if st.button("🔄 Shuffle Suggestions", key="shuffle_sug", help="Draw different questions from your config file"):
        draw_suggestions()
        st.rerun()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("rewritten"):
            st.caption(f"Searched for: {msg['rewritten']}")
        if msg.get("error"):
            st.error(msg["content"])
        else:
            st.markdown(msg["content"])

        # Display source citations if available
        if msg["role"] == "assistant" and msg.get("citations"):
            with st.expander("📚 View Local Source Citations"):
                for idx, citation in enumerate(msg["citations"]):
                    page_lbl = f", Page {citation['page']}" if citation.get("page") else ""
                    safe_source = html.escape(str(citation["source"]))
                    safe_content = html.escape(citation["content"])
                    st.markdown(
                        f"""<div class="citation-card">
                            <div class="citation-header">[{idx + 1}] Source: {safe_source}{page_lbl}</div>
                            <div>{safe_content}</div>
                        </div>""",
                        unsafe_allow_html=True
                    )

# Check if the last message in history is a user query that needs a response
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    engine = st.session_state.query_engine
    user_query = st.session_state.messages[-1]["content"]
    history = [(m["role"], m["content"]) for m in st.session_state.messages[:-1] if not m.get("error")]
    reply = {"role": "assistant", "citations": []}
    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving local facts..."):
                question = engine.standalone_question(user_query, history)
                if question != user_query:
                    reply["rewritten"] = question
                chunks, citations = engine.query_stream(question)
        except Exception as e:
            reply.update(content=f"Error querying local vector store or rewriting the question: {e}", error=True)
        else:
            try:
                # Stream for responsiveness; the stored (finalized) reply is what
                # shows after the rerun below
                streamed = st.write_stream(chunks)
                raw_answer = streamed if isinstance(streamed, str) else "".join(map(str, streamed))
                reply["content"], reply["citations"] = finalize_answer(raw_answer, citations)
            except Exception as e:
                reply.update(content=f"Error communicating with local LLM server: {e}", error=True)

    st.session_state.messages.append(reply)
    # Rerun to render the reply from history (with citations) and settle state
    st.rerun()

# Chat Input
if prompt := st.chat_input("Ask your local document twin..."):
    # Append user query and trigger rerun so the generator processes it
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()
