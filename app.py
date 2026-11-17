import streamlit as st
import streamlit.components.v1 as components
import time
import io
from pypdf import PdfReader
from dotenv import load_dotenv

from router import route_query
from main import (
    process_corpus,
    execute_adaptive_rag,
    execute_vector_baseline,
    generate_graph_html,
)

load_dotenv()

st.set_page_config(
    page_title="Adaptive RAG - Multi-Document Reasoning Engine",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Adaptive RAG with Dynamic Knowledge Graph Fusion")
st.caption("Upload any .txt or .pdf documents to dynamically route between Dense Vector Search (FAISS) and Knowledge Graph Traversal (NetworkX)")

DEFAULT_CORPUS = """
Tim Berners-Lee invented the World Wide Web in 1989 while working at CERN. 
He was advised by David Howell during his time at Oxford University.
Geoffrey Hinton, who works at the University of Toronto, won the Turing Award in 2018.
The capital of France is Paris, a major European city.
"""

def extract_text_from_file(uploaded_file) -> str:
    """Extracts text safely from any PDF or TXT file using BytesIO buffering."""
    if uploaded_file.name.endswith(".pdf"):
        try:
            uploaded_file.seek(0)
            pdf_bytes = io.BytesIO(uploaded_file.read())
            pdf_reader = PdfReader(pdf_bytes, strict=False)
            text = ""
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            return text
        except Exception as e:
            st.warning(f"⚠️ Error reading {uploaded_file.name}: {e}")
            return ""
    else:
        uploaded_file.seek(0)
        return uploaded_file.read().decode("utf-8")

# --- SIDEBAR: UNIVERSAL DOCUMENT INGESTION & CONTROLS ---
with st.sidebar:
    st.header("📂 1. Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload Any Documents (.txt or .pdf)",
        type=["txt", "pdf"],
        accept_multiple_files=True,
    )

    file_names = []
    if uploaded_files:
        combined_text = ""
        for file in uploaded_files:
            extracted = extract_text_from_file(file)
            if extracted.strip():
                combined_text += f"\n--- Start of {file.name} ---\n" + extracted + "\n"
                file_names.append(file.name)
        raw_text = combined_text
        st.success(f"Indexed {len(file_names)} file(s):\n" + "\n".join([f"• `{n}`" for n in file_names]))
    else:
        raw_text = DEFAULT_CORPUS
        st.info("Upload any .txt or .pdf files to build a live Vector + Knowledge Graph index.")

    st.divider()
    st.header("⚙️ 2. Evaluation Controls")
    compare_baseline = st.checkbox("Side-by-Side Baseline Vector RAG", value=True)
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Show medical quick-presets ONLY if the medical demo files are uploaded
    preset_query = None
    if any("cardiology" in name.lower() or "endocrinology" in name.lower() for name in file_names):
        st.divider()
        st.header("🎯 3. Clinical Demo Presets")
        if st.button("⚡ 1-Hop Query (VECTOR)", use_container_width=True):
            preset_query = "What causes systemic dyspnea in patients taking Ticagrelor?"
        if st.button("🧠 3-File Causal Chain (GRAPH)", use_container_width=True):
            preset_query = "How can an ACE inhibitor prescribed after cardiac stent placement lead to seizures in a diabetic patient?"

# Dynamic Re-indexing automatically triggers whenever uploaded files change
if "current_corpus" not in st.session_state or st.session_state.current_corpus != raw_text:
    with st.spinner("Indexing uploaded documents (FAISS Vector Store + NetworkX Graph)..."):
        t_idx = time.time()
        retriever, graph = process_corpus(raw_text)
        st.session_state.retriever = retriever
        st.session_state.graph = graph
        st.session_state.index_time = round(time.time() - t_idx, 2)
        st.session_state.current_corpus = raw_text
        st.session_state.messages = []  # Reset chat cleanly for new documents

# --- TOP TELEMETRY BAR ---
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("📄 Active Files", len(file_names) if file_names else 1)
col_m2.metric("🔵 Graph Nodes", st.session_state.graph.number_of_nodes())
col_m3.metric("🔴 Graph Edges", st.session_state.graph.number_of_edges())
col_m4.metric("⚡ Index Time", f"{st.session_state.get('index_time', 0.0)} s")
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Universal Chat Input for ANY user/panel query across ANY uploaded files
typed_query = st.chat_input("Ask any single-hop or cross-document multi-hop question...")
user_query = preset_query if preset_query else typed_query

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        start_time = time.time()
        assigned_route = route_query(user_query)

        answer = execute_adaptive_rag(
            user_query,
            st.session_state.current_corpus,
            st.session_state.retriever,
            st.session_state.graph,
            route=assigned_route,
        )

        elapsed_time = time.time() - start_time
        st.markdown(answer)

        if assigned_route == "GRAPH":
            st.success(f"🔀 Routed to: `KNOWLEDGE GRAPH (GRAPH)` | ⏱️ {elapsed_time:.2f}s")
            with st.expander("🕸️ Interactive Cross-Document Knowledge Graph (Pyvis)", expanded=True):
                st.caption("Drag nodes to inspect extracted relationships (Subject → Predicate → Object):")
                graph_html = generate_graph_html(st.session_state.graph)
                components.html(graph_html, height=420, scrolling=False)
        else:
            st.info(f"🔀 Routed to: `DENSE VECTOR STORE (VECTOR)` | ⏱️ {elapsed_time:.2f}s")

        if compare_baseline and assigned_route == "GRAPH":
            with st.expander("🔍 Baseline Comparison: Adaptive RAG vs. Standard Vector RAG", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("#### ✅ Proposed Adaptive RAG (Graph)")
                    st.write(answer)
                with c2:
                    st.markdown("#### ❌ Baseline Vector RAG (Top-k FAISS)")
                    b_ans = execute_vector_baseline(user_query, st.session_state.retriever)
                    st.write(b_ans)

        st.session_state.messages.append({"role": "assistant", "content": answer})