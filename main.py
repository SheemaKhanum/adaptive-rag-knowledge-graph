import os
import warnings
import logging
from dotenv import load_dotenv
import tempfile
from pyvis.network import Network

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")
logging.getLogger("google.genai").setLevel(logging.ERROR)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from router import route_query
from graph_module import extract_triplets, build_knowledge_graph, extract_graph_context

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.0
)

# Global lazy-loaded embedding model to save memory
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def process_corpus(corpus_text: str):
    """Processes uploaded text into FAISS vector store and dynamic triplet graph."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
    chunks = text_splitter.create_documents([corpus_text])
    
    # 1. Create Vector Store
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    
    # 2. Extract Triplet Graph
    triplets = extract_triplets(corpus_text)
    graph = build_knowledge_graph(triplets)
    
    return retriever, graph

def execute_adaptive_rag(user_query: str, corpus_text: str, retriever, graph) -> str:
    """Executes the adaptive pipeline using dynamically uploaded corpus objects."""
    route = route_query(user_query)
    
    if route == "VECTOR":
        docs = retriever.invoke(user_query)
        context = " ".join([d.page_content.strip() for d in docs])
    else:
        context = extract_graph_context(graph)

    prompt = f"""
Use the provided context to answer the question accurately and concisely.Focus strictly on the cause-and-effect relationship requested.

Context:
{context}

Question:
{user_query}

Answer:"""

    response = llm.invoke(prompt)
    if isinstance(response.content, list):
        return str(response.content[0].get('text', '')).strip()
    return str(response.content).strip()

def execute_vector_baseline(user_query: str, retriever) -> str:
    """Helper function for baseline vector retrieval comparison."""
    docs = retriever.invoke(user_query)
    context = " ".join([d.page_content.strip() for d in docs])
    prompt = f"Use the context to answer accurately.\nContext:\n{context}\nQuestion:\n{user_query}\nAnswer:"
    response = llm.invoke(prompt)
    if isinstance(response.content, list):
        return str(response.content[0].get('text', '')).strip()
    return str(response.content).strip()

# -PYVIS GRAPH HTML GENERATOR ---
def generate_graph_html(graph) -> str:
    """Converts a NetworkX graph into an interactive Pyvis HTML visualization string."""
    net = Network(height="400px", width="100%", directed=True, bgcolor="#1e1e1e", font_color="white")
    
    # Load NetworkX graph into Pyvis
    net.from_nx(graph)
    
    # Customize node & edge styling for visual clarity
    for node in net.nodes:
        node["shape"] = "dot"
        node["size"] = 15
        node["color"] = "#00d2ff"
        
    for edge in net.edges:
        edge["color"] = "#ff7675"
        edge["font"] = {"size": 10, "align": "middle", "color": "#ffffff"}

    net.toggle_physics(True)
    
    # Save temporarily and return raw HTML string
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.write_html(tmp_file.name)
        with open(tmp_file.name, "r", encoding="utf-8") as f:
            html_content = f.read()
            
    return html_content