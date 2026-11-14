import os
import time
import json
import warnings
import logging
import pandas as pd

# Suppress terminal noise
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")
logging.getLogger("google.genai").setLevel(logging.ERROR)

from dotenv import load_dotenv
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

# Benchmark Dataset with Ground Truth
eval_dataset = [
    {
        "query": "What is the capital of France?",
        "type": "Single-Hop",
        "ground_truth": "Paris"
    },
    {
        "query": "When was Geoffrey Hinton born?",
        "type": "Single-Hop",
        "ground_truth": "2018"  # Context mentions Turing Award year 2018
    },
    {
        "query": "Who advised the person who invented the World Wide Web?",
        "type": "Multi-Hop",
        "ground_truth": "David Howell"
    },
    {
        "query": "Which institution did the inventor of the World Wide Web attend?",
        "type": "Multi-Hop",
        "ground_truth": "Oxford University"
    }
]

corpus_text = """
Tim Berners-Lee invented the World Wide Web in 1989 while working at CERN. 
He was advised by David Howell during his time at Oxford University.
Geoffrey Hinton, who works at the University of Toronto, won the Turing Award in 2018.
The capital of France is Paris, a major European city.
"""

# Initialize FAISS and Graph once for benchmarking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20)
chunks = text_splitter.create_documents([corpus_text])
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

triplets = extract_triplets(corpus_text)
graph = build_knowledge_graph(triplets)
graph_context = extract_graph_context(graph)

def evaluate_response(query: str, context: str, ground_truth: str) -> dict:
    """Invokes LLM and checks if ground truth is contained in the response."""
    prompt = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer concisely:"
    start_time = time.time()
    response = llm.invoke(prompt)
    latency = round(time.time() - start_time, 2)
    
    answer = response.content[0].get('text', '') if isinstance(response.content, list) else str(response.content)
    answer_clean = answer.strip()
    
    accuracy = 1 if ground_truth.lower() in answer_clean.lower() else 0
    return {"answer": answer_clean, "accuracy": accuracy, "latency": latency}

def run_benchmark():
    results = []
    print("🚀 Starting Empirical Evaluation for Defense Benchmark...\n")

    for item in eval_dataset:
        q = item["query"]
        q_type = item["type"]
        gt = item["ground_truth"]
        
        print(f"Testing Query: '{q}' [{q_type}]")
        
        # 1. Baseline Vector RAG
        v_docs = retriever.invoke(q)
        v_ctx = " ".join([d.page_content.strip() for d in v_docs])
        v_eval = evaluate_response(q, v_ctx, gt)
        
        # 2. Adaptive Hybrid RAG
        route = route_query(q)
        a_ctx = v_ctx if route == "VECTOR" else graph_context
        a_eval = evaluate_response(q, a_ctx, gt)
        
        results.append({
            "Query": q,
            "Type": q_type,
            "Route": route,
            "Vector Acc": v_eval["accuracy"],
            "Adaptive Acc": a_eval["accuracy"],
            "Adaptive Latency (s)": a_eval["latency"]
        })

    df = pd.DataFrame(results)
    print("\n" + "="*70)
    print("📊 BENCHMARK SUMMARY TABLE")
    print("="*70)
    print(df.to_string(index=False))
    
    vector_acc = (df["Vector Acc"].mean()) * 100
    adaptive_acc = (df["Adaptive Acc"].mean()) * 100
    
    print("\n" + "="*70)
    print(f"🎯 Baseline Vector RAG Accuracy:  {vector_acc:.1f}%")
    print(f"🎯 Proposed Adaptive RAG Accuracy: {adaptive_acc:.1f}%")
    print("="*70)

if __name__ == "__main__":
    run_benchmark()