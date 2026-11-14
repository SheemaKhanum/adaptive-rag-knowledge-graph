import os
import warnings
import logging

# Suppress HuggingFace and Google SDK terminal noise
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")
logging.getLogger("google.genai").setLevel(logging.ERROR)

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

# 1. Load Environment Variables
load_dotenv()

# 2. Sample Data
sample_text = """
Tim Berners-Lee invented the World Wide Web in 1989 while working at CERN. 
He was advised by David Howell during his time at Oxford University.
Geoffrey Hinton, who works at the University of Toronto, won the Turing Award in 2018.
The capital of France is Paris, a major European city.
"""

# 3. Chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = text_splitter.create_documents([sample_text])
print(f"✅ Created {len(chunks)} chunks.")

# 4. Local Hugging Face Embeddings (100% Free, Runs on CPU)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)
print("✅ FAISS Vector Store created successfully using local embeddings.")

# 5. Retrieve Relevant Chunks
query = "Who advised the person who invented the World Wide Web?"
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
retrieved_docs = retriever.invoke(query)

print("\n--- Retrieved Context ---")
context = ""
for i, doc in enumerate(retrieved_docs):
    print(f"Chunk {i+1}: {doc.page_content.strip()}")
    context += doc.page_content + " "

# 6. LLM Generation (Gemini API)
llm = ChatGoogleGenerativeAI(
    model=os.getenv("MODEL_NAME", "gemini-1.5-flash"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.0
)

prompt = f"""
Use the following context to answer the question. If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{query}

Answer:
"""

print("\n--- LLM Answer ---")
response = llm.invoke(prompt)

# Clean text extraction
if isinstance(response.content, list):
    clean_answer = response.content[0].get('text', '')
else:
    clean_answer = response.content

print(f"Final Answer: {clean_answer}")