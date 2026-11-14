import os
from dotenv import load_dotenv
import faiss
import networkx as nx
import spacy
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if api_key:
    print("✅ API Key successfully loaded from .env!")
else:
    print("❌ Failed to load API Key. Check your .env file.")

print("✅ FAISS Version:", faiss.__version__)
print("✅ NetworkX Version:", nx.__version__)
print("✅ spaCy is installed successfully.")
print("✅ LangChain Google GenAI is ready.")