import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

router_llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.0,
)

ROUTER_PROMPT = PromptTemplate.from_template(
    """You are a Query Complexity Router for a Retrieval-Augmented Generation (RAG) system.
Classify the user's query into exactly one of two retrieval strategies based ONLY on its structural reasoning requirement:

1. VECTOR: Use for direct, single-hop factual lookups, definitions, or questions about a single entity or isolated fact.
2. GRAPH: Use for multi-hop reasoning, causal chains (A -> B -> C), cross-document synthesis, indirect relationships, or questions connecting multiple distinct entities or events.

Respond with ONLY the single word 'VECTOR' or 'GRAPH'. Do not include punctuation or explanation.

User Query: {query}
Route:"""
)


def route_query(query: str) -> str:
    """Routes incoming queries to VECTOR or GRAPH across any domain."""
    try:
        chain = ROUTER_PROMPT | router_llm
        response = chain.invoke({"query": query}).content.strip().upper()
        if "GRAPH" in response:
            return "GRAPH"
        return "VECTOR"
    except Exception:
        # Safe fallback to VECTOR on transient API issues
        return "VECTOR"