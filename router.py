import os
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

router_llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.0,
    max_retries=1,
    timeout=8,
)

ROUTER_PROMPT = PromptTemplate.from_template(
    """You are a Query Complexity Router for a Retrieval-Augmented Generation (RAG) system.
Classify the user's query into exactly one of three retrieval strategies based ONLY on its structural reasoning requirement:

1. VECTOR: Direct, single-hop factual lookups, definitions, or questions about an isolated fact in one local paragraph.
2. GRAPH: Multi-hop reasoning, causal chains (A -> B -> C), cross-document entity relationships, or comparing entities.
3. FULL_TEXT: Counting specific words/terms/occurrences, word frequency, or scanning the entire text corpus verbatim.

Respond with ONLY one word: VECTOR, GRAPH, or FULL_TEXT.

User Query: {query}
Route:"""
)


def route_query(query: str) -> str:
    """Three-way Domain-Agnostic Complexity Router: VECTOR, GRAPH, or FULL_TEXT."""
    q_lower = query.lower().strip()
    words = q_lower.split()

    # Tier 1a: Lexical / word-counting queries -> FULL_TEXT (<0.1 ms)
    if re.search(r"\b(how many times|count of|word count|frequency of|occurrences of|number of times)\b", q_lower):
        return "FULL_TEXT"

    # Tier 1b: Ultra-short, single-clause factual lookup -> VECTOR (<0.1 ms)
    clause_markers = len(
        re.findall(r"\b(and|or|which|whose|that|when|after|before|while|because|how|between)\b", q_lower)
    )
    if len(words) <= 6 and clause_markers == 0:
        return "VECTOR"

    # Tier 2: Zero-shot structural reasoning classification
    try:
        chain = ROUTER_PROMPT | router_llm
        response = chain.invoke({"query": query})
        content = response.content[0].get("text", "") if isinstance(response.content, list) else str(response.content)
        decision = content.strip().upper()
        if "FULL_TEXT" in decision:
            return "FULL_TEXT"
        if "GRAPH" in decision:
            return "GRAPH"
        return "VECTOR"
    except Exception:
        return "GRAPH"