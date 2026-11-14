import os
import json
import networkx as nx
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

extractor_llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.0,
)

TRIPLET_PROMPT = PromptTemplate.from_template(
    """You are a general-purpose Knowledge Graph extraction engine.
Extract key factual relationships from the provided text as a JSON list of [Subject, Predicate, Object] triplets.

Rules:
1. Domain-Agnostic: Extract entities and relationships across ANY domain (technical, legal, financial, scientific, medical, organizational, etc.).
2. Canonical Naming: Keep entity names concise, standardized, and consistent across sentences so shared entities connect into a unified graph.
3. Concise Predicates: Use clear, active verb phrases for predicates (e.g., "causes", "depends on", "regulates", "acquired", "triggers", "requires").
4. Output Format: Return ONLY a valid JSON array of 3-element arrays. No markdown fences, no commentary.

Example Output:
[["Entity A", "triggers", "Event B"], ["Event B", "violates", "Policy C"]]

Text:
{text}
"""
)


def extract_triplets(corpus_text: str) -> list:
    """Extracts domain-agnostic [Subject, Predicate, Object] triplets from any text."""
    if not corpus_text or not corpus_text.strip():
        return []

    chunk_size = 4000
    chunks = [corpus_text[i : i + chunk_size] for i in range(0, len(corpus_text), chunk_size)]
    all_triplets = []

    chain = TRIPLET_PROMPT | extractor_llm

    for chunk in chunks:
        try:
            response = chain.invoke({"text": chunk})
            if isinstance(response.content, list):
                raw_output = str(response.content[0].get("text", "")).strip()
            else:
                raw_output = str(response.content).strip()

            if raw_output.startswith("```"):
                raw_output = raw_output.strip("`").replace("json", "", 1).strip()

            triplets = json.loads(raw_output)
            for item in triplets:
                if isinstance(item, list) and len(item) == 3:
                    subj = str(item[0]).strip()
                    pred = str(item[1]).strip()
                    obj = str(item[2]).strip()
                    if subj and pred and obj:
                        all_triplets.append((subj, pred, obj))
        except Exception as e:
            print(f"Triplet extraction warning on chunk: {e}")

    return all_triplets


def build_knowledge_graph(triplets: list) -> nx.DiGraph:
    """Constructs a directed NetworkX graph from extracted (Subject, Predicate, Object) triplets."""
    graph = nx.DiGraph()
    for subj, pred, obj in triplets:
        # 'title' and 'label' ensure Pyvis displays edge labels cleanly on hover and on the line
        graph.add_edge(subj, obj, label=pred, title=pred)
    return graph


def extract_graph_context(graph: nx.DiGraph) -> str:
    """Serializes all directed edges in the NetworkX graph into readable reasoning chains."""
    if graph.number_of_edges() == 0:
        return "No relational triplets extracted."

    chains = [
        f"{u} --[{data.get('label', 'related_to')}]--> {v}"
        for u, v, data in graph.edges(data=True)
    ]
    return "\n".join(chains)