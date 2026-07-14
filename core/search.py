"""Semantic search over extracted contract clauses using embeddings.

Uses OpenAI's text-embedding-3-small model to create vector representations
of extracted clauses and enables natural language search queries.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import openai

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"


def _get_embedding(text: str, model: str = EMBEDDING_MODEL) -> List[float]:
    """Get embedding vector for a text string.

    Args:
        text: Text to embed.
        model: Embedding model name.

    Returns:
        List of floats representing the embedding vector.
    """
    client = openai.OpenAI()
    text = text.replace("\n", " ").strip()
    if not text:
        return []

    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def build_search_index(
    results: List[Dict[str, str]],
) -> Tuple[List[Dict], np.ndarray]:
    """Build a search index from extraction results.

    Creates embeddings for all non-empty clause extractions.

    Args:
        results: List of contract processing results.

    Returns:
        Tuple of (metadata_list, embeddings_matrix).
        metadata_list contains dicts with contract_id, clause_type, and text.
    """
    clause_types = ["termination_clause", "confidentiality_clause", "liability_clause", "summary"]

    metadata = []
    texts_to_embed = []

    for result in results:
        for clause_type in clause_types:
            text = result.get(clause_type, "")
            if text and text not in ("Not found", "Extraction failed"):
                metadata.append({
                    "contract_id": result["contract_id"],
                    "clause_type": clause_type,
                    "text": text,
                })
                texts_to_embed.append(text)

    if not texts_to_embed:
        logger.warning("No clauses found to build search index.")
        return [], np.array([])

    logger.info("Building search index with %d clause entries...", len(texts_to_embed))

    # Batch embed for efficiency
    client = openai.OpenAI()
    batch_size = 100
    all_embeddings = []

    for i in range(0, len(texts_to_embed), batch_size):
        batch = [t.replace("\n", " ").strip()[:8000] for t in texts_to_embed[i:i + batch_size]]
        response = client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    embeddings_matrix = np.array(all_embeddings)
    logger.info("Search index built: %d entries, %d dimensions.", *embeddings_matrix.shape)

    return metadata, embeddings_matrix


def search(
    query: str,
    metadata: List[Dict],
    embeddings: np.ndarray,
    top_k: int = 5,
) -> List[Dict]:
    """Search the clause index with a natural language query.

    Args:
        query: Natural language search query.
        metadata: List of metadata dicts from build_search_index.
        embeddings: Numpy array of embeddings from build_search_index.
        top_k: Number of top results to return.

    Returns:
        List of result dicts with keys: contract_id, clause_type,
        similarity, text_snippet.
    """
    if len(metadata) == 0 or embeddings.size == 0:
        logger.warning("Search index is empty.")
        return []

    query_embedding = np.array(_get_embedding(query))

    similarities = np.array([
        _cosine_similarity(query_embedding, emb) for emb in embeddings
    ])

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        text = metadata[idx]["text"]
        snippet = text[:200] + "..." if len(text) > 200 else text
        results.append({
            "contract_id": metadata[idx]["contract_id"],
            "clause_type": metadata[idx]["clause_type"],
            "similarity": round(float(similarities[idx]), 4),
            "text_snippet": snippet,
        })

    return results


def save_index(metadata: List[Dict], embeddings: np.ndarray, output_dir: Path) -> None:
    """Save the search index to disk for reuse.

    Args:
        metadata: List of metadata dicts.
        embeddings: Numpy array of embeddings.
        output_dir: Directory to save the index files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "search_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    np.save(output_dir / "search_embeddings.npy", embeddings)
    logger.info("Search index saved to %s", output_dir)


def load_index(output_dir: Path) -> Tuple[List[Dict], np.ndarray]:
    """Load a previously saved search index.

    Args:
        output_dir: Directory containing saved index files.

    Returns:
        Tuple of (metadata_list, embeddings_matrix).
    """
    meta_path = output_dir / "search_metadata.json"
    emb_path = output_dir / "search_embeddings.npy"

    if not meta_path.exists() or not emb_path.exists():
        raise FileNotFoundError(
            f"Search index not found in {output_dir}. Run the pipeline first."
        )

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    embeddings = np.load(emb_path)
    logger.info("Search index loaded: %d entries.", len(metadata))

    return metadata, embeddings
