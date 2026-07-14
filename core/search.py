"""Semantic search over extracted contract clauses using TF-IDF embeddings.

Uses scikit-learn-free TF-IDF vectorization with cosine similarity for
searching over extracted clauses. No external embedding API required.
"""

import json
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    """Simple tokenization: lowercase, split on non-alphanumeric."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _compute_tf(tokens: List[str]) -> Dict[str, float]:
    """Compute term frequency for a token list."""
    counts = Counter(tokens)
    total = len(tokens) if tokens else 1
    return {t: c / total for t, c in counts.items()}


def _compute_idf(documents: List[List[str]]) -> Dict[str, float]:
    """Compute inverse document frequency across documents."""
    n_docs = len(documents)
    df = Counter()
    for doc in documents:
        unique_tokens = set(doc)
        for token in unique_tokens:
            df[token] += 1

    return {t: math.log((n_docs + 1) / (freq + 1)) + 1 for t, freq in df.items()}


def _tfidf_vector(tokens: List[str], idf: Dict[str, float], vocab: List[str]) -> np.ndarray:
    """Create a TF-IDF vector for a tokenized document."""
    tf = _compute_tf(tokens)
    vec = np.zeros(len(vocab))
    for i, term in enumerate(vocab):
        vec[i] = tf.get(term, 0.0) * idf.get(term, 0.0)
    # Normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def build_search_index(
    results: List[Dict[str, str]],
) -> Tuple[List[Dict], np.ndarray, Dict[str, float], List[str]]:
    """Build a TF-IDF search index from extraction results.

    Creates TF-IDF vectors for all non-empty clause extractions.

    Args:
        results: List of contract processing results.

    Returns:
        Tuple of (metadata_list, tfidf_matrix, idf_dict, vocab_list).
        metadata_list contains dicts with contract_id, clause_type, and text.
    """
    clause_types = ["termination_clause", "confidentiality_clause", "liability_clause", "summary"]

    metadata = []
    all_tokens = []

    for result in results:
        for clause_type in clause_types:
            text = result.get(clause_type, "")
            if text and text not in ("Not found", "Extraction failed"):
                tokens = _tokenize(text)
                metadata.append({
                    "contract_id": result["contract_id"],
                    "clause_type": clause_type,
                    "text": text,
                })
                all_tokens.append(tokens)

    if not all_tokens:
        logger.warning("No clauses found to build search index.")
        return [], np.array([]), {}, []

    logger.info("Building TF-IDF search index with %d clause entries...", len(all_tokens))

    # Compute IDF across all documents
    idf = _compute_idf(all_tokens)

    # Build vocabulary (top 5000 terms by IDF to keep vectors manageable)
    vocab = sorted(idf.keys(), key=lambda t: idf[t], reverse=True)[:5000]

    # Build TF-IDF matrix
    vectors = []
    for tokens in all_tokens:
        vec = _tfidf_vector(tokens, idf, vocab)
        vectors.append(vec)

    tfidf_matrix = np.array(vectors)
    logger.info("Search index built: %d entries, %d features.", *tfidf_matrix.shape)

    return metadata, tfidf_matrix, idf, vocab


def search(
    query: str,
    metadata: List[Dict],
    tfidf_matrix: np.ndarray,
    idf: Dict[str, float],
    vocab: List[str],
    top_k: int = 5,
) -> List[Dict]:
    """Search the clause index with a natural language query.

    Args:
        query: Natural language search query.
        metadata: List of metadata dicts from build_search_index.
        tfidf_matrix: Numpy array of TF-IDF vectors from build_search_index.
        idf: IDF dictionary from build_search_index.
        vocab: Vocabulary list from build_search_index.
        top_k: Number of top results to return.

    Returns:
        List of result dicts with keys: contract_id, clause_type,
        similarity, text_snippet.
    """
    if len(metadata) == 0 or tfidf_matrix.size == 0:
        logger.warning("Search index is empty.")
        return []

    query_tokens = _tokenize(query)
    query_vec = _tfidf_vector(query_tokens, idf, vocab)

    similarities = np.array([
        _cosine_similarity(query_vec, row) for row in tfidf_matrix
    ])

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if similarities[idx] <= 0:
            continue
        text = metadata[idx]["text"]
        snippet = text[:200] + "..." if len(text) > 200 else text
        results.append({
            "contract_id": metadata[idx]["contract_id"],
            "clause_type": metadata[idx]["clause_type"],
            "similarity": round(float(similarities[idx]), 4),
            "text_snippet": snippet,
        })

    return results


def save_index(
    metadata: List[Dict],
    tfidf_matrix: np.ndarray,
    idf: Dict[str, float],
    vocab: List[str],
    output_dir: Path,
) -> None:
    """Save the search index to disk for reuse.

    Args:
        metadata: List of metadata dicts.
        tfidf_matrix: Numpy array of TF-IDF vectors.
        idf: IDF dictionary.
        vocab: Vocabulary list.
        output_dir: Directory to save the index files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "search_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    np.save(output_dir / "search_tfidf.npy", tfidf_matrix)

    with open(output_dir / "search_vocab.json", "w", encoding="utf-8") as f:
        json.dump({"idf": idf, "vocab": vocab}, f)

    logger.info("Search index saved to %s", output_dir)


def load_index(output_dir: Path) -> Tuple[List[Dict], np.ndarray, Dict[str, float], List[str]]:
    """Load a previously saved search index.

    Args:
        output_dir: Directory containing saved index files.

    Returns:
        Tuple of (metadata_list, tfidf_matrix, idf_dict, vocab_list).
    """
    meta_path = output_dir / "search_metadata.json"
    tfidf_path = output_dir / "search_tfidf.npy"
    vocab_path = output_dir / "search_vocab.json"

    if not meta_path.exists() or not tfidf_path.exists() or not vocab_path.exists():
        raise FileNotFoundError(
            f"Search index not found in {output_dir}. Run the pipeline first."
        )

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    tfidf_matrix = np.load(tfidf_path)

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)

    idf = vocab_data["idf"]
    vocab = vocab_data["vocab"]

    logger.info("Search index loaded: %d entries.", len(metadata))

    return metadata, tfidf_matrix, idf, vocab
