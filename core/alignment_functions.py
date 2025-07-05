import numpy as np
from typing import List, Dict, Any, Callable

def align_units(
    src_units: List[str],
    tgt_units: List[str],
    embed_func: Callable[[List[str]], np.ndarray],
    similarity_threshold: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Aligns source and target units based on semantic similarity.
    """
    if not src_units or not tgt_units:
        return []

    src_embeddings = embed_func(src_units)
    tgt_embeddings = embed_func(tgt_units)

    # Calculate cosine similarity
    # Normalize embeddings to unit vectors
    src_norm = np.linalg.norm(src_embeddings, axis=1, keepdims=True)
    tgt_norm = np.linalg.norm(tgt_embeddings, axis=1, keepdims=True)
    src_normalized = src_embeddings / (src_norm + 1e-8)
    tgt_normalized = tgt_embeddings / (tgt_norm + 1e-8)

    similarity_matrix = np.dot(src_normalized, tgt_normalized.T)

    aligned_pairs = []
    used_tgt_indices = set()

    for i, src_unit in enumerate(src_units):
        best_match_score = -1
        best_match_idx = -1

        for j, tgt_unit in enumerate(tgt_units):
            if j not in used_tgt_indices:
                score = similarity_matrix[i, j]
                if score > best_match_score and score >= similarity_threshold:
                    best_match_score = score
                    best_match_idx = j
        
        if best_match_idx != -1:
            aligned_pairs.append({
                'src_idx': i,
                'tgt_idx': best_match_idx,
                'src': src_unit,
                'tgt': tgt_units[best_match_idx],
                'score': float(best_match_score)
            })
            used_tgt_indices.add(best_match_idx)
    
    return aligned_pairs
