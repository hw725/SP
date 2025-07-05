import logging
from typing import List, Callable, Optional
import numpy as np

logger = logging.getLogger(__name__)

try:
    from FlagEmbedding import FlagModel
    FLAG_EMBEDDING_AVAILABLE = True
except ImportError:
    logger.warning("FlagEmbedding package not found. BGE embedder will not be available.")
    FLAG_EMBEDDING_AVAILABLE = False

_bge_model = None

def get_embed_func(device_id: Optional[str] = None) -> Callable[[List[str]], np.ndarray]:
    global _bge_model
    if not FLAG_EMBEDDING_AVAILABLE:
        raise ImportError("FlagEmbedding package is not installed. Please install it to use BGE embedder.")

    if _bge_model is None:
        try:
            # Load the BGE-M3 model. Adjust model name if needed.
            _bge_model = FlagModel('BAAI/bge-m3', 
                                   query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                                   device=device_id if device_id else "cpu")
            logger.info(f"BGE-M3 model loaded successfully on device: {device_id if device_id else 'cpu'}")
        except Exception as e:
            logger.error(f"Failed to load BGE-M3 model: {e}")
            raise

    def _embed(texts: List[str]) -> np.ndarray:
        if not _bge_model:
            raise RuntimeError("BGE model is not loaded.")
        return _bge_model.encode(texts)

    return _embed

def compute_embeddings_with_cache(texts: List[str], embed_func: Callable[[List[str]], np.ndarray]) -> np.ndarray:
    """Compute embeddings using the provided embed_func, with a placeholder for caching."""
    # In a real application, you would add caching logic here.
    # For now, it just calls the embed_func directly.
    return embed_func(texts)
