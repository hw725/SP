import logging
from typing import List, Callable, Optional
import numpy as np
import os

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    logger.warning("OpenAI package not found. OpenAI embedder will not be available.")
    OPENAI_AVAILABLE = False

_openai_client = None

def get_openai_client(api_key: Optional[str] = None) -> OpenAI:
    global _openai_client
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI package is not installed. Please install it to use OpenAI embedder.")

    if _openai_client is None:
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError("OpenAI API key not provided and not found in environment variables.")
        _openai_client = OpenAI(api_key=api_key)
        logger.info("OpenAI client initialized.")
    return _openai_client

def compute_embeddings_with_cache(
    texts: List[str],
    model_name: str = "text-embedding-3-large",
    api_key: Optional[str] = None,
    **kwargs
) -> np.ndarray:
    """Compute embeddings using OpenAI API, with a placeholder for caching."""
    client = get_openai_client(api_key)
    
    try:
        response = client.embeddings.create(
            input=texts,
            model=model_name
        )
        embeddings = [d.embedding for d in response.data]
        return np.array(embeddings)
    except Exception as e:
        logger.error(f"Error computing OpenAI embeddings: {e}")
        raise
