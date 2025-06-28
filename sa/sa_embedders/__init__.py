"""SA 임베더 모듈 패키지"""

from .openai import compute_embeddings_with_cache
from .openai import compute_embeddings_with_cache as openai_embedder
from .bge import compute_embeddings_with_cache as bge_embedder, get_embed_func as bge_get_embed_func, get_embed_func

def get_embedder(name: str, device_id=None):
    """임베더 이름에 따라 함수 반환 (openai/bge, device_id 지정 가능)"""
    if name == "openai":
        return openai_embedder
    return bge_get_embed_func(device_id=device_id)

__all__ = ['openai_embedder', 'bge_embedder', 'get_embedder']