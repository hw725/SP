"""SA 임베더 모듈 패키지"""

from .openai import compute_embeddings_with_cache as openai_embedder
from .bge import compute_embeddings_with_cache as bge_embedder

# 하위 호환성: compute_embeddings_with_cache는 bge_embedder로 alias
compute_embeddings_with_cache = bge_embedder

__all__ = ['openai_embedder', 'bge_embedder', 'compute_embeddings_with_cache']