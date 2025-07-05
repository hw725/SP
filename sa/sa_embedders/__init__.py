"""SA 임베더 모듈 패키지 - 의존성 및 함수 호출 에러 방지 철저 체크"""

import importlib
import logging

logger = logging.getLogger(__name__)

# 안전한 임포트 및 함수 래핑
try:
    from .openai import compute_embeddings_with_cache
    from .openai import compute_embeddings_with_cache as openai_embedder
except Exception as e:
    logger.error(f"OpenAI 임베더 임포트 실패: {e}")
    compute_embeddings_with_cache = None
    openai_embedder = None

try:
    from .bge import compute_embeddings_with_cache as bge_embedder, get_embed_func as bge_get_embed_func, get_embed_func, get_embedding_manager
except Exception as e:
    logger.error(f"BGE 임베더 임포트 실패: {e}")
    bge_embedder = None
    bge_get_embed_func = None
    get_embed_func = None
    get_embedding_manager = None

def get_embedder(name: str, device_id=None, model_name=None):
    """임베더 이름에 따라 함수 반환 (openai/bge, device_id/model_name 지정 가능, 에러 방지)"""
    if name == "openai":
        if openai_embedder is None:
            raise ImportError("OpenAI 임베더가 임포트되지 않았습니다. openai 패키지 및 sa_embedders.openai 확인 필요.")
        return openai_embedder
    elif name == "bert":
        raise ValueError("SA(문장 정렬기)에서 bert 임베더는 splitter 용도로만 사용 가능합니다. 임베더로 사용할 수 없습니다.")
    elif name == "bge" or name is None:  # 기본값은 bge
        if bge_get_embed_func is None:
            raise ImportError("BGE 임베더가 임포트되지 않았습니다. FlagEmbedding 패키지 및 sa_embedders.bge 확인 필요.")
        return bge_get_embed_func(device_id=device_id)
    else:
        raise ValueError(f"지원하지 않는 임베더: {name}. 지원: openai, bge")

__all__ = ['openai_embedder', 'bge_embedder', 'get_embedder', 'compute_embeddings_with_cache', 'get_embed_func']