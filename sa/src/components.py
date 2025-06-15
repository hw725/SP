# src/components.py
"""
모든 교체 가능한 컴포넌트(Tokenizer, Embedder, Aligner)의
인터페이스(규격)와 팩토리 함수(생성)를 정의합니다.
"""
import logging
import numpy as np
import regex # 사용되지 않으면 제거 가능
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any, Type, Optional, Callable
import os # os.path.join 사용 위함

logger = logging.getLogger(__name__)

# ================================
# 베이스 클래스들
# ================================

class BaseTokenizer(ABC):
    """토크나이저 베이스 클래스"""
    @abstractmethod
    def tokenize(self, text: str, column_name: str = None) -> List[str]:
        pass

class BaseEmbedder(ABC):
    """임베더 베이스 클래스"""
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """하위 클래스에서 구현해야 할 임베딩 메서드"""
        pass

class BaseAligner(ABC):
    """정렬기 기본 클래스"""
    @abstractmethod
    def align(self, src_units: List[str], tgt_units: List[str], embed_func: Callable) -> List[Tuple[str, str]]:
        raise NotImplementedError

# ================================
# 레지스트리 정의 (지연 import로 순환 참조 방지)
# ================================

def _get_tokenizer_registry():
    """토크나이저 레지스트리"""
    from .tokenizers import HanjaHangulTokenizer, MecabTokenizer, JiebaTokenizer
    return {
        'hanja_hangul': HanjaHangulTokenizer,
        'mecab': MecabTokenizer,
        'jieba': JiebaTokenizer
    }

def _get_embedder_registry():
    """임베더 레지스트리"""
    from .embedders import SentenceTransformerEmbedder, OpenAIEmbedder, CohereEmbedder, BGEM3Embedder
    return {
        'sentence_transformer': SentenceTransformerEmbedder,
        'openai': OpenAIEmbedder,
        'cohere': CohereEmbedder,
        'bge-m3': BGEM3Embedder
    }

def _get_aligner_registry():
    """정렬기 레지스트리"""
    from .aligner import StrictAligner, DPAligner, GreedyAligner
    return {
        'strict': StrictAligner,
        'dp': DPAligner,
        'greedy': GreedyAligner
    }

# ✅ main.py에서 사용할 레지스트리 변수들
TOKENIZER_REGISTRY = _get_tokenizer_registry()
EMBEDDER_REGISTRY = _get_embedder_registry()  
ALIGNER_REGISTRY = _get_aligner_registry()

# ================================
# 팩토리 함수들
# ================================

def get_tokenizer(tokenizer_type: str, **kwargs) -> BaseTokenizer:
    """토크나이저 팩토리 함수"""
    registry = _get_tokenizer_registry()
    if tokenizer_type not in registry:
        raise ValueError(f"지원하지 않는 토크나이저: {tokenizer_type}")
    
    tokenizer_class = registry[tokenizer_type]
    return tokenizer_class(**kwargs)

def get_embedder(embedder_type: str, config_obj: Optional[Any] = None, **kwargs) -> BaseEmbedder:
    """임베더 팩토리 함수"""
    registry = _get_embedder_registry()
    if embedder_type not in registry:
        raise ValueError(f"지원하지 않는 임베더: {embedder_type}")
    
    embedder_class = registry[embedder_type]
    
    final_kwargs = {}
    # Config 객체의 embedder_config를 우선 적용
    if config_obj and hasattr(config_obj, 'embedder_config'):
        final_kwargs.update(config_obj.embedder_config)
    
    # 명시적으로 전달된 kwargs로 덮어쓰기
    final_kwargs.update(kwargs)

    # cache_dir가 명시적으로 설정되지 않았고, Config 객체에 default_cache_root가 있다면 동적 설정
    if 'cache_dir' not in final_kwargs and config_obj and hasattr(config_obj, 'default_cache_root') and config_obj.default_cache_root:
        final_kwargs['cache_dir'] = os.path.join(config_obj.default_cache_root, embedder_type.replace('-', '_'))
        logger.info(f"Embedder '{embedder_type}'의 캐시 디렉토리 자동 설정: {final_kwargs['cache_dir']}")
    
    # API 키 처리 (Config.__post_init__에서 이미 처리하지만, 여기서도 환경 변수 fallback 가능)
    if embedder_type == "openai" and not final_kwargs.get("api_key"):
        final_kwargs["api_key"] = os.getenv("OPENAI_API_KEY")
        if not final_kwargs["api_key"]:
             logger.warning("OpenAI API 키가 설정되지 않았습니다. Config 또는 환경변수를 확인하세요.")
             # Config에서 이미 ValueError를 발생시키므로 여기서는 경고만.
    elif embedder_type == "cohere" and not final_kwargs.get("api_key"):
        final_kwargs["api_key"] = os.getenv("COHERE_API_KEY")
        if not final_kwargs["api_key"]:
            logger.warning("Cohere API 키가 설정되지 않았습니다. Config 또는 환경변수를 확인하세요.")

    logger.debug(f"임베더 '{embedder_type}' 생성 (kwargs: {final_kwargs})")
    return embedder_class(**final_kwargs)

def get_aligner(aligner_type: str, **kwargs) -> BaseAligner:
    """정렬기 팩토리 함수"""
    registry = _get_aligner_registry()
    if aligner_type not in registry:
        raise ValueError(f"지원하지 않는 정렬기: {aligner_type}")
    
    aligner_class = registry[aligner_type]
    return aligner_class(**kwargs)