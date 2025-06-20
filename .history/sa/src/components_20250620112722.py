# src/components.py
"""
모든 교체 가능한 컴포넌트(Tokenizer, Embedder, Aligner)의
인터페이스(규격)와 팩토리 함수(생성)를 정의합니다.
"""
from abc import ABC, abstractmethod
import logging
from typing import List, Dict, Tuple, Any, Type, Optional, Callable

# 인터페이스를 여기서 재수입하여 외부에서 components를 통해 접근 가능
from .interfaces import BaseTokenizer, BaseEmbedder

from .tokenizers import HanjaHangulTokenizer, MecabTokenizer, JiebaTokenizer
from .embedders import SentenceTransformerEmbedder, OpenAIEmbedder, CohereEmbedder, BGEM3Embedder
from .text_alignment import TextMasker
from .tokenizer_chain import TokenizerChain

logger = logging.getLogger(__name__)

class Prototype02SourceTokenizer(BaseTokenizer):
    """Prototype02 SourceTextSplitter 래핑"""
    
    def __init__(self, **kwargs):
        from .text_alignment import SourceTextSplitter
        self.splitter = SourceTextSplitter(**kwargs)
    
    def tokenize(self, text: str, column_name: str = None):
        return self.splitter.split(text)

class ChainedTokenizer(BaseTokenizer):
    """체인 토크나이저 래퍼"""
    
    def __init__(self, language_tokenizer_type: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.chain = TokenizerChain(language_tokenizer_type)
        self.language_type = language_tokenizer_type
    
    def tokenize(self, text: str, column_name: str = None) -> List[str]:
        """체인 토크나이징 수행"""
        if column_name == 'source' or column_name == '원문':
            # 원문: Prototype02 + 언어별 토크나이저
            _, _, src_units = self.chain.process_source_text(text)
            return src_units
        else:
            # 번역문: Prototype02 마스킹만
            masked_text, _ = self.chain.process_target_text(text)
            return [masked_text]  # 단일 텍스트로 반환 (DP에서 분할)

# 토크나이저 레지스트리 (업데이트된 버전)
TOKENIZER_REGISTRY = {
    # Prototype02만 사용
    'prototype02': lambda **kwargs: ChainedTokenizer(None, **kwargs),
    
    # Prototype02 + 언어별 토크나이저
    'prototype02-mecab': lambda **kwargs: ChainedTokenizer('mecab', **kwargs),
    'prototype02-jieba': lambda **kwargs: ChainedTokenizer('jieba', **kwargs),
    'prototype02-hanja': lambda **kwargs: ChainedTokenizer('hanja_hangul', **kwargs),
}

# 임베더 레지스트리
EMBEDDER_REGISTRY = {
    'bge-m3': BGEM3Embedder,
    'sentence-transformer': SentenceTransformerEmbedder,
    'openai': OpenAIEmbedder,
    'cohere': CohereEmbedder,
}

def get_tokenizer(tokenizer_type: str, **kwargs) -> BaseTokenizer:
    factory_func = TOKENIZER_REGISTRY.get(tokenizer_type)
    if factory_func is None:
        logger.warning(f"알 수 없는 토크나이저: {tokenizer_type}, prototype02 사용")
        factory_func = TOKENIZER_REGISTRY['prototype02']
    
    try:
        return factory_func(**kwargs)
    except Exception as e:
        logger.error(f"토크나이저 생성 실패 ({tokenizer_type}): {e}")
        return TOKENIZER_REGISTRY['prototype02'](**kwargs)

def get_embedder(embedder_type: str, **kwargs) -> BaseEmbedder:
    cls = EMBEDDER_REGISTRY.get(embedder_type)
    if cls is None:
        available = list(EMBEDDER_REGISTRY.keys())
        raise ValueError(f"Unknown embedder type: {embedder_type}. Available: {available}")
    
    try:
        return cls(**kwargs)
    except Exception as e:
        logger.error(f"임베더 생성 실패 ({embedder_type}): {e}")
        raise

def list_available_tokenizers():
    """사용 가능한 토크나이저 체인 목록"""
    return list(TOKENIZER_REGISTRY.keys())

def list_available_embedders():
    return list(EMBEDDER_REGISTRY.keys())