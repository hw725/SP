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

logger = logging.getLogger(__name__)

class Prototype02SourceTokenizer(BaseTokenizer):
    """Prototype02 SourceTextSplitter 래핑"""
    
    def __init__(self, **kwargs):
        from .text_alignment import SourceTextSplitter
        self.splitter = SourceTextSplitter(**kwargs)
    
    def tokenize(self, text: str, column_name: str = None):
        return self.splitter.split(text)

# 토크나이저 레지스트리 (업데이트된 버전)
TOKENIZER_REGISTRY = {
    'hanja_hangul': HanjaHangulTokenizer,
    'mecab': MecabTokenizer,
    'jieba': JiebaTokenizer,
    'prototype02': Prototype02SourceTokenizer,
}

# 임베더 레지스트리
EMBEDDER_REGISTRY = {
    'bge-m3': BGEM3Embedder,
    'sentence-transformer': SentenceTransformerEmbedder,
    'openai': OpenAIEmbedder,
    'cohere': CohereEmbedder,
}

def get_tokenizer(tokenizer_type: str, **kwargs) -> BaseTokenizer:
    cls = TOKENIZER_REGISTRY.get(tokenizer_type)
    if cls is None:
        logger.warning(f"알 수 없는 토크나이저: {tokenizer_type}, Prototype02 사용")
        cls = Prototype02SourceTokenizer
    
    try:
        return cls(**kwargs)
    except Exception as e:
        logger.error(f"토크나이저 생성 실패 ({tokenizer_type}): {e}")
        return Prototype02SourceTokenizer(**kwargs)

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
    return list(TOKENIZER_REGISTRY.keys())

def list_available_embedders():
    return list(EMBEDDER_REGISTRY.keys())