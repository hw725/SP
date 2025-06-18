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

def _get_tokenizer_registry() -> Dict[str, Type[BaseTokenizer]]:
    return {
        'hanja_hangul': HanjaHangulTokenizer,
        'mecab': MecabTokenizer,
        'jieba': JiebaTokenizer
    }

def _get_embedder_registry() -> Dict[str, Type[BaseEmbedder]]:
    return {
        'sentence-transformer': SentenceTransformerEmbedder,
        'openai': OpenAIEmbedder,
        'cohere': CohereEmbedder,
        'bge-m3': BGEM3Embedder
    }

TOKENIZER_REGISTRY = _get_tokenizer_registry()
EMBEDDER_REGISTRY = _get_embedder_registry()

def get_tokenizer(tokenizer_type: str, **kwargs) -> BaseTokenizer:
    cls = TOKENIZER_REGISTRY.get(tokenizer_type)
    if not cls:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
    return cls(**kwargs)

def get_embedder(embedder_type: str, **kwargs) -> BaseEmbedder:
    cls = EMBEDDER_REGISTRY.get(embedder_type)
    if not cls:
        raise ValueError(f"Unknown embedder type: {embedder_type}")
    return cls(**kwargs)

def get_text_masker(**kwargs) -> TextMasker:
    return TextMasker(**kwargs)