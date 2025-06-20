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

"""컴포넌트 팩토리 - 분석기 지원"""
import logging
from typing import Dict, Any, Optional

from .interfaces import BaseTokenizer, BaseEmbedder
from .text_alignment import TextAlignmentProcessor
from .analyzers import MecabAnalyzer, JiebaAnalyzer, DummyAnalyzer
from .embedders import BGEM3Embedder, SentenceTransformerEmbedder, OpenAIEmbedder, CohereEmbedder

logger = logging.getLogger(__name__)

class AnalyzerAwareTokenizer(BaseTokenizer):
    """분석기 정보를 활용하는 토크나이저"""
    
    def __init__(self, source_analyzer_type: Optional[str] = None, 
                 target_analyzer_type: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        
        # 분석기 생성
        self.source_analyzer = self._create_analyzer(source_analyzer_type)
        self.target_analyzer = self._create_analyzer(target_analyzer_type)
        
        # 프로세서 생성
        self.processor = TextAlignmentProcessor(
            source_analyzer=self.source_analyzer,
            target_analyzer=self.target_analyzer,
            min_tokens=kwargs.get('min_tokens', 1)
        )
        
        logger.info(f"분석기 토크나이저 초기화: {source_analyzer_type or 'None'} (원문), {target_analyzer_type or 'None'} (번역문)")
    
    def _create_analyzer(self, analyzer_type: Optional[str]):
        """분석기 생성"""
        if not analyzer_type or analyzer_type == 'none':
            return DummyAnalyzer()
        elif analyzer_type == 'mecab':
            return MecabAnalyzer()
        elif analyzer_type == 'jieba':
            return JiebaAnalyzer()
        else:
            logger.warning(f"알 수 없는 분석기: {analyzer_type}")
            return DummyAnalyzer()
    
    def tokenize(self, text: str, column_name: str = None) -> List[str]:
        """분석기 정보를 활용한 토크나이징"""
        # 이 메서드는 하위 호환성을 위해 유지
        # 실제 처리는 process_row에서 수행
        from .text_alignment import SourceTextSplitter
        fallback_splitter = SourceTextSplitter()
        return fallback_splitter.split(text)
    
    def process_row(self, src_text: str, tgt_text: str, embed_func):
        """행 단위 처리 - 분석기 정보 활용"""
        return self.processor.process(src_text, tgt_text, embed_func)

# 토크나이저 레지스트리 (분석기 조합)
TOKENIZER_REGISTRY = {
    # Prototype02만 사용
    'prototype02': lambda **kwargs: AnalyzerAwareTokenizer(None, None, **kwargs),
    
    # Prototype02 + 분석기 조합
    'prototype02-mecab': lambda **kwargs: AnalyzerAwareTokenizer('mecab', None, **kwargs),
    'prototype02-jieba': lambda **kwargs: AnalyzerAwareTokenizer('jieba', None, **kwargs),
    
    # 원문/번역문 다른 분석기
    'mecab-jieba': lambda **kwargs: AnalyzerAwareTokenizer('mecab', 'jieba', **kwargs),
    'jieba-mecab': lambda **kwargs: AnalyzerAwareTokenizer('jieba', 'mecab', **kwargs),
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
    """사용 가능한 토크나이저 조합 목록"""
    return list(TOKENIZER_REGISTRY.keys())

def list_available_embedders():
    return list(EMBEDDER_REGISTRY.keys())