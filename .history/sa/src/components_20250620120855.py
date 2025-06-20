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

"""컴포넌트 팩토리 - 원문 jieba, 번역문 mecab 기본값 고정"""
import logging
from typing import Dict, Any, Optional, List

from .interfaces import BaseTokenizer, BaseEmbedder
from .text_alignment import TextAlignmentProcessor
from .analyzers import create_analyzer
from .embedders import BGEM3Embedder, SentenceTransformerEmbedder, OpenAIEmbedder, CohereEmbedder

logger = logging.getLogger(__name__)

class AnalyzerAwareTokenizer(BaseTokenizer):
    """분석기 정보를 활용하는 토크나이저"""
    
    def __init__(self, source_analyzer_type: Optional[str] = 'jieba',     # 기본값: 원문 jieba
                 target_analyzer_type: Optional[str] = 'mecab',      # 기본값: 번역문 mecab
                 **kwargs):
        super().__init__(**kwargs)
        
        # 분석기 생성
        self.source_analyzer = create_analyzer(source_analyzer_type or 'none')
        self.target_analyzer = create_analyzer(target_analyzer_type or 'none')
        
        # 프로세서 생성
        self.processor = TextAlignmentProcessor(
            source_analyzer=self.source_analyzer,
            target_analyzer=self.target_analyzer,
            min_tokens=kwargs.get('min_tokens', 1)
        )
        
        logger.info(f"분석기 토크나이저 초기화: {source_analyzer_type or 'None'} (원문), {target_analyzer_type or 'None'} (번역문)")
    
    def tokenize(self, text: str, column_name: str = None) -> List[str]:
        """분석기 정보를 활용한 토크나이징"""
        from .text_alignment import SourceTextSplitter
        fallback_splitter = SourceTextSplitter()
        return fallback_splitter.split(text)
    
    def process_row(self, src_text: str, tgt_text: str, embed_func):
        """행 단위 처리 - 분석기 정보 활용"""
        return self.processor.process(src_text, tgt_text, embed_func)

class MultiAnalyzerTokenizer(BaseTokenizer):
    """번역문에 여러 분석기를 조합하는 토크나이저"""
    
    def __init__(self, source_analyzer_type: str = 'jieba',
                 target_analyzer_types: List[str] = ['jieba', 'mecab'],  # 조합형
                 **kwargs):
        super().__init__(**kwargs)
        
        # 원문 분석기 (단일)
        self.source_analyzer = create_analyzer(source_analyzer_type)
        
        # 번역문 분석기들 (복수)
        self.target_analyzers = []
        for analyzer_type in target_analyzer_types:
            analyzer = create_analyzer(analyzer_type)
            if analyzer:
                self.target_analyzers.append(analyzer)
        
        # 프로세서 생성 (첫 번째 번역문 분석기를 메인으로)
        main_target_analyzer = self.target_analyzers[0] if self.target_analyzers else None
        self.processor = TextAlignmentProcessor(
            source_analyzer=self.source_analyzer,
            target_analyzer=main_target_analyzer,
            min_tokens=kwargs.get('min_tokens', 1)
        )
        
        logger.info(f"다중 분석기 토크나이저 초기화: {source_analyzer_type} (원문), {target_analyzer_types} (번역문)")
    
    def tokenize(self, text: str, column_name: str = None) -> List[str]:
        from .text_alignment import SourceTextSplitter
        fallback_splitter = SourceTextSplitter()
        return fallback_splitter.split(text)
    
    def process_row(self, src_text: str, tgt_text: str, embed_func):
        """다중 분석기를 활용한 행 처리"""
        # 기본 프로세서 사용 (향후 다중 분석기 결합 로직 확장 가능)
        return self.processor.process(src_text, tgt_text, embed_func)

# 토크나이저 레지스트리 - 기본값 고정
TOKENIZER_REGISTRY = {
    # *** 기본값: 원문 jieba, 번역문 mecab ***
    'default': lambda **kwargs: AnalyzerAwareTokenizer('jieba', 'mecab', **kwargs),
    
    # 조합형 기본값 (번역문에 jieba+mecab)
    'default-combo': lambda **kwargs: MultiAnalyzerTokenizer('jieba', ['jieba', 'mecab'], **kwargs),
    
    # Prototype02만 (분석기 없음)
    'prototype02': lambda **kwargs: AnalyzerAwareTokenizer(None, None, **kwargs),
    
    # 단일 분석기들
    'prototype02-mecab': lambda **kwargs: AnalyzerAwareTokenizer('mecab', None, **kwargs),
    'prototype02-jieba': lambda **kwargs: AnalyzerAwareTokenizer('jieba', None, **kwargs),
    'prototype02-okt': lambda **kwargs: AnalyzerAwareTokenizer('okt', None, **kwargs),
    'prototype02-komoran': lambda **kwargs: AnalyzerAwareTokenizer('komoran', None, **kwargs),
    
    # 조합형들
    'jieba-mecab': lambda **kwargs: AnalyzerAwareTokenizer('jieba', 'mecab', **kwargs),
    'jieba-okt': lambda **kwargs: AnalyzerAwareTokenizer('jieba', 'okt', **kwargs),
    'mecab-jieba': lambda **kwargs: AnalyzerAwareTokenizer('mecab', 'jieba', **kwargs),
    
    # 다중 분석기 조합들
    'jieba-multi': lambda **kwargs: MultiAnalyzerTokenizer('jieba', ['jieba', 'mecab', 'okt'], **kwargs),
}

# 임베더 레지스트리
EMBEDDER_REGISTRY = {
    'bge-m3': BGEM3Embedder,
    'sentence-transformer': SentenceTransformerEmbedder,
    'openai': OpenAIEmbedder,
    'cohere': CohereEmbedder,
}

def get_tokenizer(tokenizer_type: str = 'default', **kwargs) -> BaseTokenizer:
    """토크나이저 생성 - 기본값: 원문 jieba, 번역문 mecab"""
    factory_func = TOKENIZER_REGISTRY.get(tokenizer_type)
    if factory_func is None:
        logger.warning(f"알 수 없는 토크나이저: {tokenizer_type}, 기본값(jieba-mecab) 사용")
        factory_func = TOKENIZER_REGISTRY['default']
    
    try:
        return factory_func(**kwargs)
    except Exception as e:
        logger.error(f"토크나이저 생성 실패 ({tokenizer_type}): {e}")
        return TOKENIZER_REGISTRY['prototype02'](**kwargs)  # 최후 폴백

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