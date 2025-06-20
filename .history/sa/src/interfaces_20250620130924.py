"""기본 인터페이스 정의"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable, Tuple

class BaseTokenizer(ABC):
    """토크나이저 기본 클래스"""
    
    def __init__(self, **kwargs):
        self.config = kwargs
    
    @abstractmethod
    def tokenize(self, text: str, column_name: str = None) -> List[str]:
        """텍스트를 토큰으로 분할"""
        pass
    
    def process_row(self, src_text: str, tgt_text: str, embed_func: Callable) -> Tuple[str, str, Dict[str, Any]]:
        """행 단위 처리 (기본 구현)"""
        # 기본적으로는 단순 토큰화만 수행
        src_tokens = self.tokenize(src_text, 'source')
        tgt_tokens = self.tokenize(tgt_text, 'target')
        
        aligned_src = ' | '.join(src_tokens)
        aligned_tgt = ' | '.join(tgt_tokens)
        
        return aligned_src, aligned_tgt, {'status': 'basic_tokenization'}

class BaseEmbedder(ABC):
    """임베더 기본 클래스"""
    
    def __init__(self, **kwargs):
        self.config = kwargs
    
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """텍스트 목록을 임베딩으로 변환"""
        pass
    
    def __call__(self, texts: List[str]) -> List[List[float]]:
        """임베딩 함수로 호출 가능"""
        return self.embed(texts)