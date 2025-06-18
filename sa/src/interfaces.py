from abc import ABC, abstractmethod
from typing import List, Any

class BaseTokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str, column_name: str = None) -> List[str]:
        """텍스트를 토큰 리스트로 분할"""
        pass

class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> Any:
        """텍스트 리스트를 임베딩 배열로 변환"""
        pass