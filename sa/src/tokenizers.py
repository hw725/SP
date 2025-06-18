"""토크나이저 구현"""
import logging
from typing import List
from .interfaces import BaseTokenizer  # components 대신 interfaces 임포트

logger = logging.getLogger(__name__)

class HanjaHangulTokenizer(BaseTokenizer):
    """한자-한글 토크나이저"""
    def __init__(self, **kwargs):
        pass
        
    def tokenize(self, text: str, column_name: str = None) -> List[str]:
        # 한자와 한글을 분리하는 로직
        import regex
        tokens = regex.findall(r'[\u4e00-\u9fff]+|[\uac00-\ud7af]+|\S', text)
        return [token.strip() for token in tokens if token.strip()]

class MecabTokenizer(BaseTokenizer):
    """MeCab 토크나이저"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            import MeCab
            self.tagger = MeCab.Tagger()
        except ImportError:
            logger.warning("MeCab이 설치되지 않았습니다.")
            self.tagger = None

    def tokenize(self, text: str, column_name: str = None) -> List[str]:
        if not self.tagger:
            return text.split()
        result = self.tagger.parse(text)
        tokens = []
        for line in result.split('\n'):
            if not line or line == 'EOS':
                continue
            surface = line.split('\t')[0]
            tokens.append(surface)
        return tokens

class JiebaTokenizer(BaseTokenizer):
    """Jieba 토크나이저"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            import jieba
            self.jieba = jieba
        except ImportError:
            logger.warning("jieba가 설치되지 않았습니다.")
            self.jieba = None

    def tokenize(self, text: str, column_name: str = None) -> List[str]:
        if not self.jieba:
            return text.split()
        return list(self.jieba.cut(text))