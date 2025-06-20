"""텍스트 분석기들 - 분할하지 않고 분석만 수행"""
import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseAnalyzer(ABC):
    """텍스트 분석기 기본 클래스"""
    
    @abstractmethod
    def analyze(self, text: str) -> List[Dict[str, Any]]:
        """텍스트 분석 - 분할하지 않고 분석 정보만 반환"""
        pass

class MecabAnalyzer(BaseAnalyzer):
    """MeCab 형태소 분석기"""
    
    def __init__(self, **kwargs):
        try:
            import MeCab
            self.tagger = MeCab.Tagger()
            logger.info("MeCab 분석기 초기화 성공")
        except ImportError:
            logger.warning("MeCab이 설치되지 않았습니다.")
            self.tagger = None
    
    def analyze(self, text: str) -> List[Dict[str, Any]]:
        """MeCab 형태소 분석"""
        if not self.tagger:
            return []
        
        try:
            result = self.tagger.parse(text)
            morphemes = []
            
            for line in result.split('\n'):
                if not line or line == 'EOS':
                    continue
                    
                parts = line.split('\t')
                if len(parts) >= 2:
                    surface = parts[0]
                    features = parts[1].split(',')
                    
                    morpheme_info = {
                        'surface': surface,
                        'pos': features[0] if features else 'Unknown',
                        'pos_detail1': features[1] if len(features) > 1 else '',
                        'pos_detail2': features[2] if len(features) > 2 else '',
                        'inflection': features[3] if len(features) > 3 else '',
                        'conjugation': features[4] if len(features) > 4 else '',
                        'base_form': features[5] if len(features) > 5 else surface,
                        'reading': features[6] if len(features) > 6 else '',
                        'pronunciation': features[7] if len(features) > 7 else ''
                    }
                    morphemes.append(morpheme_info)
            
            return morphemes
            
        except Exception as e:
            logger.debug(f"MeCab 분석 실패: {e}")
            return []

class JiebaAnalyzer(BaseAnalyzer):
    """Jieba 분석기"""
    
    def __init__(self, **kwargs):
        try:
            import jieba
            import jieba.posseg as pseg
            self.jieba = jieba
            self.pseg = pseg
            logger.info("Jieba 분석기 초기화 성공")
        except ImportError:
            logger.warning("jieba가 설치되지 않았습니다.")
            self.jieba = None
            self.pseg = None
    
    def analyze(self, text: str) -> List[Dict[str, Any]]:
        """Jieba 품사 분석"""
        if not self.pseg:
            return []
        
        try:
            words = self.pseg.cut(text)
            morphemes = []
            
            for word, flag in words:
                morpheme_info = {
                    'surface': word,
                    'pos': flag,
                    'pos_detail1': '',
                    'pos_detail2': '',
                    'base_form': word,
                    'length': len(word)
                }
                morphemes.append(morpheme_info)
            
            return morphemes
            
        except Exception as e:
            logger.debug(f"Jieba 분석 실패: {e}")
            return []

class DummyAnalyzer(BaseAnalyzer):
    """더미 분석기 - 분석 정보 없음"""
    
    def analyze(self, text: str) -> List[Dict[str, Any]]:
        return []