"""텍스트 분석기들 - 다양한 한국어 형태소 분석기 지원"""
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
            logger.warning("MeCab이 설치되지 않았습니다. pip install mecab-python3")
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
                        'base_form': features[5] if len(features) > 5 else surface,
                        'reading': features[6] if len(features) > 6 else '',
                        'analyzer': 'mecab'
                    }
                    morphemes.append(morpheme_info)
            
            return morphemes
            
        except Exception as e:
            logger.debug(f"MeCab 분석 실패: {e}")
            return []

class KonlpyAnalyzer(BaseAnalyzer):
    """KoNLPy 기반 분석기 (Okt, Komoran, Hannanum 지원)"""
    
    def __init__(self, engine='okt', **kwargs):
        self.engine_name = engine
        self.analyzer = None
        
        try:
            if engine == 'okt':
                from konlpy.tag import Okt
                self.analyzer = Okt()
            elif engine == 'komoran':
                from konlpy.tag import Komoran
                self.analyzer = Komoran()
            elif engine == 'hannanum':
                from konlpy.tag import Hannanum
                self.analyzer = Hannanum()
            elif engine == 'kkma':
                from konlpy.tag import Kkma
                self.analyzer = Kkma()
            else:
                raise ValueError(f"지원하지 않는 KoNLPy 엔진: {engine}")
            
            logger.info(f"KoNLPy {engine.upper()} 분석기 초기화 성공")
            
        except ImportError:
            logger.warning(f"KoNLPy가 설치되지 않았습니다. pip install konlpy")
            self.analyzer = None
        except Exception as e:
            logger.warning(f"KoNLPy {engine} 초기화 실패: {e}")
            self.analyzer = None
    
    def analyze(self, text: str) -> List[Dict[str, Any]]:
        """KoNLPy 형태소 분석"""
        if not self.analyzer:
            return []
        
        try:
            # pos 메서드로 형태소 분석
            morphemes = []
            pos_result = self.analyzer.pos(text)
            
            for surface, pos in pos_result:
                morpheme_info = {
                    'surface': surface,
                    'pos': pos,
                    'pos_detail1': '',
                    'pos_detail2': '',
                    'base_form': surface,
                    'analyzer': f'konlpy_{self.engine_name}'
                }
                morphemes.append(morpheme_info)
            
            return morphemes
            
        except Exception as e:
            logger.debug(f"KoNLPy 분석 실패: {e}")
            return []

class KiwifixAnalyzer(BaseAnalyzer):
    """Kiwifix 형태소 분석기"""
    
    def __init__(self, **kwargs):
        try:
            from kiwipiepy import Kiwi
            self.kiwi = Kiwi()
            logger.info("Kiwifix 분석기 초기화 성공")
        except ImportError:
            logger.warning("Kiwifix가 설치되지 않았습니다. pip install kiwipiepy")
            self.kiwi = None
    
    def analyze(self, text: str) -> List[Dict[str, Any]]:
        """Kiwifix 형태소 분석"""
        if not self.kiwi:
            return []
        
        try:
            result = self.kiwi.analyze(text)
            morphemes = []
            
            for token in result:
                for morph in token:
                    morpheme_info = {
                        'surface': morph.form,
                        'pos': morph.tag,
                        'pos_detail1': '',
                        'pos_detail2': '',
                        'base_form': morph.form,
                        'score': getattr(morph, 'score', 0.0),
                        'analyzer': 'kiwi'
                    }
                    morphemes.append(morpheme_info)
            
            return morphemes
            
        except Exception as e:
            logger.debug(f"Kiwifix 분석 실패: {e}")
            return []

class JiebaAnalyzer(BaseAnalyzer):
    """Jieba 분석기 (중국어)"""
    
    def __init__(self, **kwargs):
        try:
            import jieba
            import jieba.posseg as pseg
            self.jieba = jieba
            self.pseg = pseg
            logger.info("Jieba 분석기 초기화 성공")
        except ImportError:
            logger.warning("jieba가 설치되지 않았습니다. pip install jieba")
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
                    'length': len(word),
                    'analyzer': 'jieba'
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

# *** 누락된 create_analyzer 함수 추가 ***
def create_analyzer(analyzer_type: str, **kwargs) -> BaseAnalyzer:
    """분석기 생성 팩토리"""
    
    if not analyzer_type or analyzer_type.lower() in ('none', ''):
        return DummyAnalyzer()
    
    analyzer_type = analyzer_type.lower()
    
    try:
        if analyzer_type == 'mecab':
            return MecabAnalyzer(**kwargs)
        elif analyzer_type == 'okt':
            return KonlpyAnalyzer('okt', **kwargs)
        elif analyzer_type == 'komoran':
            return KonlpyAnalyzer('komoran', **kwargs)
        elif analyzer_type == 'hannanum':
            return KonlpyAnalyzer('hannanum', **kwargs)
        elif analyzer_type == 'kkma':
            return KonlpyAnalyzer('kkma', **kwargs)
        elif analyzer_type == 'kiwi':
            return KiwifixAnalyzer(**kwargs)
        elif analyzer_type == 'jieba':
            return JiebaAnalyzer(**kwargs)
        else:
            logger.warning(f"알 수 없는 분석기: {analyzer_type}, 더미 분석기 사용")
            return DummyAnalyzer()
    
    except Exception as e:
        logger.error(f"분석기 생성 실패 ({analyzer_type}): {e}")
        return DummyAnalyzer()

# 사용 가능한 분석기 목록
def list_available_analyzers():
    """사용 가능한 분석기 목록 반환"""
    return ['mecab', 'okt', 'komoran', 'hannanum', 'kkma', 'kiwi', 'jieba', 'none']