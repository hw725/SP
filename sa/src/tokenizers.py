from typing import List
import logging

logger = logging.getLogger(__name__)

class BaseTokenizer:
    """토크나이저 기본 클래스"""
    def tokenize(self, text: str, column_name: str = None) -> List[str]:
        raise NotImplementedError

class HanjaHangulTokenizer(BaseTokenizer):
    """한자한글 토크나이저 (기본 토크나이저)"""
    def tokenize(self, text: str, column_name: str = None) -> List[str]:
        return text.split()

class MecabTokenizer(BaseTokenizer):
    """MeCab 토크나이저 (mecab_ko 전용)"""
    def __init__(self):
        self.tokenizer = None
        self.tokenizer_type = None
        self._load_tokenizer()
    
    def _load_tokenizer(self):
        try:
            import mecab_ko
            self.tokenizer = mecab_ko.Tagger()
            self.tokenizer_type = "mecab_ko"
            logger.info("mecab_ko.Tagger() 로드 완료")
        except ImportError:
            self.tokenizer = None
            self.tokenizer_type = None
            logger.warning("mecab_ko 패키지를 찾을 수 없습니다. `pip install mecab_ko`를 시도하세요.")
        except Exception as e:
            self.tokenizer = None
            self.tokenizer_type = None
            logger.error(f"mecab_ko 로드 중 에러 발생: {e}")
    
    def _parse_mecab_output(self, parsed_text: str) -> List[str]:
        """MeCab 출력을 형태소 리스트로 변환"""
        morphemes = []
        for line in parsed_text.strip().split('\n'):
            if line and '\t' in line:
                morpheme = line.split('\t')[0]
                if morpheme and morpheme != 'EOS':
                    morphemes.append(morpheme)
        return morphemes
    
    def _parse_mecab_pos_output(self, parsed_text: str) -> List[tuple]:
        """MeCab 출력을 (형태소, 품사) 리스트로 변환"""
        pos_tags = []
        for line in parsed_text.strip().split('\n'):
            if line and '\t' in line:
                parts = line.split('\t')
                morpheme = parts[0]
                if morpheme and morpheme != 'EOS':
                    # 품사는 첫 번째 필드에서 추출
                    pos_info = parts[1].split(',')[0] if len(parts) > 1 else 'UNKNOWN'
                    pos_tags.append((morpheme, pos_info))
        return pos_tags
    
    def tokenize(self, text: str, column_name: str = None) -> List[str]:
        if not self.tokenizer or not text.strip():
            # 토크나이저가 없거나 빈 텍스트면 기본 분할
            return text.split()
        
        try:
            # mecab_ko.Tagger().parse() 사용
            parsed = self.tokenizer.parse(text)
            return self._parse_mecab_output(parsed)
                
        except Exception as e:
            logger.warning(f"MeCab 토크나이징 실패 ({self.tokenizer_type}): {e}")
            return text.split()

    def analyze_structure(self, text: str) -> dict:
        """내부 구조 분석 (분할하지 않음)"""
        if not self.tokenizer or not text.strip():
            return {"original": text, "internal_structure": [text]}
        
        try:
            # mecab_ko: parse() 결과에서 품사 정보 추출
            parsed = self.tokenizer.parse(text)
            pos_tags = self._parse_mecab_pos_output(parsed)
            return {
                "original": text,
                "internal_structure": pos_tags,
                "morpheme_hints": {morpheme: f"형태소_{pos}_{morpheme}" for morpheme, pos in pos_tags}
            }
                
        except Exception as e:
            logger.warning(f"MeCab 구조 분석 실패 ({self.tokenizer_type}): {e}")
            return {"original": text, "internal_structure": [text]}

class JiebaTokenizer(BaseTokenizer):
    """Jieba 토크나이저"""
    def __init__(self):
        try:
            import jieba
            self.tokenizer = jieba
            logger.info("Jieba 토크나이저 로드 완료")
        except ImportError:
            self.tokenizer = None
            logger.warning("jieba 패키지 없음")
    
    def tokenize(self, text: str, column_name: str = None) -> List[str]:
        if self.tokenizer and text.strip():
            try:
                return list(self.tokenizer.cut(text))
            except Exception as e:
                logger.warning(f"Jieba 토크나이징 실패: {e}")
        return text.split()

    def analyze_structure(self, text: str) -> dict:
        """내부 구조 분석 (분할하지 않음)"""
        if not self.tokenizer or not text.strip():
            return {"original": text, "internal_structure": [text]}
        
        try:
            segments = list(self.tokenizer.cut(text))
            return {
                "original": text,
                "internal_structure": segments,
                "meaning_hints": {seg: f"중국어_단어_{seg}" for seg in segments if len(seg) >= 2}
            }
        except Exception as e:
            logger.warning(f"Jieba 구조 분석 실패: {e}")
            return {"original": text, "internal_structure": [text]}