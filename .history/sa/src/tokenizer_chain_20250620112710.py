"""토크나이저 체인 - Prototype02 + 언어별 토크나이저 순차 처리"""
import logging
from typing import List, Optional, Dict, Any
from .text_alignment import TextMasker, SourceTextSplitter
from .tokenizers import MecabTokenizer, JiebaTokenizer, HanjaHangulTokenizer

logger = logging.getLogger(__name__)

class TokenizerChain:
    """Prototype02 전처리 + 언어별 토크나이저 체인"""
    
    def __init__(self, language_tokenizer_type: Optional[str] = None):
        # 1단계: Prototype02 공통 전처리
        self.text_masker = TextMasker()
        self.src_splitter = SourceTextSplitter()
        
        # 2단계: 언어별 토크나이저 (선택적)
        self.language_tokenizer = None
        if language_tokenizer_type:
            self.language_tokenizer = self._create_language_tokenizer(language_tokenizer_type)
        
        logger.info(f"토크나이저 체인 초기화: Prototype02 + {language_tokenizer_type or 'None'}")
    
    def _create_language_tokenizer(self, tokenizer_type: str):
        """언어별 토크나이저 생성"""
        tokenizer_map = {
            'mecab': MecabTokenizer,
            'jieba': JiebaTokenizer,
            'hanja_hangul': HanjaHangulTokenizer
        }
        
        tokenizer_class = tokenizer_map.get(tokenizer_type)
        if tokenizer_class:
            try:
                return tokenizer_class()
            except Exception as e:
                logger.warning(f"언어별 토크나이저 초기화 실패 ({tokenizer_type}): {e}")
                return None
        else:
            logger.warning(f"알 수 없는 토크나이저 타입: {tokenizer_type}")
            return None
    
    def process_source_text(self, text: str) -> tuple[str, List[str], List[str]]:
        """원문 처리: Prototype02 → 언어별 토크나이저"""
        
        # 1단계: Prototype02 마스킹
        masked_text, masks = self.text_masker.mask(text, text_type="source")
        
        # 2단계: Prototype02 의미 단위 분할
        src_units = self.src_splitter.split(masked_text)
        if not src_units:
            src_units = [masked_text]
        
        # 3단계: 언어별 토크나이저 적용 (선택적)
        if self.language_tokenizer:
            refined_units = []
            for unit in src_units:
                try:
                    sub_tokens = self.language_tokenizer.tokenize(unit)
                    if sub_tokens and len(sub_tokens) > 1:
                        # 세분화된 토큰들을 추가
                        refined_units.extend(sub_tokens)
                    else:
                        # 세분화되지 않으면 원본 유지
                        refined_units.append(unit)
                except Exception as e:
                    logger.debug(f"언어별 토크나이징 실패: {e}")
                    refined_units.append(unit)
            src_units = refined_units
        
        return masked_text, masks, src_units
    
    def process_target_text(self, text: str) -> tuple[str, List[str]]:
        """번역문 처리: Prototype02 마스킹만 적용"""
        
        # 번역문은 Prototype02 마스킹만 적용
        # (의미 단위 분할은 나중에 DP 알고리즘에서 수행)
        masked_text, masks = self.text_masker.mask(text, text_type="target")
        return masked_text, masks
    
    def restore_masks(self, text: str, masks: List[str]) -> str:
        """마스크 복원"""
        return self.text_masker.restore(text, masks)