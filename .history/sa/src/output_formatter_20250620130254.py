"""구 단위 분할 출력 포맷터"""
import logging
import pandas as pd
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class PhraseUnitFormatter:
    """구 단위로 분할하여 출력하는 포맷터"""
    
    def __init__(self):
        pass
    
    def format_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """결과를 구 단위로 분할하여 새로운 DataFrame 생성"""
        
        logger.info("구 단위 분할 포맷팅 시작")
        
        formatted_rows = []
        
        for idx, row in df.iterrows():
            try:
                sentence_id = row.get('문장식별자', idx + 1)
                original_text = str(row.get('원문', ''))
                translation_text = str(row.get('번역문', ''))
                
                # 정렬 결과 가져오기
                aligned_source = str(row.get('aligned_source', ''))
                aligned_target = str(row.get('aligned_target', ''))
                
                # | 구분자로 분할
                source_phrases = self._split_phrases(aligned_source)
                target_phrases = self._split_phrases(aligned_target)
                
                # 길이 맞춤
                max_phrases = max(len(source_phrases), len(target_phrases))
                source_phrases = self._pad_phrases(source_phrases, max_phrases)
                target_phrases = self._pad_phrases(target_phrases, max_phrases)
                
                # 구 단위로 행 생성
                for phrase_idx, (src_phrase, tgt_phrase) in enumerate(zip(source_phrases, target_phrases)):
                    if src_phrase.strip() or tgt_phrase.strip():  # 빈 구는 제외
                        formatted_row = {
                            '문장식별자': sentence_id,
                            '구식별자': phrase_idx + 1,
                            '원문구': src_phrase.strip(),
                            '번역구': tgt_phrase.strip()
                        }
                        formatted_rows.append(formatted_row)
                
            except Exception as e:
                logger.error(f"행 {idx} 포맷팅 실패: {e}")
                # 실패시 원본 그대로 추가
                formatted_rows.append({
                    '문장식별자': sentence_id,
                    '구식별자': 1,
                    '원문구': original_text,
                    '번역구': translation_text
                })
        
        result_df = pd.DataFrame(formatted_rows)
        logger.info(f"구 단위 분할 완료: {len(result_df)}개 구 생성")
        
        return result_df
    
    def _split_phrases(self, text: str) -> List[str]:
        """| 구분자로 구 분할"""
        if not text or pd.isna(text):
            return []
        
        phrases = [phrase.strip() for phrase in str(text).split('|')]
        return [phrase for phrase in phrases if phrase]  # 빈 구 제거
    
    def _pad_phrases(self, phrases: List[str], target_length: int) -> List[str]:
        """구 목록을 목표 길이로 패딩"""
        while len(phrases) < target_length:
            phrases.append('')
        return phrases[:target_length]

class CompactFormatter:
    """기존 compact 형태 유지 포맷터"""
    
    def __init__(self):
        pass
    
    def format_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """기존 형태 그대로 반환"""
        return df

def get_formatter(format_type: str = 'phrase_unit'):
    """포맷터 팩토리"""
    if format_type == 'phrase_unit':
        return PhraseUnitFormatter()
    elif format_type == 'compact':
        return CompactFormatter()
    else:
        logger.warning(f"알 수 없는 포맷 타입: {format_type}, phrase_unit 사용")
        return PhraseUnitFormatter()