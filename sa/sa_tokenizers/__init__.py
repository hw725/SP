"""SA 토크나이저 모듈 패키지"""

from .jieba_mecab import (
    # 주요 분할 함수들
    split_tgt_meaning_units,  # 번역문 분할 (기존 호환성)
    split_tgt_meaning_units_sequential,  # 🆕 순차 분할 방식 (메인)
    split_tgt_by_src_units,  # 단순 분할
    split_tgt_by_src_units_semantic,  # 의미 기반 분할 (순차로 대체됨)
    
    # 기본 텍스트 처리 함수들  
    tokenize_text,
    pos_tag_text,
    sentence_split,
    
    # 🆕 문법적 표지 관련 함수들
    is_boundary_marker,
    get_boundary_strength
)
from .bert_tokenizer import split_src_meaning_units as bert_split_src_meaning_units, split_src_sentences as bert_split_src_sentences

def split_src_meaning_units(text: str, *args, **kwargs):
    """원문(한문)은 무조건 jieba로 의미 단위 분할 (tokenizer 인자 무시)"""
    from .jieba_mecab import split_src_meaning_units as jieba_split
    return jieba_split(text, *args, **kwargs)

__all__ = [
    # 주요 분할 함수들
    'split_src_meaning_units',
    'split_tgt_meaning_units', 
    'split_tgt_meaning_units_sequential',  # 🆕 순차 분할 (메인)
    'split_tgt_by_src_units',
    'split_tgt_by_src_units_semantic',
    
    # 기본 텍스트 처리
    'tokenize_text',
    'pos_tag_text', 
    'sentence_split',
    
    # 🆕 문법적 표지 함수들
    'is_boundary_marker',
    'get_boundary_strength'
]