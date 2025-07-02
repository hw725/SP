"""SA 토크나이저 모듈 패키지"""

from .jieba_mecab import (
    # [ACTIVE] 현재 사용되는 핵심 함수들
    split_src_meaning_units,              # 원문 분할 (jieba 기반)
    split_tgt_meaning_units_sequential,   # 🆕 번역문 순차 분할 (메인 함수)
    split_tgt_by_src_units_semantic,      # 의미 기반 분할 (폴백용)
    
    # [ACTIVE] 문법적 표지 관련 함수들
    is_boundary_marker,
    get_boundary_strength,
    should_attach_to_previous,
    
    # [ACTIVE] 기본 텍스트 처리 함수들  
    tokenize_text,
    pos_tag_text,
    sentence_split,
    split_inside_chunk,
    
    # [DEPRECATED] 하위 호환성을 위해 유지
    split_tgt_meaning_units,              # → split_tgt_meaning_units_sequential로 대체됨
    split_tgt_by_src_units,               # → split_tgt_by_src_units_semantic으로 대체됨
)

# BERT 토크나이저 (선택적 사용)
try:
    from .bert_tokenizer import (
        split_src_meaning_units as bert_split_src_meaning_units, 
        split_src_sentences as bert_split_src_sentences
    )
except ImportError:
    # BERT 의존성이 없는 경우 무시
    bert_split_src_meaning_units = None
    bert_split_src_sentences = None

__all__ = [
    # [ACTIVE] 핵심 분할 함수들
    'split_src_meaning_units',
    'split_tgt_meaning_units_sequential',  # 🆕 메인 번역문 분할 함수
    'split_tgt_by_src_units_semantic',
    
    # [ACTIVE] 문법적 표지 함수들
    'is_boundary_marker',
    'get_boundary_strength', 
    'should_attach_to_previous',
    
    # [ACTIVE] 기본 텍스트 처리
    'tokenize_text',
    'pos_tag_text', 
    'sentence_split',
    'split_inside_chunk',
    
    # [DEPRECATED] 하위 호환성
    'split_tgt_meaning_units',
    'split_tgt_by_src_units',
]