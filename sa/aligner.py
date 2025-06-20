"""토큰 정렬 모듈 - regex 지원"""

import numpy as np
import re
import regex  # 🆕 유니코드 속성 정규식
from typing import List, Dict, Tuple, Optional, Callable, Any
import logging
from sa_embedders import compute_embeddings_with_cache  # 🔧 수정

logger = logging.getLogger(__name__)

def align_tokens_with_embeddings(
    src_units: List[str], 
    tgt_units: List[str], 
    embed_func: Callable = None,  # 🔧 파라미터 추가
    similarity_threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """토큰 정렬 함수 - embed_func 파라미터 지원"""
    
    if not src_units or not tgt_units:
        return []
    
    # embed_func가 없으면 기본 임베더 사용
    if embed_func is None:
        from sa_embedders import compute_embeddings_with_cache
        embed_func = compute_embeddings_with_cache
    
    try:
        # 임베딩 생성
        src_embeddings = embed_func(src_units)
        tgt_embeddings = embed_func(tgt_units)
        
        alignments = []
        
        # 각 원문 단위에 대해 최고 매칭 찾기
        for i, src_unit in enumerate(src_units):
            src_emb = src_embeddings[i]
            
            best_score = -1.0
            best_tgt_idx = -1
            
            for j, tgt_unit in enumerate(tgt_units):
                tgt_emb = tgt_embeddings[j]
                
                # 코사인 유사도 계산
                similarity = np.dot(src_emb, tgt_emb) / (
                    np.linalg.norm(src_emb) * np.linalg.norm(tgt_emb) + 1e-8
                )
                
                if similarity > best_score and similarity >= similarity_threshold:
                    best_score = similarity
                    best_tgt_idx = j
            
            # 정렬 결과 추가
            if best_tgt_idx != -1:
                alignment = {
                    'src_idx': i,
                    'tgt_idx': best_tgt_idx,
                    'src': src_unit,
                    'tgt': tgt_units[best_tgt_idx],
                    'score': float(best_score)
                }
                alignments.append(alignment)
        
        logger.info(f"✅ 정렬 완료: {len(src_units)} → {len(tgt_units)} ({len(alignments)}개 정렬)")
        return alignments
        
    except Exception as e:
        logger.error(f"❌ 정렬 실패: {e}")
        return []

def _calculate_enhanced_confidence(src_text: str, tgt_text: str, similarity_matrix: np.ndarray) -> float:
    """강화된 신뢰도 계산"""
    
    base_confidence = np.max(similarity_matrix) if similarity_matrix.size > 0 else 0.3
    
    # 한자 매칭 보너스
    han_bonus = _calculate_han_matching_bonus(src_text, tgt_text)
    
    # 길이 비율 보너스
    len_ratio = min(len(src_text), len(tgt_text)) / max(len(src_text), len(tgt_text)) if max(len(src_text), len(tgt_text)) > 0 else 0
    length_bonus = len_ratio * 0.1
    
    return min(1.0, base_confidence + han_bonus + length_bonus)

def _calculate_han_matching_bonus(src_text: str, tgt_text: str) -> float:
    """🆕 한자 매칭 보너스 계산"""
    
    try:
        # 원문에서 한자 추출
        src_han = set(regex.findall(r'\p{Han}', src_text))
        # 번역문에서 한자 추출 
        tgt_han = set(regex.findall(r'\p{Han}', tgt_text))
        
        if not src_han:
            return 0.0
        
        # 한자 일치율 계산
        common_han = src_han & tgt_han
        if common_han:
            match_ratio = len(common_han) / len(src_han)
            return match_ratio * 0.3  # 최대 0.3 보너스
        
        return 0.0
        
    except Exception as e:
        logger.debug(f"한자 매칭 보너스 계산 실패: {e}")
        return 0.0

def _fallback_alignment(src_units: List[str], tgt_units: List[str]) -> List[Dict]:
    """백업 정렬"""
    
    alignments = []
    min_len = min(len(src_units), len(tgt_units))
    
    for i in range(min_len):
        # 🆕 백업에서도 한자 매칭 시도
        han_bonus = _calculate_han_matching_bonus(src_units[i], tgt_units[i])
        confidence = 0.3 + han_bonus
        
        alignments.append({
            'src_idx': i,
            'tgt_idx': i,
            'src_text': src_units[i],
            'tgt_text': tgt_units[i],
            'confidence': float(confidence),
            'alignment_type': '1:1-fallback'
        })
    
    return alignments

# 기존 함수와의 호환성을 위한 wrapper
def align_tokens(src_units: List[str], tgt_units: List[str], embed_func: Callable = None) -> List[Dict[str, Any]]:
    """processor.py 호환용 wrapper"""
    return align_tokens_with_embeddings(src_units, tgt_units, embed_func=embed_func)