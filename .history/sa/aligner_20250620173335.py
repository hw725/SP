"""토큰 정렬 모듈 - regex 지원"""

import numpy as np
import re
import regex  # 🆕 유니코드 속성 정규식
from typing import List, Dict, Tuple, Optional
import logging
from sa_embedders import compute_embeddings_with_cache  # 🔧 수정

logger = logging.getLogger(__name__)

def align_tokens_with_embeddings(
    src_units: List[str],
    tgt_units: List[str],
    src_text: str = "",
    tgt_text: str = "",
    threshold: float = 0.3
) -> List[Dict]:
    """임베딩 기반 토큰 정렬"""
    
    try:
        if not src_units or not tgt_units:
            logger.warning("⚠️ 빈 토큰 리스트")
            return []
        
        logger.info(f"🔗 토큰 정렬 시작: {len(src_units)} → {len(tgt_units)}")
        
        # 유사도 매트릭스 계산
        similarity_matrix = batch_similarity(src_units, tgt_units)
        
        if similarity_matrix.size == 0:
            logger.error("❌ 유사도 매트릭스 계산 실패")
            return _fallback_alignment(src_units, tgt_units)
        
        alignments = []
        
        # 정렬 로직 + 한자 매칭 가중치
        if len(src_units) == 1 and len(tgt_units) > 1:
            # 1:N 정렬
            tgt_combined = ' '.join(tgt_units)
            confidence = _calculate_enhanced_confidence(
                src_units[0], tgt_combined, similarity_matrix
            )
            
            alignments.append({
                'src_idx': 0,
                'tgt_idx': list(range(len(tgt_units))),
                'src_text': src_units[0],
                'tgt_text': tgt_combined,
                'confidence': float(confidence),
                'alignment_type': f'1:{len(tgt_units)}'
            })
            
        elif len(src_units) > 1 and len(tgt_units) == 1:
            # N:1 정렬
            src_combined = ' '.join(src_units)
            confidence = _calculate_enhanced_confidence(
                src_combined, tgt_units[0], similarity_matrix
            )
            
            alignments.append({
                'src_idx': list(range(len(src_units))),
                'tgt_idx': 0,
                'src_text': src_combined,
                'tgt_text': tgt_units[0],
                'confidence': float(confidence),
                'alignment_type': f'{len(src_units)}:1'
            })
            
        else:
            # 1:1 정렬 with 한자 매칭 보너스
            min_len = min(len(src_units), len(tgt_units))
            
            for i in range(min_len):
                base_confidence = similarity_matrix[i, i] if (i < similarity_matrix.shape[0] and i < similarity_matrix.shape[1]) else 0.3
                
                # 🆕 한자 매칭 보너스
                han_bonus = _calculate_han_matching_bonus(src_units[i], tgt_units[i])
                final_confidence = min(1.0, base_confidence + han_bonus)
                
                alignments.append({
                    'src_idx': i,
                    'tgt_idx': i,
                    'src_text': src_units[i],
                    'tgt_text': tgt_units[i],
                    'confidence': float(final_confidence),
                    'alignment_type': '1:1'
                })
        
        logger.info(f"✅ 정렬 완료: {len(alignments)}개 쌍")
        return alignments
        
    except Exception as e:
        logger.error(f"❌ 토큰 정렬 실패: {e}")
        return _fallback_alignment(src_units, tgt_units)

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

# 함수명을 processor.py에서 호출하는 이름과 맞춤
def align_tokens(src_units, tgt_units, embed_func=None):
    """processor.py 호환용 wrapper"""
    return align_tokens_with_embeddings(src_units, tgt_units, embed_func=embed_func)