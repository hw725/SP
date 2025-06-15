"""정렬 관련 유틸리티 함수들"""
import numpy as np
from typing import List, Tuple

def calculate_alignment_matrix(src_embeddings, tgt_embeddings):
    """임베딩 간의 유사도 매트릭스 계산"""
    if src_embeddings.size == 0 or tgt_embeddings.size == 0:
        return np.array([])
    
    # 코사인 유사도 계산
    src_norm = np.linalg.norm(src_embeddings, axis=1, keepdims=True)
    tgt_norm = np.linalg.norm(tgt_embeddings, axis=1, keepdims=True)
    
    src_normalized = src_embeddings / np.maximum(src_norm, 1e-12)
    tgt_normalized = tgt_embeddings / np.maximum(tgt_norm, 1e-12)
    
    similarity_matrix = np.dot(src_normalized, tgt_normalized.T)
    return similarity_matrix

def pad_alignment_with_deduplication(src_units, tgt_units):
    """정렬 결과를 패딩하고 중복 제거"""
    if not src_units and not tgt_units:
        return []
    
    max_len = max(len(src_units), len(tgt_units))
    
    # 패딩
    padded_src = src_units + [""] * (max_len - len(src_units))
    padded_tgt = tgt_units + [""] * (max_len - len(tgt_units))
    
    # 쌍으로 만들기
    pairs = list(zip(padded_src, padded_tgt))
    
    # 중복 제거
    seen = set()
    deduplicated = []
    for pair in pairs:
        if pair not in seen:
            seen.add(pair)
            deduplicated.append(pair)
    
    return deduplicated

def verify_and_deduplicate_alignment(aligned_pairs):
    """정렬 결과 검증 및 중복 제거"""
    if not aligned_pairs:
        return []
    
    # 중복 제거
    seen = set()
    deduplicated = []
    for pair in aligned_pairs:
        if pair not in seen:
            seen.add(pair)
            deduplicated.append(pair)
    
    return deduplicated