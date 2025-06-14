"""원문과 번역문 구 간의 정렬 알고리즘 모듈"""

import logging
import numpy as np
from typing import List, Tuple, Any, Callable

logger = logging.getLogger(__name__)

def cosine_similarity(vec1: Any, vec2: Any) -> float:
    """Calculate cosine similarity (handling zero vectors)."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))

def align_src_tgt(
    src_units: List[str], 
    tgt_units: List[str], 
    embed_func: Callable
) -> List[Tuple[str, str]]:
    """Align source and target units."""
    logger.info(f"Source units: {len(src_units)} items, Target units: {len(tgt_units)} items")

    if len(src_units) != len(tgt_units):
        try:
            # 지연 임포트로 순환 참조 방지
            from tokenizer import split_tgt_by_src_units_semantic
            
            flatten_tgt = " ".join(tgt_units)
            new_tgt_units = split_tgt_by_src_units_semantic(
                src_units, flatten_tgt, embed_func, min_tokens=1
            )
            if len(new_tgt_units) == len(src_units):
                logger.info("Semantic re-alignment successful")
                return list(zip(src_units, new_tgt_units))
            else:
                logger.warning(f"Length mismatch after re-alignment: Source={len(src_units)}, Target={len(new_tgt_units)}")
        except Exception as e:
            logger.error(f"Error during semantic re-alignment: {e}")

        # 길이가 맞지 않으면 패딩
        if len(src_units) > len(tgt_units):
            tgt_units.extend([""] * (len(src_units) - len(tgt_units)))
        else:
            src_units.extend([""] * (len(tgt_units) - len(src_units)))

    return list(zip(src_units, tgt_units))

def calculate_alignment_matrix(src_embs, tgt_embs, batch_size=512):
    """Optimized function for calculating large similarity matrices."""
    src_len, tgt_len = len(src_embs), len(tgt_embs)
    similarity_matrix = np.zeros((src_len, tgt_len))

    for i in range(0, src_len, batch_size):
        batch_src = src_embs[i:i + batch_size]
        for j in range(0, tgt_len, batch_size):
            batch_tgt = tgt_embs[j:j + batch_size]
            batch_src_norm = np.linalg.norm(batch_src, axis=1, keepdims=True)
            batch_tgt_norm = np.linalg.norm(batch_tgt, axis=1, keepdims=True)

            dots = np.matmul(batch_src, batch_tgt.T)
            norms = np.matmul(batch_src_norm, batch_tgt_norm.T)
            batch_sim = dots / (norms + 1e-8)

            similarity_matrix[i:i + batch_size, j:j + batch_size] = batch_sim

    return similarity_matrix

def align_with_dynamic_programming(
    src_units: List[str],
    tgt_units: List[str],
    embed_func: Callable,
    min_similarity: float = 0.3
) -> List[Tuple[str, str]]:
    """
    동적 프로그래밍을 사용한 고급 정렬 알고리즘
    
    Args:
        src_units: 원문 단위 리스트
        tgt_units: 번역문 단위 리스트
        embed_func: 임베딩 함수
        min_similarity: 최소 유사도 임계값
    
    Returns:
        정렬된 (원문, 번역문) 튜플 리스트
    """
    if not src_units or not tgt_units:
        return []
    
    # 임베딩 계산
    src_embs = embed_func(src_units)
    tgt_embs = embed_func(tgt_units)
    
    # 유사도 매트릭스 계산
    similarity_matrix = calculate_alignment_matrix(src_embs, tgt_embs)
    
    # DP 테이블 초기화
    m, n = len(src_units), len(tgt_units)
    dp = np.full((m + 1, n + 1), -np.inf)
    backtrack = np.zeros((m + 1, n + 1, 2), dtype=int)
    
    dp[0, 0] = 0.0
    
    # DP 계산
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 and j == 0:
                continue
                
            # 원문 단위 건너뛰기 (삭제)
            if i > 0 and dp[i-1, j] > dp[i, j]:
                dp[i, j] = dp[i-1, j] - 0.1  # 패널티
                backtrack[i, j] = [i-1, j]
            
            # 번역문 단위 건너뛰기 (삽입)
            if j > 0 and dp[i, j-1] > dp[i, j]:
                dp[i, j] = dp[i, j-1] - 0.1  # 패널티
                backtrack[i, j] = [i, j-1]
            
            # 매칭
            if i > 0 and j > 0:
                sim_score = similarity_matrix[i-1, j-1]
                if sim_score >= min_similarity:
                    match_score = dp[i-1, j-1] + sim_score
                    if match_score > dp[i, j]:
                        dp[i, j] = match_score
                        backtrack[i, j] = [i-1, j-1]
    
    # 백트래킹으로 정렬 경로 찾기
    aligned_pairs = []
    i, j = m, n
    
    while i > 0 or j > 0:
        prev_i, prev_j = backtrack[i, j]
        
        if prev_i == i - 1 and prev_j == j - 1:
            # 매칭
            aligned_pairs.append((src_units[i-1], tgt_units[j-1]))
        elif prev_i == i - 1:
            # 원문만 (번역문 없음)
            aligned_pairs.append((src_units[i-1], ""))
        else:
            # 번역문만 (원문 없음)
            aligned_pairs.append(("", tgt_units[j-1]))
        
        i, j = prev_i, prev_j
    
    return aligned_pairs[::-1]

def align_with_greedy_matching(
    src_units: List[str],
    tgt_units: List[str], 
    embed_func: Callable,
    similarity_threshold: float = 0.4
) -> List[Tuple[str, str]]:
    """
    탐욕적 매칭 알고리즘을 사용한 정렬
    
    Args:
        src_units: 원문 단위 리스트
        tgt_units: 번역문 단위 리스트
        embed_func: 임베딩 함수
        similarity_threshold: 유사도 임계값
        
    Returns:
        정렬된 (원문, 번역문) 튜플 리스트
    """
    if not src_units or not tgt_units:
        return []
    
    # 임베딩 계산
    src_embs = embed_func(src_units)
    tgt_embs = embed_func(tgt_units)
    
    # 유사도 매트릭스 계산
    similarity_matrix = calculate_alignment_matrix(src_embs, tgt_embs)
    
    aligned_pairs = []
    used_tgt = set()
    
    for i, src_unit in enumerate(src_units):
        best_j = -1
        best_score = similarity_threshold
        
        for j, tgt_unit in enumerate(tgt_units):
            if j in used_tgt:
                continue
                
            score = similarity_matrix[i, j]
            if score > best_score:
                best_score = score
                best_j = j
        
        if best_j >= 0:
            aligned_pairs.append((src_unit, tgt_units[best_j]))
            used_tgt.add(best_j)
        else:
            aligned_pairs.append((src_unit, ""))
    
    # 매칭되지 않은 번역문 단위 추가
    for j, tgt_unit in enumerate(tgt_units):
        if j not in used_tgt:
            aligned_pairs.append(("", tgt_unit))
    
    return aligned_pairs