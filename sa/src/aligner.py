# src/aligner.py
"""
의미적 유사도를 기반으로 원문(source)과 번역문(target)을 정렬하는 로직.
"""
from typing import List, Tuple, Any, Callable, Set
import numpy as np
import logging
from abc import ABC, abstractmethod # components.BaseAligner 대신 직접 ABC 사용 가능 (일관성 위해 components 것 사용 권장)
from .components import BaseAligner # components의 BaseAligner 사용
from .alignment_utils import pad_alignment_with_deduplication, verify_and_deduplicate_alignment # 필요시 사용

logger = logging.getLogger(__name__)

def calculate_alignment_matrix_internal(src_embs: np.ndarray, tgt_embs: np.ndarray, batch_size=128) -> np.ndarray:
    """유사도 매트릭스 계산 (aligner 내부용)"""
    if src_embs.ndim == 1: src_embs = np.expand_dims(src_embs, axis=0)
    if tgt_embs.ndim == 1: tgt_embs = np.expand_dims(tgt_embs, axis=0)
    if src_embs.size == 0 or tgt_embs.size == 0:
        return np.array([])

    src_norms = np.linalg.norm(src_embs, axis=1, keepdims=True)
    tgt_norms = np.linalg.norm(tgt_embs, axis=1, keepdims=True)
    
    src_embs_norm = np.divide(src_embs, src_norms, out=np.zeros_like(src_embs), where=src_norms!=0)
    tgt_embs_norm = np.divide(tgt_embs, tgt_norms, out=np.zeros_like(tgt_embs), where=tgt_norms!=0)
    
    num_src, num_tgt = src_embs_norm.shape[0], tgt_embs_norm.shape[0]
    similarity_matrix = np.zeros((num_src, num_tgt))
    
    for i in range(0, num_src, batch_size):
        src_batch_end = min(i + batch_size, num_src)
        src_batch = src_embs_norm[i:src_batch_end]
        for j in range(0, num_tgt, batch_size):
            tgt_batch_end = min(j + batch_size, num_tgt)
            tgt_batch = tgt_embs_norm[j:tgt_batch_end]
            batch_sim = np.dot(src_batch, tgt_batch.T)
            similarity_matrix[i:src_batch_end, j:tgt_batch_end] = batch_sim
    return similarity_matrix

class StrictAligner(BaseAligner):
    def align(self, src_units: List[str], tgt_units: List[str], embed_func: Callable) -> List[Tuple[str, str]]:
        # StrictAligner는 src_units와 tgt_units의 길이가 같다고 가정하고 1:1 매칭
        # pipeline.py에서 이미 tgt_units가 src_units 길이에 맞춰 조정됨
        if len(src_units) != len(tgt_units):
            logger.warning(f"StrictAligner: 원문({len(src_units)})과 번역문({len(tgt_units)}) 유닛 길이가 다릅니다. 패딩 정렬 시도.")
            return pad_alignment_with_deduplication(src_units, tgt_units) # alignment_utils 사용
        return verify_and_deduplicate_alignment(list(zip(src_units, tgt_units))) # alignment_utils 사용

class DPAligner(BaseAligner):
    def __init__(self, min_similarity: float = 0.1, gap_penalty: float = -0.5):
        self.min_similarity = min_similarity # 현재 DP 로직에서 직접 사용 안함
        self.gap_penalty = gap_penalty
    
    def align(self, src_units: List[str], tgt_units: List[str], embed_func: Callable) -> List[Tuple[str, str]]:
        if not src_units or not tgt_units:
            return pad_alignment_with_deduplication(src_units, tgt_units)
        
        src_embs = embed_func(src_units)
        tgt_embs = embed_func(tgt_units)

        if src_embs is None or tgt_embs is None or src_embs.size == 0 or tgt_embs.size == 0:
            logger.warning("DPAligner: 임베딩 생성 실패 또는 빈 임베딩. 패딩 정렬 시도.")
            return pad_alignment_with_deduplication(src_units, tgt_units)
        
        similarity_matrix = calculate_alignment_matrix_internal(src_embs, tgt_embs)
        if similarity_matrix.size == 0:
            logger.warning("DPAligner: 유사도 행렬 계산 실패. 패딩 정렬 시도.")
            return pad_alignment_with_deduplication(src_units, tgt_units)

        N, M = similarity_matrix.shape
        
        dp = np.full((N + 1, M + 1), -np.inf)
        path = {} 
        dp[0,0] = 0.0
        
        for i in range(N + 1):
            for j in range(M + 1):
                if i == 0 and j == 0: continue
                match_score = dp[i-1, j-1] + similarity_matrix[i-1, j-1] if i > 0 and j > 0 else -np.inf
                delete_score = dp[i-1, j] + self.gap_penalty if i > 0 else -np.inf
                insert_score = dp[i, j-1] + self.gap_penalty if j > 0 else -np.inf
                
                if i == 0: # 첫 행, insert만 가능
                    dp[i,j] = insert_score; path[(i,j)] = (i, j-1, 'insert_tgt')
                elif j == 0: # 첫 열, delete만 가능
                    dp[i,j] = delete_score; path[(i,j)] = (i-1, j, 'delete_src')
                elif match_score >= delete_score and match_score >= insert_score:
                    dp[i,j] = match_score; path[(i,j)] = (i-1, j-1, 'match')
                elif delete_score >= insert_score:
                    dp[i,j] = delete_score; path[(i,j)] = (i-1, j, 'delete_src')
                else:
                    dp[i,j] = insert_score; path[(i,j)] = (i, j-1, 'insert_tgt')
        
        aligned_pairs_reversed = []
        curr_i, curr_j = N, M
        while curr_i > 0 or curr_j > 0:
            if (curr_i, curr_j) not in path: break 
            prev_i, prev_j, action = path[(curr_i, curr_j)]
            if action == 'match': aligned_pairs_reversed.append((src_units[curr_i-1], tgt_units[curr_j-1]))
            elif action == 'delete_src': aligned_pairs_reversed.append((src_units[curr_i-1], ""))
            elif action == 'insert_tgt': aligned_pairs_reversed.append(("", tgt_units[curr_j-1]))
            curr_i, curr_j = prev_i, prev_j
        return verify_and_deduplicate_alignment(aligned_pairs_reversed[::-1])


class GreedyAligner(BaseAligner):
    def __init__(self, similarity_threshold: float = 0.4):
        self.similarity_threshold = similarity_threshold
    
    def align(self, src_units: List[str], tgt_units: List[str], embed_func: Callable) -> List[Tuple[str, str]]:
        if not src_units or not tgt_units:
            return pad_alignment_with_deduplication(src_units, tgt_units)
        
        src_embs = embed_func(src_units)
        tgt_embs = embed_func(tgt_units)

        if src_embs is None or tgt_embs is None or src_embs.size == 0 or tgt_embs.size == 0:
            logger.warning("GreedyAligner: 임베딩 생성 실패. 패딩 정렬 시도.")
            return pad_alignment_with_deduplication(src_units, tgt_units)

        similarity_matrix = calculate_alignment_matrix_internal(src_embs, tgt_embs)
        if similarity_matrix.size == 0 or similarity_matrix.shape[0] != len(src_units) or similarity_matrix.shape[1] != len(tgt_units):
            logger.warning("GreedyAligner: 유사도 행렬 계산 실패 또는 차원 불일치. 패딩 정렬 시도.")
            return pad_alignment_with_deduplication(src_units, tgt_units)
        
        aligned_pairs: List[Tuple[str, str]] = [("", "")] * len(src_units) # 원문 길이만큼 결과 생성
        used_tgt_indices: Set[int] = set()
        
        # 각 원문 유닛에 대해 최적의 번역문 유닛 탐욕적 선택
        for i in range(len(src_units)):
            best_j = -1
            max_sim = -1.0 
            for j in range(len(tgt_units)):
                if j in used_tgt_indices: continue
                current_sim = similarity_matrix[i, j]
                if current_sim > max_sim and current_sim >= self.similarity_threshold:
                    max_sim = current_sim
                    best_j = j
            
            if best_j != -1:
                aligned_pairs[i] = (src_units[i], tgt_units[best_j])
                used_tgt_indices.add(best_j)
            else:
                aligned_pairs[i] = (src_units[i], "") # 매칭 실패 시 빈 문자열

        # 매칭되지 않은 타겟 유닛 처리 (선택적):
        # 현재는 매칭되지 않은 타겟은 버려짐. 만약 이를 마지막 src 유닛의 타겟에 추가하고 싶다면,
        # for j in range(len(tgt_units)):
        #     if j not in used_tgt_indices:
        #         # 마지막 src 유닛 또는 가장 유사도 높았던 src 유닛의 타겟에 추가
        #         # 예: aligned_pairs[-1] = (aligned_pairs[-1][0], aligned_pairs[-1][1] + " " + tgt_units[j])
        #         pass # 여기서는 단순화 위해 생략
        
        return verify_and_deduplicate_alignment(aligned_pairs)