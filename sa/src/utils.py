# src/utils.py
"""보조 기능 모음: 파일 I/O, 괄호 마스킹, 의미 단위 분할 등"""
import pandas as pd
import regex
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any
import logging

from .structure_analysis import (
    analyze_unit_structure, analyze_merged_eojeols, 
    embed_with_structure_analysis, calculate_structure_similarity
)
from .alignment_utils import ( # 현재 사용되지 않는 함수들은 제거 가능
    calculate_alignment_matrix as util_calc_align_matrix, # aligner.py의 것과 이름 충돌 방지
    pad_alignment_with_deduplication as util_pad_align, 
    verify_and_deduplicate_alignment as util_verify_dedup
)
from .file_io import load_data, save_data

logger = logging.getLogger(__name__)

def mask_brackets(text: str, text_type: str = "source") -> Tuple[str, Dict[str, str]]:
    """괄호 및 특수 기호 마스킹"""
    masks = {}
    mask_counter = 0
    
    # 괄호 패턴들
    patterns = [
        (r'\([^)]*\)', 'PAREN'),
        (r'\[[^\]]*\]', 'BRACKET'), 
        (r'\{[^}]*\}', 'BRACE'),
        (r'<[^>]*>', 'ANGLE')
    ]
    
    masked_text = text
    for pattern, prefix in patterns:
        def replace_func(match):
            nonlocal mask_counter
            mask_key = f"__{prefix}_{mask_counter}__"
            masks[mask_key] = match.group(0)
            mask_counter += 1
            return mask_key
        
        masked_text = regex.sub(pattern, replace_func, masked_text)
    
    return masked_text, masks

def restore_masks(text: str, masks: Dict[str, str]) -> str:
    """마스킹 복원"""
    restored_text = text
    for mask_key, original in masks.items():
        restored_text = restored_text.replace(mask_key, original)
    return restored_text

def split_src_meaning_units(text: str) -> List[str]:
    """한문 텍스트를 '한자+조사+어미' 단위로 묶어서 분할"""
    if not text.strip():
        return []
        
    text = text.replace('\n', ' ').replace('：', '： ')
    tokens = regex.findall(r'\S+', text)
    units: List[str] = []
    i = 0
    
    while i < len(tokens):
        current_unit = tokens[i]
        i += 1
        
        # 다음 토큰들이 조사나 어미인지 확인하여 결합
        while i < len(tokens):
            next_token = tokens[i]
            
            # 한자가 아닌 경우 (조사, 어미 등으로 판단)
            if not regex.match(r'^[\u4e00-\u9fff]+', next_token):
                current_unit += ' ' + next_token
                i += 1
            else:
                break
        
        units.append(current_unit.strip())
    
    return units if units else [text.strip()]

def split_tgt_by_src_units_with_eojeol_merge(
    src_units: List[str], 
    tgt_text: str, 
    embed_func: Callable, # embedder.embed 메서드
    source_analyzer=None,
    target_analyzer=None,
    default_embedding_dim: int = 768 # BGE-base, SRoBERTa 등 일반적 차원
) -> List[str]:
    if not src_units or not tgt_text.strip():
        return [""] * len(src_units) if src_units else []
    
    tgt_eojeols = tgt_text.strip().split()
    if not tgt_eojeols:
        return [""] * len(src_units)
    
    src_analyses = [analyze_unit_structure(unit, source_analyzer, target_analyzer, "source") for unit in src_units]
    
    src_embeddings_1d: List[np.ndarray] = []
    current_dim = default_embedding_dim # 초기 차원

    for i, unit in enumerate(src_units):
        emb_1d = embed_with_structure_analysis(unit, src_analyses[i], embed_func)
        if emb_1d is not None:
            src_embeddings_1d.append(emb_1d)
            if i == 0: # 첫 성공 임베딩에서 차원 업데이트
                current_dim = emb_1d.shape[0]
        else:
            logger.warning(f"원문 유닛 임베딩 실패: '{unit[:50]}...'. 0벡터 사용 (차원: {current_dim}).")
            src_embeddings_1d.append(np.zeros(current_dim)) 
    
    if not src_embeddings_1d:
        logger.error("모든 원문 유닛 임베딩 실패. DP 분할 불가.")
        # 모든 타겟 어절을 하나의 문자열로 합쳐 첫 번째 유닛에 할당하거나, 빈 문자열 리스트 반환
        return [" ".join(tgt_eojeols)] + [""] * (len(src_units) -1) if src_units else []


    n_src = len(src_units)
    n_tgt = len(tgt_eojeols)
    dp = [[-float('inf')] * (n_tgt + 1) for _ in range(n_src + 1)]
    parent = [[None] * (n_tgt + 1) for _ in range(n_src + 1)]
    dp[0][0] = 0.0
    
    for i in range(n_src + 1):
        for j in range(n_tgt + 1):
            if dp[i][j] == -float('inf'): continue
            
            if i < n_src: # Case 1: 현재 원문 단위를 빈 문자열로 매칭 (스킵)
                skip_penalty = -0.3 
                if dp[i + 1][j] < dp[i][j] + skip_penalty:
                    dp[i + 1][j] = dp[i][j] + skip_penalty
                    parent[i + 1][j] = (i, j, "skip_src")
            
            if i < n_src: # Case 2: 현재 원문 단위를 j부터 k-1까지의 번역문 어절들과 매칭
                max_span_length = min(n_tgt - j, max(5, (n_tgt // n_src if n_src > 0 else n_tgt) + 3))
                for span_len in range(1, max_span_length + 1):
                    k = j + span_len
                    if k > n_tgt: break
                    current_tgt_span_eojeols = tgt_eojeols[j:k]
                    current_tgt_span = " ".join(current_tgt_span_eojeols)
                    
                    tgt_span_analysis = analyze_merged_eojeols(current_tgt_span_eojeols, source_analyzer, target_analyzer, "target")
                    tgt_span_emb_1d = embed_with_structure_analysis(current_tgt_span, tgt_span_analysis, embed_func)
                    
                    if tgt_span_emb_1d is not None:
                        src_emb_current_1d = src_embeddings_1d[i]
                        if src_emb_current_1d.shape != tgt_span_emb_1d.shape:
                            logger.warning(f"DP: 임베딩 차원 불일치 Src({src_emb_current_1d.shape}) vs Tgt({tgt_span_emb_1d.shape}). 스킵. Src: {src_units[i][:20]}, Tgt: {current_tgt_span[:20]}")
                            continue # 차원이 다르면 유사도 계산 불가

                        norm_src = np.linalg.norm(src_emb_current_1d)
                        norm_tgt = np.linalg.norm(tgt_span_emb_1d)
                        semantic_sim = float(np.dot(src_emb_current_1d, tgt_span_emb_1d) / (norm_src * norm_tgt)) if norm_src > 0 and norm_tgt > 0 else 0.0
                        
                        structure_sim = calculate_structure_similarity(src_analyses[i], tgt_span_analysis)
                        length_penalty = -0.03 * max(0, span_len - 3) # 길이 패널티 조정
                        
                        total_score = 0.8 * semantic_sim + 0.15 * structure_sim + length_penalty
                        
                        if dp[i + 1][k] < dp[i][j] + total_score:
                            dp[i + 1][k] = dp[i][j] + total_score
                            parent[i + 1][k] = (i, j, f"match_{j}_{k}")
    
    best_j_final = n_tgt # 모든 번역문 어절을 사용하는 것을 우선
    if dp[n_src][n_tgt] == -float('inf'): # 모든 어절 사용 경로가 없다면 최적 경로 탐색
        best_score_final = -float('inf')
        for j_idx in range(n_tgt + 1):
            if dp[n_src][j_idx] > best_score_final:
                best_score_final = dp[n_src][j_idx]
                best_j_final = j_idx
    
    path = []
    curr_i_path, curr_j_path = n_src, best_j_final
    while parent[curr_i_path][curr_j_path] is not None:
        prev_i_path, prev_j_path, action_path = parent[curr_i_path][curr_j_path]
        if action_path == "skip_src":
            path.append((prev_i_path, "")) # 원문 인덱스와 빈 타겟
        elif action_path.startswith("match_"):
            _, start_j_str, end_j_str = action_path.split("_")
            matched_text = " ".join(tgt_eojeols[int(start_j_str):int(end_j_str)])
            path.append((prev_i_path, matched_text)) # 원문 인덱스와 매칭된 타겟
        curr_i_path, curr_j_path = prev_i_path, prev_j_path
    path.reverse()
    
    result = [""] * n_src
    # 경로에서 매칭된 타겟 채우기
    for src_idx_in_path, tgt_span_in_path in path:
        if 0 <= src_idx_in_path < n_src:
            result[src_idx_in_path] = tgt_span_in_path
            
    # DP 경로에서 사용된 타겟 어절 추적 (정확한 추적은 복잡, 여기서는 단순화)
    # 남은 어절을 마지막 비어있지 않은 유닛에 추가하는 방식 유지
    # 또는, DP 경로 복원 시 사용된 타겟 어절 범위를 정확히 기록하여 남은 어절을 더 정교하게 배분 가능
    
    # 현재는 경로 복원 후, 남은 어절을 배분하는 로직은 제거하고 DP 결과를 그대로 사용.
    # 만약 DP가 모든 타겟 어절을 커버하지 못하는 경우가 문제라면,
    # 남은 어절을 가장 마지막에 매칭된 타겟 유닛 뒤에 붙이거나,
    # 혹은 가장 유사도가 높았던 매칭 쌍 근처에 배분하는 등의 후처리 고려 가능.
    # 여기서는 DP 결과를 신뢰하고 그대로 반환.
    
    return result

def split_tgt_meaning_units(
    tgt_text: str,
    src_units: List[str],
    embed_func: Callable,
    source_analyzer=None,
    target_analyzer=None,
    default_embedding_dim: int = 768
) -> List[str]:
    if not tgt_text or not src_units:
        logger.warning("split_tgt_meaning_units: 입력값이 비어 있습니다.")
        return [""] * len(src_units) if src_units else []

    return split_tgt_by_src_units_with_eojeol_merge(
        src_units, tgt_text, embed_func,
        source_analyzer=source_analyzer,
        target_analyzer=target_analyzer,
        default_embedding_dim=default_embedding_dim
    )


