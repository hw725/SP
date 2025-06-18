"""보조 기능 모음: 괄호 마스킹, 의미 단위 분할 등"""
from typing import Dict, List, Tuple, Callable, Optional, Any
import regex
import numpy as np
import logging

from .structure_analysis import (
    analyze_unit_structure, analyze_merged_eojeols, 
    embed_with_structure_analysis, calculate_structure_similarity
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
    embed_func: Callable,
    source_analyzer=None,
    target_analyzer=None,
    default_embedding_dim: int = 1024  # 768 → 1024로 변경
) -> List[str]:
    """
    원문 단위 수에 맞춰 번역문을 분할하는 함수 (DP 알고리즘 사용)
    """
    if not src_units or not tgt_text.strip():
        return [""] * len(src_units) if src_units else []
    
    tgt_eojeols = tgt_text.strip().split()
    if not tgt_eojeols:
        return [""] * len(src_units)
    
    # 원문 단위들의 구조 분석
    src_analyses = []
    for unit in src_units:
        analysis = analyze_unit_structure(unit, source_analyzer, target_analyzer, "source")
        src_analyses.append(analysis)
    
    # 원문 단위들 임베딩
    src_embeddings_list = []
    try:
        src_embeddings_raw = embed_func(src_units)
        if src_embeddings_raw is not None and src_embeddings_raw.size > 0:
            if src_embeddings_raw.ndim == 1:
                src_embeddings_raw = src_embeddings_raw.reshape(1, -1)
            
            for i, analysis in enumerate(src_analyses):
                if i < len(src_embeddings_raw):
                    enhanced_emb = embed_with_structure_analysis(
                        src_units[i], analysis, embed_func
                    )
                    if enhanced_emb is not None:
                        src_embeddings_list.append(enhanced_emb)
                    else:
                        src_embeddings_list.append(src_embeddings_raw[i])
                else:
                    zero_emb = np.zeros(default_embedding_dim)
                    src_embeddings_list.append(zero_emb)
        else:
            for _ in src_units:
                src_embeddings_list.append(np.zeros(default_embedding_dim))
    except Exception as e:
        logger.warning(f"원문 임베딩 생성 실패: {e}")
        for _ in src_units:
            src_embeddings_list.append(np.zeros(default_embedding_dim))
    
    n_src = len(src_units)
    n_tgt = len(tgt_eojeols)
    
    # DP 테이블 초기화
    dp = [[-float('inf')] * (n_tgt + 1) for _ in range(n_src + 1)]
    parent = [[None] * (n_tgt + 1) for _ in range(n_src + 1)]
    dp[0][0] = 0.0
    
    # DP 계산
    for i in range(n_src + 1):
        for j in range(n_tgt + 1):
            if dp[i][j] == -float('inf'):
                continue
            
            # 원문 단위 i를 번역문 어절 j~k에 매칭
            if i < n_src:
                for k in range(j, n_tgt + 1):
                    if k == j:
                        # 빈 매칭 (원문 단위를 빈 번역으로)
                        score = dp[i][j] - 0.5
                        if dp[i+1][k] < score:
                            dp[i+1][k] = score
                            parent[i+1][k] = (i, j, k)
                    else:
                        # 실제 매칭
                        tgt_span = " ".join(tgt_eojeols[j:k])
                        similarity_score = calculate_matching_score(
                            src_units[i], tgt_span, 
                            src_embeddings_list[i], embed_func,
                            src_analyses[i], source_analyzer, target_analyzer
                        )
                        
                        # 길이 패널티 (너무 많은 어절을 한 번에 매칭하는 것 방지)
                        length_penalty = -0.1 * (k - j - 1) if (k - j) > 2 else 0
                        
                        score = dp[i][j] + similarity_score + length_penalty
                        if dp[i+1][k] < score:
                            dp[i+1][k] = score
                            parent[i+1][k] = (i, j, k)
    
    # 최적 경로 찾기 (모든 번역문 어절을 사용하는 것을 우선)
    best_score = -float('inf')
    best_j = n_tgt
    
    for j in range(n_tgt + 1):
        if dp[n_src][j] > best_score:
            best_score = dp[n_src][j]
            best_j = j
    
    # 경로 복원
    path = []
    curr_i, curr_j = n_src, best_j
    
    while parent[curr_i][curr_j] is not None:
        prev_i, start_j, end_j = parent[curr_i][curr_j]
        path.append((prev_i, start_j, end_j))
        curr_i, curr_j = prev_i, start_j
    
    path.reverse()
    
    # 결과 구성
    result = [""] * n_src
    for src_idx, start_j, end_j in path:
        if start_j < end_j:
            result[src_idx] = " ".join(tgt_eojeols[start_j:end_j])

    # --- 남은 어절 처리: 첫/마지막 의미단위에 붙이기 ---
    if path:
        # 첫 번째 의미단위 앞에 남은 어절이 있으면 prefix로 추가
        first_src, first_start, _ = path[0]
        if first_start > 0:
            prefix = " ".join(tgt_eojeols[0:first_start])
            result[first_src] = f"{prefix} {result[first_src]}".strip()

        # 마지막 의미단위 뒤에 남은 어절이 있으면 suffix로 추가
        last_src, _, last_end = path[-1]
        if last_end < n_tgt:
            suffix = " ".join(tgt_eojeols[last_end:n_tgt])
            result[last_src] = f"{result[last_src]} {suffix}".strip()
    else:
        # 경로가 아예 없으면 모든 어절을 마지막 단위에 할당
        all_span = " ".join(tgt_eojeols)
        result = [""] * (n_src - 1) + [all_span]
    
    return result

def calculate_matching_score(
    src_unit: str,
    tgt_span: str,
    src_embedding: np.ndarray,
    embed_func: Callable,
    src_analysis: Dict,
    source_analyzer=None,
    target_analyzer=None,
    weight_semantic: float = 0.8,    # 기본 가중치
    weight_structure: float = 0.15   # 기본 가중치
) -> float:
    """원문 단위와 번역문 스팬 간의 매칭 점수 계산"""
    if not tgt_span.strip():
        return -1.0  # 빈 매칭에 대한 패널티
    
    try:
        # 번역문 스팬 임베딩
        tgt_embeddings = embed_func([tgt_span])
        if tgt_embeddings is None or tgt_embeddings.size == 0:
            return 0.0
        
        if tgt_embeddings.ndim == 1:
            tgt_embeddings = tgt_embeddings.reshape(1, -1)
        
        tgt_embedding = tgt_embeddings[0]
        
        # 코사인 유사도 계산
        src_norm = np.linalg.norm(src_embedding)
        tgt_norm = np.linalg.norm(tgt_embedding)
        
        if src_norm == 0 or tgt_norm == 0:
            semantic_similarity = 0.0
        else:
            semantic_similarity = np.dot(src_embedding, tgt_embedding) / (src_norm * tgt_norm)
        
        # 구조적 유사도 계산
        tgt_eojeols = tgt_span.split()
        tgt_analysis = analyze_merged_eojeols(
            tgt_eojeols, source_analyzer, target_analyzer, "target"
        )
        
        structure_similarity = calculate_structure_similarity(src_analysis, tgt_analysis)
        
        # 최종 점수 (가중 평균) - 구조적 유사도 비중 증가
        final_score = weight_semantic * semantic_similarity \
                    + weight_structure * structure_similarity
        
        return max(final_score, 0.0)  # 음수 방지
        
    except Exception as e:
        logger.warning(f"매칭 점수 계산 실패: {e}")
        return 0.0

def split_tgt_meaning_units(
    tgt_text: str,
    src_units: List[str],
    embed_func: Callable,
    source_analyzer=None,
    target_analyzer=None,
    default_embedding_dim: int = 1024  # 768 → 1024로 변경
) -> List[str]:
    """번역문을 원문 단위에 맞춰 분할"""
    if not tgt_text or not src_units:
        return [""] * len(src_units) if src_units else []

    return split_tgt_by_src_units_with_eojeol_merge(
        src_units, tgt_text, embed_func,
        source_analyzer=source_analyzer,
        target_analyzer=target_analyzer,
        default_embedding_dim=default_embedding_dim
    )

class TextMasker:
    """텍스트 마스킹/언마스킹을 처리하는 클래스"""
    
    def __init__(self, **kwargs):
        # 필요한 설정이 있다면 여기서 처리
        pass
    
    def mask(self, text: str, text_type: str = "source") -> tuple:
        """텍스트 마스킹 (기존 mask_brackets 함수 사용)"""
        return mask_brackets(text, text_type)
    
    def unmask(self, text: str, mask_map: dict, unmask_type: str = 'original') -> str:
        """텍스트 언마스킹"""
        if unmask_type == 'original':
            return restore_masks(text, mask_map)
        elif unmask_type == 'remove_all_parentheses':
            # 마스킹된 부분을 완전히 제거
            result = text
            for mask_key in mask_map.keys():
                result = result.replace(mask_key, '')
            return result
        else:
            # 기본적으로 원본 복원
            return restore_masks(text, mask_map)