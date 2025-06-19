"""보조 기능 모음: 괄호 마스킹, 의미 단위 분할 등"""
from typing import Dict, List, Tuple, Callable, Optional, Any
import regex
import numpy as np
import logging
from typing import Callable, Dict, List, Any
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .structure_analysis import (
    analyze_unit_structure,
    analyze_merged_eojeols,
    embed_with_structure_analysis,
    calculate_structure_similarity
)

logger = logging.getLogger(__name__)

# ——————————————————————————————
# 1) Cross‐Encoder 모델 로드 (public 모델로 변경)
# ——————————————————————————————
_cross_encoder_name = "cross-encoder/stsb-roberta-base"  # 공개 모델
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    cross_tokenizer = AutoTokenizer.from_pretrained(_cross_encoder_name)
    cross_model     = AutoModelForSequenceClassification.from_pretrained(_cross_encoder_name)
    cross_model.to(device)
    cross_model.eval()
    weight_cross_default = 0.05
    logger.info(f"Loaded cross‐encoder '{_cross_encoder_name}' on {device}")
except Exception as e:
    logger.warning(f"Cross‐Encoder 로딩 실패('{_cross_encoder_name}'): {e}. 재순위 비활성화")
    cross_tokenizer = None
    cross_model     = None
    weight_cross_default = 0.0

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

# -------------------------------------------------
# split_tgt_meaning_units 래퍼 함수
# -------------------------------------------------
def split_tgt_meaning_units(
    src_units: List[str],
    masked_tgt: str,
    embed_func: Callable,
    source_analyzer=None,
    target_analyzer=None
) -> List[str]:
    """
    매개변수 이름 명시 불필요하도록 위치 매개변수로 정의
    """
    return split_tgt_by_src_units_with_eojeol_merge(
        src_units,
        masked_tgt,
        embed_func,
        source_analyzer,
        target_analyzer
    )

# ─────────────────────────────────────────
# TextMasker 클래스 (components.py에서 필요)
# ─────────────────────────────────────────
class TextMasker:
    def __init__(self, **kwargs):
        # 마스킹 옵션 등 필요 시 kwargs 처리
        pass

    def mask(self, text: str, text_type: str = "source"):
        # mask_brackets는 이미 이 파일에서 정의돼 있어야 합니다.
        return mask_brackets(text, text_type)

    def restore(self, text: str, masks):
        # restore_masks도 이 파일에 정의돼 있어야 합니다.
        return restore_masks(text, masks)

# ─────────────────────────────────────────────────
# 1) calculate_matching_score 함수 정의 추가
# ─────────────────────────────────────────────────
def calculate_matching_score(
    src_unit: str,
    tgt_span: str,
    src_embedding: np.ndarray,
    embed_func: Callable,
    src_analysis: Dict[str, Any],
    source_analyzer=None,
    target_analyzer=None,
    weight_semantic: float = 0.7,
    weight_structure: float = 0.15,
    weight_pos: float = 0.1,
    weight_cross: float = 0.05,
    cross_tok=None,
    cross_enc=None
) -> float:
    # 의미적 유사도
    tgt_embedding = embed_func([tgt_span])[0]
    semantic_similarity = np.dot(src_embedding, tgt_embedding) / (
        np.linalg.norm(src_embedding) * np.linalg.norm(tgt_embedding) + 1e-8
    )
    # 구조적 유사도
    tgt_analysis = analyze_unit_structure(tgt_span, source_analyzer, target_analyzer, "target")
    structure_similarity = calculate_structure_similarity(src_analysis, tgt_analysis)
    # POS 일치 보정 (옵션)
    pos_score = 0.0
    if source_analyzer and target_analyzer:
        src_pos = [p for _, p in source_analyzer.tag(src_unit)]
        tgt_pos = [p for _, p in target_analyzer.tag(tgt_span)]
        matches = sum(1 for a, b in zip(src_pos, tgt_pos) if a == b)
        pos_score = matches / max(len(src_pos), len(tgt_pos), 1)
    # Cross‐Encoder re‐rank (옵션)
    cross_score = 0.0
    if cross_tok and cross_enc and weight_cross > 0:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        inputs = cross_tok(
            src_unit, tgt_span,
            return_tensors="pt", truncation=True, max_length=128
        ).to(device)
        
        with torch.no_grad():
            logits = cross_enc(**inputs).logits
            
            # 로그 크기에 따른 분기 처리
            if logits.dim() == 2 and logits.size(1) > 1:
                # 분류 모델 (multiple classes)
                probs = torch.softmax(logits, dim=1)
                cross_score = probs[0, 1].item()
            else:
                # 회귀 모델 또는 단일 출력
                val = logits.view(-1)[0]
                cross_score = torch.sigmoid(val).item()
    
    # 최종 가중합
    final_score = (
          weight_semantic  * semantic_similarity
        + weight_structure * structure_similarity
        + weight_pos       * pos_score
        + weight_cross     * cross_score
    )
    return max(final_score, 0.0)

# ─────────────────────────────────────────────────
# 이하 split_tgt_by_src_units_with_eojeol_merge 등
# ─────────────────────────────────────────────────

def split_tgt_by_src_units_with_eojeol_merge(
    src_units: List[str],
    tgt_text: str,
    embed_func: Callable,
    source_analyzer=None,
    target_analyzer=None,
    weight_semantic: float = 0.6,
    weight_structure: float = 0.25,
    weight_pos: float = 0.1,
    weight_cross: float = 0.05,
    cross_tok=None,
    cross_enc=None,
    max_span_len: int = 5,        # ← 최대 스팬 길이 제한
    empty_penalty: float = 0.5,
    length_penalty: float = -0.1
) -> List[str]:
    tgt_eojeols = tgt_text.split()
    n_src, n_tgt = len(src_units), len(tgt_eojeols)
    if n_src == 0 or n_tgt == 0:
        return [""] * n_src

    # 1) src 임베딩 + 구조분석 한 번만
    src_embeds    = embed_func(src_units)  # (n_src, dim)
    src_analyses  = [
        analyze_unit_structure(u, source_analyzer, target_analyzer, "source")
        for u in src_units
    ]

    # 2) 가능한 tgt 스팬만 미리 만들고 배치 임베딩
    span_keys = []
    for j in range(n_tgt):
        for l in range(1, max_span_len+1):
            k = j + l
            if k > n_tgt: break
            span_keys.append((j, k))
    span_texts = [" ".join(tgt_eojeols[j:k]) for j,k in span_keys]
    span_embeds = embed_func(span_texts) if span_texts else np.empty((0, src_embeds.shape[1]))

    # map (j,k) → 임베딩 인덱스
    span_idx = {span_keys[i]: i for i in range(len(span_keys))}

    # 3) NumPy DP 초기화
    dp     = np.full((n_src+1, n_tgt+1), -np.inf, dtype=float)
    parent = np.zeros((n_src+1, n_tgt+1, 3), dtype=int)  # (prev_i, prev_j, end_j)
    dp[0,0] = 0.0

    # fast_score: 반복 비용을 줄이기 위한 래퍼
    def fast_score(i: int, j: int, k: int) -> float:
        src_unit     = src_units[i]
        src_emb      = src_embeds[i]
        src_ana      = src_analyses[i]
        tgt_span     = " ".join(tgt_eojeols[j:k])
        tgt_emb      = span_embeds[span_idx[(j,k)]]
        return calculate_matching_score(
            src_unit, tgt_span,
            src_emb, embed_func,
            src_ana, source_analyzer, target_analyzer,
            weight_semantic, weight_structure, weight_pos,
            weight_cross, cross_tok, cross_enc
        )

    # 4) DP loop (i,j)→(i+1,k)
    for i in range(n_src):
        row_dp = dp[i]         # 참조 횟수 최소화
        next_dp = dp[i+1]
        for j in range(n_tgt+1):
            base = row_dp[j]
            if base == -np.inf:
                continue
            # 4-1) 빈 매칭
            val0 = base - empty_penalty
            if val0 > next_dp[j]:
                next_dp[j] = val0
                parent[i+1,j] = (i, j, j)
            # 4-2) 실제 매칭 (길이 제한)
            for l in range(1, max_span_len+1):
                k = j + l
                if k > n_tgt:
                    break
                val = base + fast_score(i, j, k) + length_penalty * (l-1)
                if val > next_dp[k]:
                    next_dp[k] = val
                    parent[i+1,k] = (i, j, k)

    # 5) 경로 복원
    # 최적 종료점
    end_j = int(np.argmax(dp[n_src]))
    logger.info(f"[DEBUG] 최적 종료점: {end_j}, 전체 tgt 길이: {n_tgt}")
    
    aligned = ["" for _ in range(n_src)]
    i, j = n_src, end_j
    matched_ranges = []  # 매핑된 번역문 범위 추적
    
    while i > 0:
        pi, pj, pk = parent[i,j]
        if pk > pj:
            aligned[pi] = " ".join(tgt_eojeols[pj:pk])
            matched_ranges.append((pj, pk))
        i, j = pi, pj
    
    matched_ranges.sort()  # 범위를 순서대로 정렬
    
    # 누락된 번역문 처리
    if matched_ranges:
        # 첫 번째 매핑 이전의 번역문 (prefix)
        if matched_ranges[0][0] > 0:
            prefix = " ".join(tgt_eojeols[0:matched_ranges[0][0]])
            # 첫 번째 매핑된 원문에 추가
            first_mapped = next(i for i, text in enumerate(aligned) if text)
            aligned[first_mapped] = f"{prefix} {aligned[first_mapped]}".strip()
        
        # 마지막 매핑 이후의 번역문 (suffix)
        if matched_ranges[-1][1] < n_tgt:
            suffix = " ".join(tgt_eojeols[matched_ranges[-1][1]:n_tgt])
            # 마지막 매핑된 원문에 추가
            last_mapped = next(i for i in reversed(range(n_src)) if aligned[i])
            aligned[last_mapped] = f"{aligned[last_mapped]} {suffix}".strip()
        
        # 매핑 사이의 누락된 번역문 처리
        for i in range(len(matched_ranges) - 1):
            curr_end = matched_ranges[i][1]
            next_start = matched_ranges[i + 1][0]
            
            if curr_end < next_start:
                # 사이에 누락된 번역문이 있음
                gap_text = " ".join(tgt_eojeols[curr_end:next_start])
                # 다음 매핑에 추가 (또는 현재 매핑에 추가)
                next_mapped_idx = next(idx for idx, text in enumerate(aligned) if text and any((start, end) for start, end in matched_ranges if start == matched_ranges[i+1][0]))
                aligned[next_mapped_idx] = f"{gap_text} {aligned[next_mapped_idx]}".strip()
    
    else:
        # 아예 매핑이 안 된 경우 - 전체 번역문을 첫 번째 원문에 할당
        if n_tgt > 0:
            aligned[0] = tgt_text
    
    # 빈 문자열 처리
    for i in range(len(aligned)):
        if not aligned[i].strip():
            aligned[i] = ""
    
    return aligned
