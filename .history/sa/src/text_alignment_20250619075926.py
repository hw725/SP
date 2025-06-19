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
    cross_enc=None
) -> List[str]:
    
    tgt_eojeols = tgt_text.split()
    n_src, n_tgt = len(src_units), len(tgt_eojeols)
    
    # 1) 배치 임베딩: 모든 src 단위를 한 번에 처리
    src_embeds = embed_func(src_units)  # shape: (n_src, embed_dim)
    
    # 2) 가능한 모든 tgt 스팬을 미리 생성하고 배치 임베딩
    tgt_spans = []
    span_to_idx = {}
    for j in range(n_tgt):
        for k in range(j+1, min(j+5, n_tgt+1)):  # 최대 길이 제한
            span = " ".join(tgt_eojeols[j:k])
            if span not in span_to_idx:
                span_to_idx[span] = len(tgt_spans)
                tgt_spans.append(span)
    
    if tgt_spans:
        tgt_embeds = embed_func(tgt_spans)  # 배치 처리
    else:
        tgt_embeds = np.array([])

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
                        score = calculate_matching_score(
                            src_units[i],
                            tgt_span,
                            src_embeds[i],
                            embed_func,
                            src_analyses[i],
                            source_analyzer,
                            target_analyzer,
                            weight_semantic,
                            weight_structure,
                            weight_pos,
                            weight_cross,
                            cross_tokenizer,
                            cross_model
                        )
                        if dp[i+1][k] < score:
                            dp[i+1][k] = score
                            parent[i+1][k] = (i, j, k)
    
    # 최적 경로 역추적
    aligned_segments = []
    i, j, k = n_src, 0, 0
    while i > 0 and j < n_tgt:
        if parent[i][j] is None:
            break
        pi, pj, pk = parent[i][j]
        
        # 실제 매칭인 경우에만 추가
        if pk > pj:
            aligned_segments.append((src_units[pi], " ".join(tgt_eojeols[pj:pk])))
        
        i, j, k = pi, pj, pk
    
    aligned_segments.reverse()
    return [tgt for src, tgt in aligned_segments]

# -------------------------------------------------
# split_tgt_meaning_units 래퍼 함수
# -------------------------------------------------
def split_tgt_meaning_units(
    masked_tgt: str,
    src_units: list,
    embed_func,
    source_analyzer=None,
    target_analyzer=None
) -> list:
    """
    pipeline.py 에서 호출하는 함수.
    내부적으로 split_tgt_by_src_units_with_eojeol_merge(src_units, tgt_text, …)를 호출합니다.
    """
    return split_tgt_by_src_units_with_eojeol_merge(
        src_units,      # ★ 순서를 바꿔 src_units 먼저
        masked_tgt,     # ★ 그다음에 masked_tgt
        embed_func,
        source_analyzer,
        target_analyzer
        # 이후 인자는 모두 기본값 사용
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
    weight_semantic: float = 0.6,
    weight_structure: float = 0.25,
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
        inputs = cross_tok(
            src_unit, tgt_span,
            return_tensors="pt", truncation=True, max_length=128
        ).to(device)
        with torch.no_grad():
            logits = cross_enc(**inputs).logits  # shape (batch, labels) or (batch,1)
        # 1) classification head일 때 (labels>1): softmax[1]
        if logits.dim() == 2 and logits.size(1) > 1:
            probs = torch.softmax(logits, dim=1)
            cross_score = probs[0, 1].item()
        # 2) regression/single-logit head일 때: sigmoid(logit)
        else:
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
    cross_enc=None
) -> List[str]:
    
    tgt_eojeols = tgt_text.split()
    n_src, n_tgt = len(src_units), len(tgt_eojeols)
    
    # 1) 배치 임베딩: 모든 src 단위를 한 번에 처리
    src_embeds = embed_func(src_units)  # shape: (n_src, embed_dim)
    
    # 2) 가능한 모든 tgt 스팬을 미리 생성하고 배치 임베딩
    tgt_spans = []
    span_to_idx = {}
    for j in range(n_tgt):
        for k in range(j+1, min(j+5, n_tgt+1)):  # 최대 길이 제한
            span = " ".join(tgt_eojeols[j:k])
            if span not in span_to_idx:
                span_to_idx[span] = len(tgt_spans)
                tgt_spans.append(span)
    
    if tgt_spans:
        tgt_embeds = embed_func(tgt_spans)  # 배치 처리
    else:
        tgt_embeds = np.array([])

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
                        score = calculate_matching_score(
                            src_units[i],
                            tgt_span,
                            src_embeds[i],
                            embed_func,
                            src_analyses[i],
                            source_analyzer,
                            target_analyzer,
                            weight_semantic,
                            weight_structure,
                            weight_pos,
                            weight_cross,
                            cross_tokenizer,
                            cross_model
                        )
                        if dp[i+1][k] < score:
                            dp[i+1][k] = score
                            parent[i+1][k] = (i, j, k)
    
    # 최적 경로 역추적
    aligned_segments = []
    i, j, k = n_src, 0, 0
    while i > 0 and j < n_tgt:
        if parent[i][j] is None:
            break
        pi, pj, pk = parent[i][j]
        
        # 실제 매칭인 경우에만 추가
        if pk > pj:
            aligned_segments.append((src_units[pi], " ".join(tgt_eojeols[pj:pk])))
        
        i, j, k = pi, pj, pk
    
    aligned_segments.reverse()
    return [tgt for src, tgt in aligned_segments]

# -------------------------------------------------
# split_tgt_meaning_units 래퍼 함수
# -------------------------------------------------
def split_tgt_meaning_units(
    masked_tgt: str,
    src_units: list,
    embed_func,
    source_analyzer=None,
    target_analyzer=None
) -> list:
    """
    pipeline.py 에서 호출하는 함수.
    내부적으로 split_tgt_by_src_units_with_eojeol_merge(src_units, tgt_text, …)를 호출합니다.
    """
    return split_tgt_by_src_units_with_eojeol_merge(
        src_units,      # ★ 순서를 바꿔 src_units 먼저
        masked_tgt,     # ★ 그다음에 masked_tgt
        embed_func,
        source_analyzer,
        target_analyzer
        # 이후 인자는 모두 기본값 사용
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
    weight_semantic: float = 0.6,
    weight_structure: float = 0.25,
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
        inputs = cross_tok(
            src_unit, tgt_span,
            return_tensors="pt", truncation=True, max_length=128
        ).to(device)
        with torch.no_grad():
            logits = cross_enc(**inputs).logits  # shape (batch, labels) or (batch,1)
        # 1) classification head일 때 (labels>1): softmax[1]
        if logits.dim() == 2 and logits.size(1) > 1:
            probs = torch.softmax(logits, dim=1)
            cross_score = probs[0, 1].item()
        # 2) regression/single-logit head일 때: sigmoid(logit)
        else:
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
    cross_enc=None
) -> List[str]:
    
    tgt_eojeols = tgt_text.split()
    n_src, n_tgt = len(src_units), len(tgt_eojeols)
    
    # 1) 배치 임베딩: 모든 src 단위를 한 번에 처리
    src_embeds = embed_func(src_units)  # shape: (n_src, embed_dim)
    
    # 2) 가능한 모든 tgt 스팬을 미리 생성하고 배치 임베딩
    tgt_spans = []
    span_to_idx = {}
    for j in range(n_tgt):
        for k in range(j+1, min(j+5, n_tgt+1)):  # 최대 길이 제한
            span = " ".join(tgt_eojeols[j:k])
            if span not in span_to_idx:
                span_to_idx[span] = len(tgt_spans)
                tgt_spans.append(span)
    
    if tgt_spans:
        tgt_embeds = embed_func(tgt_spans)  # 배치 처리
    else:
        tgt_embeds = np.array([])

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
                        score = calculate_matching_score(
                            src_units[i],
                            tgt_span,
                            src_embeds[i],
                            embed_func,
                            src_analyses[i],
                            source_analyzer,
                            target_analyzer,
                            weight_semantic,
                            weight_structure,
                            weight_pos,
                            weight_cross,
                            cross_tokenizer,
                            cross_model
                        )
                        if dp[i+1][k] < score:
                            dp[i+1][k] = score
                            parent[i+1][k] = (i, j, k)
    
    # 최적 경로 역추적
    aligned_segments = []
    i, j, k = n_src, 0, 0
    while i > 0 and j < n_tgt:
        if parent[i][j] is None:
            break
        pi, pj, pk = parent[i][j]
        
        # 실제 매칭인 경우에만 추가
        if pk > pj:
            aligned_segments.append((src_units[pi], " ".join(tgt_eojeols[pj:pk])))
        
        i, j, k = pi, pj, pk
    
    aligned_segments.reverse()
    return [tgt for src, tgt in aligned_segments]

# -------------------------------------------------
# split_tgt_meaning_units 래퍼 함수
# -------------------------------------------------
def split_tgt_meaning_units(
    masked_tgt: str,
    src_units: list,
    embed_func,
    source_analyzer=None,
    target_analyzer=None
) -> list:
    """
    pipeline.py 에서 호출하는 함수.
    내부적으로 split_tgt_by_src_units_with_eojeol_merge(src_units, tgt_text, …)를 호출합니다.
    """
    return split_tgt_by_src_units_with_eojeol_merge(
        src_units,      # ★ 순서를 바꿔 src_units 먼저
        masked_tgt,     # ★ 그다음에 masked_tgt
        embed_func,
        source_analyzer,
        target_analyzer
        # 이후 인자는 모두 기본값 사용
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

# 3) DP with beam search (상위 K개 경로만 유지)
    beam_size = min(5, n_src)
    dp = np.full((n_src+1, n_tgt+1), -np.inf)
    parent = {}
    dp[0][0] = 0.0
    
    for i in range(n_src):
        # 현재 스텝에서 유효한 상태들만 처리 (빔 서치)
        valid_states = [(j, dp[i][j]) for j in range(n_tgt+1) if dp[i][j] > -np.inf]
        valid_states.sort(key=lambda x: x[1], reverse=True)
        valid_states = valid_states[:beam_size]
        
        for j, _ in valid_states:
            # 빈 매칭
            if dp[i][j] - 0.5 > dp[i+1][j]:
                dp[i+1][j] = dp[i][j] - 0.5
                parent[i+1, j] = (i, j, j)
            
            # 실제 매칭 (길이 제한)
            for k in range(j+1, min(j+5, n_tgt+1)):
                span = " ".join(tgt_eojeols[j:k])
                if span in span_to_idx:
                    tgt_idx = span_to_idx[span]
                    score = fast_calculate_score(
                        src_embeds[i], tgt_embeds[tgt_idx],
                        src_units[i], span, i, j, k
                    )
                    val = dp[i][j] + score - 0.1 * max(0, k-j-2)
                    if val > dp[i+1][k]:
                        dp[i+1][k] = val
                        parent[i+1, k] = (i, j, k)
    
    # 4) 최적 종료점 찾기
    best_j = max(range(n_tgt+1), key=lambda j: dp[n_src][j])
    
    # 5) 경로 복원 (역추적)
    path = []
    i, j = n_src, best_j
    while i > 0:
        if (i, j) in parent:
            prev_i, prev_j, end_j = parent[i, j]
            if prev_j < end_j:  # 실제 매칭
                path.append((prev_i, prev_j, end_j))
            i, j = prev_i, prev_j
        else:
            break
    
    path.reverse()
    
    # 6) 결과 조합 with 누락 방지
    result = [""] * n_src
    for src_idx, start_j, end_j in path:
        result[src_idx] = " ".join(tgt_eojeols[start_j:end_j])
    
    # prefix/suffix 처리
    if path:
        first_start = path[0][1]
        if first_start > 0:
            prefix = " ".join(tgt_eojeols[0:first_start])
            result[path[0][0]] = f"{prefix} {result[path[0][0]]}".strip()
        
        last_end = path[-1][2]
        if last_end < n_tgt:
            suffix = " ".join(tgt_eojeols[last_end:n_tgt])
            result[path[-1][0]] = f"{result[path[-1][0]]} {suffix}".strip()
    
    return [r if r.strip() else tgt_text for r in result]
