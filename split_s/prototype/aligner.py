# aligner.py - 원문/번역문 구 간 정렬을 위한 DP 기반 매칭 알고리즘

import logging
from typing import List, Tuple, Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

# 전역 DP 배열 (io_manager.py에서 초기화해 줘야 함)
dp_prev_global: Optional[np.ndarray] = None
dp_curr_global: Optional[np.ndarray] = None


def generate_tgt_segments(
    tgt_tokens: List[str],
    max_merge_len: int
) -> Tuple[List[str], List[Tuple[int, int]], Dict[Tuple[int, int], int]]:
    """
    tgt_tokens에서 (start, end) 가능한 모든 구간을 생성:
    - tgt_segments: ["단어1 단어2", ...]
    - segment_indices: [(start_idx, end_idx), ...]
    - segment_dict: { (start, end): 인덱스 }
    """
    m = len(tgt_tokens)
    tgt_segments = []
    segment_indices = []
    segment_dict: Dict[Tuple[int,int], int] = {}
    idx_counter = 0

    for start in range(m):
        for end in range(start + 1, min(m + 1, start + 1 + max_merge_len)):
            seg = " ".join(tgt_tokens[start:end])
            tgt_segments.append(seg)
            segment_indices.append((start, end))
            segment_dict[(start, end)] = idx_counter
            idx_counter += 1

    return tgt_segments, segment_indices, segment_dict


def compute_dp_alignment(
    src_norm: np.ndarray,
    tgt_norm: np.ndarray,
    segment_indices: List[Tuple[int, int]],
    segment_dict: Dict[Tuple[int, int], int],
    tgt_tokens: List[str],
    n: int,
    m: int,
) -> List[str]:
    """
    DP로 src_norm(크기 n×d)과 tgt_norm(크기 (#segments)×d)의 코사인 유사도 기반 최적 정렬 결과를 반환.
    - dp_prev_global, dp_curr_global: 크기 m+1 배열 (io_manager.py에서 미리 초기화 필요)
    """
    global dp_prev_global, dp_curr_global

    # dp_prev_global, dp_curr_global는 반드시 (m+1) 길이로 초기화되어 있어야 함
    if dp_prev_global is None or dp_curr_global is None:
        raise ValueError("[Aligner] DP 전역 배열이 초기화되지 않았습니다.")

    dp_prev = dp_prev_global
    dp_curr = dp_curr_global

    # 1) 초기화
    dp_prev[: m + 1] = -np.inf
    dp_prev[0] = 0.0

    # parent[i][j] = 이전 (i-1, k) 좌표. None이면 매핑 불가.
    parent: List[List[Optional[Tuple[int, int]]]] = [
        [None] * (m + 1) for _ in range(n + 1)
    ]

    # 2) DP 순회
    for i in range(1, n + 1):
        dp_curr[: m + 1] = -np.inf
        for j in range(i, m + 1):
            # src index i-1와 tgt 구간 (k, j) 매핑 시도
            for k in range(i - 1, j):
                seg_idx = segment_dict.get((k, j))
                if seg_idx is None:
                    continue
                sim = float(np.dot(src_norm[i - 1], tgt_norm[seg_idx]))
                score = dp_prev[k] + sim
                if score > dp_curr[j]:
                    dp_curr[j] = score
                    parent[i][j] = (i - 1, k)
        # 다음 i 단계로 넘어가기 전에 복사
        dp_prev[: m + 1] = dp_curr[: m + 1]

    # 3) 역추적 (Backtracking)
    aligned_segments: List[str] = []
    i, j = n, m
    while i > 0 and j > 0:
        if parent[i][j] is None:
            logger.warning(f"[Aligner] 역추적 실패: i={i}, j={j}")
            # 남은 src 단위는 모두 빈 문자열 매핑
            for _ in range(i, 0, -1):
                aligned_segments.append("")
            break
        pi, pj = parent[i][j]
        seg_str = " ".join(tgt_tokens[pj:j])  # (pj, j) 구간 문자열
        aligned_segments.append(seg_str)
        i, j = pi, pj

    aligned_segments.reverse()

    # 길이 보정: aligned_segments가 n보다 짧거나 길면 맞춰준다
    if len(aligned_segments) < n:
        aligned_segments = [""] * (n - len(aligned_segments)) + aligned_segments
    elif len(aligned_segments) > n:
        aligned_segments = aligned_segments[-n:]

    return aligned_segments


def align_src_tgt(
    src_units: List[str],
    tgt_tokens: List[str],
    embedder,               # 외부에서 embedder.compute_embeddings_with_cache 전달
    batch_size: int = 128,
    show_batch_progress: bool = False
) -> List[str]:
    """
    src_units(의미 단위) 개수에 맞추어 tgt_tokens를 병합·정렬하여 aligned segments 반환.
    - embedder: embedder.compute_embeddings_with_cache 함수
    """
    n = len(src_units)
    m = len(tgt_tokens)
    if n == 0:
        return []

    # 1) src 임베딩 & 정규화
    src_emb = embedder(src_units, batch_size=batch_size, show_batch_progress=show_batch_progress)
    src_norm = src_emb / (np.linalg.norm(src_emb, axis=1, keepdims=True) + 1e-8)

    # 2) max_merge_len 계산 (예: (m//n)*2, 단 상한 max 15)
    if n > 0:
        tentative = (m // n) * 2
        max_merge_len = max(1, min(tentative, 15))
    else:
        max_merge_len = 1

    tgt_segments, segment_indices, segment_dict = generate_tgt_segments(tgt_tokens, max_merge_len)

    # 3) tgt 임베딩 & 정규화
    tgt_emb = embedder(tgt_segments, batch_size=batch_size, show_batch_progress=show_batch_progress)
    tgt_norm = tgt_emb / (np.linalg.norm(tgt_emb, axis=1, keepdims=True) + 1e-8)

    # 4) DP 정렬
    aligned_segments = compute_dp_alignment(
        src_norm, tgt_norm,
        segment_indices, segment_dict,
        tgt_tokens, n, m
    )
    return aligned_segments