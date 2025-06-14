# aligner.py - 원문/번역문 구 간 정렬을 위한 DP 기반 매칭 알고리즘

import numpy as np
from typing import List, Callable, Optional, Any
from embedder import compute_embeddings_with_cache
from tokenizer import split_src_meaning_units, split_tgt_meaning_units, split_tgt_by_src_units_semantic

# ── IO 모듈에서 초기화해서 쓰기 위한 전역 DP 배열 선언
dp_prev_global = np.array([])
dp_curr_global = np.array([])

def cosine_similarity(vec1: Any, vec2: Any) -> float:
    """코사인 유사도 계산 (벡터가 0인 경우 대비)"""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))

def align_src_tgt(src_units, tgt_units, embed_func=compute_embeddings_with_cache):
    if len(src_units) != len(tgt_units):
        flatten_tgt = " ".join(tgt_units)
        new_tgt_units = split_tgt_by_src_units_semantic(src_units, flatten_tgt, embed_func)
        assert len(new_tgt_units) == len(src_units)
        return list(zip(src_units, new_tgt_units))
    else:
        return list(zip(src_units, tgt_units))
    print(f"[DEBUG] align_src_tgt 호출됨 → src_units 개수={len(src_units)}, tgt_units 개수={len(tgt_units)}")


    # 1) 임베딩 계산 (배치 처리)
    src_embs = []
    tgt_embs = []

    # src 임베딩
    for i in range(0, src_len, batch_size):
        batch = src_units[i : i + batch_size]
        emb_batch = embed_func(batch)  # 실제 임베딩 수행 결과를 emb_batch에 할당
        # 디버그 출력: 배치 크기와 반환된 벡터 모양 확인
        print(f"[DEBUG] src 임베딩 → batch 크기={len(batch)}, 반환 벡터 모양={np.array(emb_batch).shape}")
        src_embs.extend(emb_batch)    # tgt 임베딩
    # tgt 임베딩
    for i in range(0, tgt_len, batch_size):
        batch = tgt_units[i : i + batch_size]
        emb_batch = embed_func(batch)  # 실제 임베딩 수행 결과를 emb_batch에 할당
        # 디버그 출력: 배치 크기와 반환된 벡터 모양 확인
        print(f"[DEBUG] tgt 임베딩 → batch 크기={len(batch)}, 반환 벡터 모양={np.array(emb_batch).shape}")
        tgt_embs.extend(emb_batch)
    # 2) DP 배열 초기화 (전역 배열 활용)
    # dp_prev_global, dp_curr_global는 크기가 tgt_len + 1 이어야 함
    if dp_prev_global is None or len(dp_prev_global) < tgt_len + 1:
        dp_prev_global = np.full(tgt_len + 1, -np.inf)
    if dp_curr_global is None or len(dp_curr_global) < tgt_len + 1:
        dp_curr_global = np.full(tgt_len + 1, -np.inf)

    # dp_prev_global[0] = 0으로 시작
    dp_prev_global.fill(-np.inf)
    dp_curr_global.fill(-np.inf)
    dp_prev_global[0] = 0.0

    # 3) DP 테이블 및 backtracking 테이블 초기화
    backtrack = [[None] * (tgt_len + 1) for _ in range(src_len + 1)]

    # 최대 병합 길이 제한
    max_merge_len_cap = min(max_merge_len_cap, tgt_len)

    # 4) DP 알고리즘 수행
    for i in range(1, src_len + 1):
        dp_curr_global.fill(-np.inf)
        for j in range(tgt_len + 1):
            # 이전 상태에서 dp 값 유지 (아무것도 안하는 경우)
            # (실제로 j=0인 경우 제외, 그냥 dp_curr_global[j] = -inf 유지)
            pass

        for j in range(tgt_len + 1):
            if dp_prev_global[j] == -np.inf:
                continue
            # 4-1) src 단위 i (1-based)
            src_idx = i - 1

            # 4-2) tgt 병합 후보 길이: 1 ~ max_merge_len_cap
            for merge_len in range(1, max_merge_len_cap + 1):
                tgt_start = j
                tgt_end = j + merge_len
                if tgt_end > tgt_len:
                    break

                # 병합된 tgt 임베딩 평균 계산
                merged_vec = np.mean(tgt_embs[tgt_start:tgt_end], axis=0)

                # 유사도 계산
                if similarity_metric == "cosine":
                    sim = cosine_similarity(src_embs[src_idx], merged_vec)
                else:
                    raise ValueError(f"지원하지 않는 similarity_metric: {similarity_metric}")

                # 유사도 임계값 체크 (있으면)
                if similarity_threshold is not None and sim < similarity_threshold:
                    continue

                new_score = dp_prev_global[j] + sim
                if new_score > dp_curr_global[tgt_end]:
                    dp_curr_global[tgt_end] = new_score
                    backtrack[i][tgt_end] = (j, merge_len)

        # dp_prev_global 갱신
        dp_prev_global, dp_curr_global = dp_curr_global, dp_prev_global

    # 5) 최적 경로 역추적
    aligned_tgt_units = []
    i = src_len
    j = tgt_len

    # BUG FIX: 역추적이 불가능한 경우(즉, backtrack[i][j]가 None이고 i>0)에는
    # j를 1씩 줄여가며 backtrack[i][j]가 존재하는 위치를 찾고, 그 사이의 tgt_units는 마지막에 합쳐서 반환
    while i > 0:
        if backtrack[i][j] is None:
            # 역추적 실패 시, j를 줄여가며 마지막 가능한 구간을 합쳐서 할당
            found = False
            for jj in range(j-1, -1, -1):
                if backtrack[i][jj] is not None:
                    # 남은 구간 합치기
                    merged_text = "".join(tgt_units[jj:j])
                    aligned_tgt_units.append(merged_text)
                    i -= 1
                    j = jj
                    found = True
                    break
            if not found:
                aligned_tgt_units.append("")
                i -= 1
                # j는 그대로 유지
        else:
            prev_j, merge_len = backtrack[i][j]
            # tgt 병합 구간
            merged_text = "".join(t + " " for t in tgt_units[prev_j : j]).strip()
            aligned_tgt_units.append(merged_text)
            i -= 1
            j = prev_j

    aligned_tgt_units.reverse()

    # 만약 src 길이보다 tgt 병합된 결과가 부족한 경우 빈 문자열로 채움
    while len(aligned_tgt_units) < src_len:
        aligned_tgt_units.append("")

    return aligned_tgt_units
