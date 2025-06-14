# tokenizer.py - 원문/번역문을 구 단위로 분할하는 토크나이저 모듈
# ※ Onigmo 엔진 호환 정규식 사용 (ruby, 고급 에디터 등)

import regex as re
from typing import List, Tuple, Optional
import spacy

# ──────────────────────────────────────────────────────────────────────────────
# 0) SpaCy 한국어 모델 로드 (기존과 동일)
# ──────────────────────────────────────────────────────────────────────────────
try:
    nlp = spacy.load("ko_core_news_lg")
except OSError:
    raise ImportError(
        "SpaCy 모델 'ko_core_news_lg'이 설치되어 있지 않습니다.\n"
        "설치: python -m spacy download ko_core_news_lg"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 1) split_src_meaning_units: 원문(한문) 의미 단위 분할 (기존 로직 그대로)
# ──────────────────────────────────────────────────────────────────────────────
def split_src_meaning_units(text: str) -> List[str]:
    """
    한문 텍스트를 '한자+조사' 단위로 묶어서 분할
    - 줄바꿈 제거, 콜론 뒤 공백 처리
    - regex.find: \p{Han} (한자)와 \p{Hangul} (한글) 패턴 사용
    """
    text = text.replace('\n', ' ').replace('：', '： ')
    tokens = re.findall(r"\S+", text)
    hanja_pattern = r'\p{Han}+'
    units: List[str] = []
    i = 0

    while i < len(tokens):
        token = tokens[i]
        if re.search(hanja_pattern, token):
            unit = token
            j = i + 1
            # 뒤에 순수 한글 토큰(모두 한글) 있으면 묶기
            while j < len(tokens) and re.match(r'^\p{Hangul}+$', tokens[j]):
                unit += tokens[j]
                j += 1
            units.append(unit)
            i = j
        else:
            units.append(token)
            i += 1

    return units


# ──────────────────────────────────────────────────────────────────────────────
# 2) split_inside_chunk: 작은 덩어리 내부에서 ’을/를/때/고/며/하고’ 기준으로 세분화 (기존 로직 그대로)
# ──────────────────────────────────────────────────────────────────────────────
def split_inside_chunk(chunk: str) -> List[str]:
    """
    조사나 어미 기준으로 의미 단위를 나누되, 
    원래 텍스트에서 붙어 있던 조사는 절대로 앞 어절과 분리하지 않는다.
    즉, 공백 삽입 없이 원형 보존.
    """
    doc = nlp(chunk)
    tokens = list(doc)

    units: List[str] = []
    current = ""
    for i, tok in enumerate(tokens):
        is_suffix = (tok.pos_ in {"ADP"} or tok.tag_ in {"JKS", "JKO", "JKG", "JKB", "JKC"})
        is_tt = (tok.text == "때" and tok.pos_ in {"NOUN", "SCONJ"})

        # token + 원래 공백 포함
        token_str = tok.text + tok.whitespace_

        if is_suffix:
            current += tok.text + tok.whitespace_
            units.append(current)
            current = ""
        else:
            current += token_str

    if current.strip():
        units.append(current.strip())

    return units


# ──────────────────────────────────────────────────────────────────────────────
# 3) (원본) find_target_span_end_simple: 원문 단위가 번역문에서 끝나는 위치 찾기 (기존 히윗)
# ──────────────────────────────────────────────────────────────────────────────
def find_target_span_end_simple(src_unit: str, remaining_tgt: str) -> int:
    """
    Very simple heuristic:
    - src_unit에서 한자만 추출하여, 마지막 한자(last_hanja)를 구함
    - remaining_tgt 내에서 마지막 한자 위치(idx) 탐색
    - idx+len(last_hanja) 이후 첫 공백까지 포함하여 끝 인덱스(end_idx) 반환
    - 찾지 못하면 remaining_tgt 전체 길이 반환
    """
    hanja_chars = re.findall(r'\p{Han}+', src_unit)
    if not hanja_chars:
        return 0
    last_hanja = hanja_chars[-1]
    idx = remaining_tgt.rfind(last_hanja)
    if idx == -1:
        return len(remaining_tgt)
    end_idx = idx + len(last_hanja)
    next_space = remaining_tgt.find(' ', end_idx)
    if next_space != -1:
        return next_space + 1
    return len(remaining_tgt)


# ──────────────────────────────────────────────────────────────────────────────
# 3-b) (고도화) find_target_span_end_semantic: 임베딩 기반 의미 유사도 매칭
# ──────────────────────────────────────────────────────────────────────────────
#   - src_unit ↔ remaining_tgt(일부 구문) 간 코사인 유사도 최대가 되는 구간을 찾는다.
#   - 미리 “remaining_tgt”를 작게 자른 토큰 단위(스페이스 기준)로 리스트화한 뒤,
#     그 중 연속된 몇 개를 묶어 볼 때 가장 유사도가 높은 경계 위치를 선택한다.
#   - 너무 짧거나 너무 긴 구간은 배제하기 위해 min_tokens, max_tokens 인자로 제한을 둘 수 있다.
#
#   **전제**: 여기서는 이미 파이프라인 상위에서 임베딩 모델을 로드해 두었거나,
#   또는 직접 이 함수 내에서 SentenceTransformer 등을 import해서 사용해도 무방하다.
#
#   아래 예시는, pipeline/embedder.py의 compute_embeddings_with_cache 함수를 그대로 가져와서 사용한다고 가정합니다.
from embedder import compute_embeddings_with_cache
import numpy as np

def find_target_span_end_semantic(
    src_unit: str,
    remaining_tgt: str,
    embed_func=compute_embeddings_with_cache,
    min_tokens: int = 1,
    max_tokens: int = 50,
    similarity_metric: str = "cosine"
) -> int:
    """
    임베딩 기반 의미 유사도를 활용해, remaining_tgt 내에서
    src_unit과 가장 유사한 “토큰 묶음”의 끝 인덱스를 찾아 반환.

    Args:
        src_unit (str): 원문 구 단위 텍스트 (마스킹된 상태일 수도 있음)
        remaining_tgt (str): 번역문 중 아직 처리되지 않은 뒷부분 전체
        embed_func: 임베딩 함수 (문자열 리스트 -> 임베딩 ndarray)
        min_tokens: 최소 토큰 수 (공백 분리)로 묶을 크기
        max_tokens: 최대 토큰 수로 묶을 크기 (너무 크면 연산량이 늘어나니 제한)
        similarity_metric: 'cosine' (현재만 지원)

    Returns:
        int: remaining_tgt[0:end_idx] 까지가 해당 src_unit에 대응될 가능성이 가장 높은 구간 길이.
    """
    print("[DEBUG] find_target_span_end_semantic 호출됨 - src_unit:", src_unit)

    # 1) src_unit 임베딩 계산
    src_emb = embed_func([src_unit])[0]  # (1, D) → (D,)

    # 2) remaining_tgt를 “토큰 단위”로 분리 (공백 기준)
    tgt_tokens = remaining_tgt.split()
    if not tgt_tokens:
        return 0

    # 3) 가능한 토큰 묶음 조합을 만들어 가면서 유사도 계산
    #    cursor=0에 고정된 상태에서, end_token_index ∈ [min_tokens-1, min(max_tokens, len(tgt_tokens))-1] 사이
    #    각 구간(remaining_tgt_tokens[0 : end_token_index+1]) 을 문자열로 합쳐서 임베딩 → 유사도 비교
    best_score = -1.0
    best_end_char_idx = len(remaining_tgt)  # fallback: 전체 길이
    cumulative_lengths = [0] * (len(tgt_tokens) + 1)
    # cumulative_lengths[i] = remaining_tgt 상에서 i번째 토큰이 시작하는 문자 인덱스
    idx = 0
    for i, tok in enumerate(tgt_tokens):
        cumulative_lengths[i] = idx
        idx += len(tok) + 1  # 토큰 + 공백 1개
    cumulative_lengths[len(tgt_tokens)] = len(remaining_tgt)

    # 3-a) 후보 segment들을 한꺼번에 모아서 배치 임베딩 → 유사도 계산
    upper = min(len(tgt_tokens), max_tokens)
    # 누적 문자 위치 계산
    cumulative_lengths = [0] * (len(tgt_tokens) + 1)
    idx = 0
    for i, tok in enumerate(tgt_tokens):
        cumulative_lengths[i] = idx
        idx += len(tok) + 1
    cumulative_lengths[len(tgt_tokens)] = len(remaining_tgt)

    # 후보 문자열 리스트 생성
    candidates = [" ".join(tgt_tokens[: end_i + 1]) for end_i in range(min_tokens - 1, upper)]
    # 해당 문자열들 배치 임베딩
    try:
        cand_embs = embed_func(candidates)
    except Exception:
        return find_target_span_end_simple(src_unit, remaining_tgt)

    # src_unit 임베딩
    try:
        src_emb = embed_func([src_unit])[0]
    except Exception:
        return find_target_span_end_simple(src_unit, remaining_tgt)

    # 4) cosine 유사도 계산
        def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

        for i, emb in enumerate(cand_embs):
            score = cosine_sim(src_emb, emb)
            if score > best_score:
                best_score = score
                best_end_char_idx = cumulative_lengths[min_tokens - 1 + i + 1]  # +1 for inclusive

    except Exception as e:
        print(f"[ERROR] 임베딩 계산 중 오류 발생: {e}")
        return len(remaining_tgt)  # fallback

    return best_end_char_idx

# ──────────────────────────────────────────────────────────────────────────────
# 4) (원본) split_tgt_by_src_units: “원문 단위별로 번역문 구간을 떼어오고 내부 세분화” (기존 히윗)
# ──────────────────────────────────────────────────────────────────────────────
def split_tgt_by_src_units(src_units: List[str], tgt_text: str) -> List[str]:
    """
    src_units: ['作詁訓傳時에', '移其篇第하고', '因改之耳라', ...]
    tgt_text: 전체 번역문 한 문장 (공백·구두점 포함)

    1) 각 src_unit이 번역된 끝 위치(end_len)를 find_target_span_end_simple 로 구함
    2) cursor ~ cursor+end_len 구간을 tgt_chunk로 떼어냄
    3) split_inside_chunk(tgt_chunk) 통해 내부 세분화
    4) 결과를 순서대로 쌓아 리턴
    5) 마지막 src_unit 뒤 남은 텍스트가 있으면 추가 세분화하여 리턴
    """
    results: List[str] = []
    cursor = 0
    total_len = len(tgt_text)

    for src_u in src_units:
        remaining = tgt_text[cursor:]
        end_len = find_target_span_end_simple(src_u, remaining)
        tgt_chunk = tgt_text[cursor: cursor + end_len]
        sub_chunks = split_inside_chunk(tgt_chunk)
        results.extend(sub_chunks)
        cursor += end_len

    if cursor < total_len:
        trailing = tgt_text[cursor:]
        trailing_sub = split_inside_chunk(trailing)
        results.extend(trailing_sub)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# 4-b) (고도화) split_tgt_by_src_units_semantic: 의미 기반 매칭으로 대체
# ──────────────────────────────────────────────────────────────────────────────
def split_tgt_by_src_units_semantic(src_units, tgt_text, embed_func=compute_embeddings_with_cache, min_tokens=1):
    tgt_tokens = tgt_text.split()
    N, T = len(src_units), len(tgt_tokens)
    if N == 0 or T == 0:
        return []

    import numpy as np
    src_embs = embed_func(src_units)
    dp = np.full((N+1, T+1), -np.inf)
    back = np.zeros((N+1, T+1), dtype=int)
    dp[0, 0] = 0.0

    # DP 테이블 채우기 (j 루프 범위 주의!)
    for i in range(1, N+1):
        for j in range(i*min_tokens, T-(N-i)*min_tokens+1):
            for k in range((i-1)*min_tokens, j-min_tokens+1):
                span = " ".join(tgt_tokens[k:j])
                tgt_emb = embed_func([span])[0]
                sim = float(np.dot(src_embs[i-1], tgt_emb)/((np.linalg.norm(src_embs[i-1])*np.linalg.norm(tgt_emb))+1e-8))
                score = dp[i-1, k] + sim
                if score > dp[i, j]:
                    dp[i, j] = score
                    back[i, j] = k

    # Traceback
    cuts = [T]
    curr = T
    for i in range(N, 0, -1):
        prev = int(back[i, curr])
        cuts.append(prev)
        curr = prev
    cuts = cuts[::-1]
    assert cuts[0] == 0 and cuts[-1] == T and len(cuts) == N + 1

    # Build actual spans
    tgt_spans = []
    for i in range(N):
        span = " ".join(tgt_tokens[cuts[i]:cuts[i+1]]).strip()
        tgt_spans.append(span)
    return tgt_spans

# ──────────────────────────────────────────────────────────────────────────────
# 5) split_tgt_meaning_units: 편의 래퍼 함수 (최종 버전)
# ──────────────────────────────────────────────────────────────────────────────
def split_tgt_meaning_units(
    src_text: str,
    tgt_text: str,
    use_semantic: bool = True,
    min_tokens: int = 1,
    max_tokens: int = 50
) -> List[str]:
    """
    원문(src_text)과 번역문(tgt_text)을 받아,
    1) split_src_meaning_units(src_text) → src_units 생성
    2) split_tgt_by_src_units_semantic 또는 split_tgt_by_src_units 로 분할
    """
    # 1) 원문 구 단위 분할
    src_units = split_src_meaning_units(src_text)

    # 2) semantic-based 분할(기본) 또는 기존 히윗 분할
    if use_semantic:
        return split_tgt_by_src_units_semantic(
            src_units,
            tgt_text,
            embed_func=compute_embeddings_with_cache,
            min_tokens=min_tokens,
        )
    else:
        return split_tgt_by_src_units(src_units, tgt_text)
