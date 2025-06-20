"""원문과 번역문을 의미 단위로 분할하는 모듈 - prototype02 + MeCab"""

import logging
import numpy as np
import regex
import re
from typing import List, Callable
from soynlp.tokenizer import LTokenizer

logger = logging.getLogger(__name__)

# 기본 설정값
DEFAULT_MIN_TOKENS = 1
DEFAULT_MAX_TOKENS = 50
DEFAULT_SIMILARITY_THRESHOLD = 0.4

# SoyNLP tokenizer 초기화
tokenizer = LTokenizer()

# 🔧 MeCab 초기화 (Kkma 대신)
mecab = None
try:
    import MeCab
    mecab = MeCab.Tagger()
    logger.info("✅ MeCab 초기화 성공")
except ImportError:
    logger.warning("⚠️ MeCab 없음, 기본 토크나이저 사용")
    mecab = None
except Exception as e:
    logger.warning(f"⚠️ MeCab 초기화 실패: {e}")
    mecab = None

# 미리 컴파일된 정규식
hanja_re = regex.compile(r'\p{Han}+')
hangul_re = regex.compile(r'^\p{Hangul}+$')
combined_re = regex.compile(
    r'(\p{Han}+)+(?:\p{Hangul}+)(?:은|는|이|가|을|를|에|에서|으로|로|와|과|도|만|며|고|하고|의|때)?'
)

def split_src_meaning_units(text: str) -> List[str]:
    """한문 텍스트를 '한자+조사+어미' 단위로 묶어서 분할"""
    text = text.replace('\n', ' ').replace('：', '： ')
    tokens = regex.findall(r'\S+', text)
    units: List[str] = []
    i = 0

    while i < len(tokens):
        tok = tokens[i]

        # 1) 한자+한글+조사 어미 복합패턴 우선 매칭
        m = combined_re.match(tok)
        if m:
            units.append(m.group(0))
            i += 1
            continue

        # 2) 순수 한자 토큰
        if hanja_re.search(tok):
            unit = tok
            j = i + 1
            # 뒤따르는 순수 한글 토큰이 있으면 묶기
            while j < len(tokens) and hangul_re.match(tokens[j]):
                unit += tokens[j]
                j += 1
            units.append(unit)
            i = j
            continue

        # 3) 순수 한글 토큰: SoyNLP LTokenizer 사용
        if hangul_re.match(tok):
            korean_tokens = tokenizer.tokenize(tok)
            units.extend(korean_tokens)
            i += 1
            continue

        # 4) 기타 토큰 (숫자, 로마자 등) 그대로 보존
        units.append(tok)
        i += 1

    return units

def split_inside_chunk(chunk: str) -> List[str]:
    """조사, 어미, 그리고 '：' 기준으로 의미 단위 분할"""
    delimiters = ['을', '를', '이', '가', '은', '는', '에', '에서', '로', '으로',
                  '와', '과', '고', '며', '하고', '때', '의', '도', '만', '때에', '：']
    
    pattern = '|'.join([f'(?<={re.escape(d)})' for d in delimiters])
    try:
        parts = re.split(pattern, chunk)
        return [p.strip() for p in parts if p.strip()]
    except:
        return [p for p in chunk.split() if p.strip()]

def find_target_span_end_simple(src_unit: str, remaining_tgt: str) -> int:
    """간단한 타겟 스팬 탐색"""
    hanja_chars = regex.findall(r'\p{Han}+', src_unit)
    if not hanja_chars:
        return 0
    last = hanja_chars[-1]
    idx = remaining_tgt.rfind(last)
    if idx == -1:
        return len(remaining_tgt)
    end = idx + len(last)
    next_space = remaining_tgt.find(' ', end)
    return next_space + 1 if next_space != -1 else len(remaining_tgt)

def find_target_span_end_semantic(
    src_unit: str,
    remaining_tgt: str,
    embed_func: Callable,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
) -> int:
    """최적화된 타겟 스팬 탐색 함수"""
    if not src_unit or not remaining_tgt:
        return 0
        
    try:
        # 1) 원문 임베딩 (단일 계산)
        src_emb = embed_func([src_unit])[0]
        
        # 2) 번역문 토큰 분리 및 누적 길이 계산
        tgt_tokens = remaining_tgt.split()
        if not tgt_tokens:
            return 0
            
        upper = min(len(tgt_tokens), max_tokens)
        cumulative_lengths = [0]
        current_length = 0
        
        for tok in tgt_tokens:
            current_length += len(tok) + 1
            cumulative_lengths.append(current_length)
            
        # 3) 후보 세그먼트 생성
        candidates = []
        candidate_indices = []
        
        step_size = 1 if upper <= 10 else 2
        
        for end_i in range(min_tokens-1, upper, step_size):
            cand = " ".join(tgt_tokens[:end_i+1])
            candidates.append(cand)
            candidate_indices.append(end_i)
            
        # 4) 배치 임베딩
        cand_embs = embed_func(candidates)
        
        # 5) 최적 매칭 탐색
        best_score = -1.0
        best_end_idx = cumulative_lengths[-1]
        
        for i, emb in enumerate(cand_embs):
            score = np.dot(src_emb, emb) / (np.linalg.norm(src_emb) * np.linalg.norm(emb) + 1e-8)
            
            end_i = candidate_indices[i]
            length_ratio = (end_i + 1) / len(tgt_tokens)
            length_penalty = min(1.0, length_ratio * 2)
            
            adjusted_score = score * length_penalty
            
            if adjusted_score > best_score and score >= similarity_threshold:
                best_score = adjusted_score
                best_end_idx = cumulative_lengths[end_i + 1]
                
        return best_end_idx
        
    except Exception as e:
        logger.warning(f"의미 매칭 오류, 단순 매칭으로 대체: {e}")
        return find_target_span_end_simple(src_unit, remaining_tgt)

def split_tgt_by_src_units(src_units: List[str], tgt_text: str) -> List[str]:
    """원문 단위에 따른 번역문 분할 (단순 방식)"""
    results = []
    cursor = 0
    total = len(tgt_text)
    for src_u in src_units:
        remaining = tgt_text[cursor:]
        end_len = find_target_span_end_simple(src_u, remaining)
        chunk = tgt_text[cursor:cursor+end_len]
        results.extend(split_inside_chunk(chunk))
        cursor += end_len
    if cursor < total:
        results.extend(split_inside_chunk(tgt_text[cursor:]))
    return results

def split_tgt_by_src_units_semantic(
    src_units: List[str], 
    tgt_text: str, 
    embed_func: Callable, 
    min_tokens: int = DEFAULT_MIN_TOKENS
) -> List[str]:
    """원문 단위에 따른 번역문 분할 (의미 기반)"""
    tgt_tokens = tgt_text.split()
    N, T = len(src_units), len(tgt_tokens)
    if N == 0 or T == 0:
        return []

    dp = np.full((N+1, T+1), -np.inf)
    back = np.zeros((N+1, T+1), dtype=int)
    dp[0, 0] = 0.0

    # 원문 임베딩 계산
    src_embs = embed_func(src_units)

    # DP 테이블 채우기
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

def split_tgt_meaning_units(
    src_text: str,
    tgt_text: str,
    use_semantic: bool = True,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    embed_func: Callable = None
) -> List[str]:
    """번역문을 의미 단위로 분할"""
    # 지연 임포트로 순환 참조 방지
    if embed_func is None:
        from embedder import compute_embeddings_with_cache
        embed_func = compute_embeddings_with_cache
        
    src_units = split_src_meaning_units(src_text)

    if use_semantic:
        return split_tgt_by_src_units_semantic(
            src_units,
            tgt_text,
            embed_func=embed_func,
            min_tokens=min_tokens
        )
    else:
        return split_tgt_by_src_units(src_units, tgt_text)

# 🔧 MeCab 버전으로 변경
def tokenize_text(text):
    """형태소 분석 및 토큰화 - MeCab 사용"""
    if mecab:
        result = mecab.parse(text)
        tokens = []
        for line in result.split('\n'):
            if line and line != 'EOS':
                parts = line.split('\t')
                if len(parts) >= 1:
                    tokens.append(parts[0])
        return tokens
    else:
        # MeCab 없으면 기본 분할
        return text.split()

def pos_tag_text(text):
    """품사 태깅 - MeCab 사용"""
    if mecab:
        result = mecab.parse(text)
        pos_tags = []
        for line in result.split('\n'):
            if line and line != 'EOS':
                parts = line.split('\t')
                if len(parts) >= 2:
                    surface = parts[0]
                    pos = parts[1].split(',')[0]
                    pos_tags.append((surface, pos))
        return pos_tags
    else:
        # MeCab 없으면 기본 처리
        return [(word, 'UNKNOWN') for word in text.split()]

def sentence_split(text):
    """문장 단위로 분리 - 기본 구두점 기반"""
    sentences = re.split(r'[.!?。！？]+', text)
    return [s.strip() for s in sentences if s.strip()]

if __name__ == "__main__":
    # 테스트
    logging.basicConfig(level=logging.INFO)
    
    print("🔧 prototype02 + MeCab 버전")
    print(f"MeCab 상태: {'사용 가능' if mecab else '사용 불가'}")
    print("=" * 50)
    
    test_cases = [
        ("興也라", "興이다."),
        ("蒹은 薕(렴)이요 葭는 蘆也라", "蒹은 물억새이고 葭는 갈대이다."),
        ("白露凝戾爲霜然後에 歲事成이요", "白露가 얼어 서리가 된 뒤에야 歲事가 이루어지고")
    ]
    
    for i, (src, tgt) in enumerate(test_cases, 1):
        print(f"\n테스트 {i}:")
        print(f"원문: {src}")
        print(f"번역: {tgt}")
        
        src_units = split_src_meaning_units(src)
        tgt_units = split_tgt_meaning_units(src, tgt, use_semantic=False)
        
        print(f"원문 분할: {src_units}")
        print(f"번역 분할: {tgt_units}")