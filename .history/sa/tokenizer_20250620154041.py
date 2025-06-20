"""원문과 번역문을 의미 단위로 분할하는 모듈"""

import logging
import numpy as np
import regex
import re
from typing import List, Callable, Tuple
from soynlp.tokenizer import LTokenizer
import jieba  # 상단에 추가
import MeCab  # mecab-ko 패키지 사용

logger = logging.getLogger(__name__)

# MeCab 인스턴스 초기화 (전역)
try:
    mecab = MeCab.Tagger()
    logger.info("MeCab 초기화 성공")
except Exception as e:
    logger.error(f"MeCab 초기화 실패: {e}")
    mecab = None

# 기본 설정값
DEFAULT_MIN_TOKENS = 1
DEFAULT_MAX_TOKENS = 50
DEFAULT_SIMILARITY_THRESHOLD = 0.4

# SoyNLP tokenizer 초기화
tokenizer = LTokenizer()

# 미리 컴파일된 정규식
hanja_re = regex.compile(r'\p{Han}+')
hangul_re = regex.compile(r'^\p{Hangul}+$')
combined_re = regex.compile(
    r'(\p{Han}+)+(?:\p{Hangul}+)(?:은|는|이|가|을|를|에|에서|으로|로|와|과|도|만|며|고|하고|의|때)?'
)

def get_jieba_boundaries(text: str) -> set:
    """jieba 토큰 경계 위치 계산"""
    try:
        jieba_tokens = list(jieba.cut(text, cut_all=False))
        boundaries = set()
        pos = 0
        for token in jieba_tokens:
            pos += len(token)
            boundaries.add(pos)
        return boundaries
    except Exception as e:
        logger.warning(f"jieba 분석 실패: {e}")
        return set()

def split_src_meaning_units(text: str) -> List[str]:
    """한문 텍스트를 '한자+조사+어미' 단위로 묶어서 분할 (jieba 기본 적용)"""
    text = text.replace('\n', ' ').replace('：', '： ')
    
    # 공백으로 분리된 토큰들
    tokens = regex.findall(r'\S+', text)
    
    # jieba 경계 분석
    jieba_boundaries = get_jieba_boundaries(text)
    
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

        # 2) 순수 한자 토큰: jieba 경계 확인해서 병합
        if hanja_re.search(tok):
            unit = tok
            j = i + 1
            
            # 한글 토큰들과 병합 (jieba 경계 고려)
            while j < len(tokens) and hangul_re.match(tokens[j]):
                # 병합할지 결정 (단순하게 처리)
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

        # 4) 기타 토큰
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

def calculate_token_count_score(actual_count: int, target_count: int) -> float:
    """토큰 수에 따른 점수 계산 (누락된 함수 추가)"""
    if target_count <= 0:
        return 0.5
    
    ratio = actual_count / target_count
    if 0.8 <= ratio <= 1.2:
        return 1.0
    elif 0.5 <= ratio <= 1.5:
        return 0.8
    elif 0.3 <= ratio <= 2.0:
        return 0.6
    else:
        return 0.3

def calculate_mecab_completeness(span: str) -> float:
    """MeCab 기반 완전성 점수"""
    if mecab is None:
        return 0.5
        
    try:
        result = mecab.parse(span).strip()
        
        # MeCab 출력 파싱
        morphs = []
        for line in result.split('\n'):
            if line and '\t' in line and line != 'EOS':
                parts = line.split('\t')
                if len(parts) >= 2:
                    morph = parts[0]
                    features = parts[1].split(',')
                    pos = features[0] if features else 'UNK'
                    morphs.append((morph, pos))
        
        if not morphs:
            return 0.5
            
        score = 0.0
        
        # 한국어 품사 태그에 맞춘 완전성 확인
        has_noun = any(pos.startswith('NN') or pos.startswith('NP') for _, pos in morphs)
        has_verb = any(pos.startswith('VV') or pos.startswith('VA') for _, pos in morphs)
        has_ending = any(pos.startswith('EF') or pos.startswith('EC') for _, pos in morphs)
        has_josa = any(pos.startswith('JK') or pos.startswith('JX') for _, pos in morphs)
        has_adjective = any(pos.startswith('MM') or pos.startswith('MA') for _, pos in morphs)
        
        # 점수 계산
        if has_noun: score += 0.2
        if has_verb: score += 0.2
        if has_ending: score += 0.2
        if has_josa: score += 0.1
        if has_adjective: score += 0.1
        
        # 형태소 수 적절성
        morph_count = len(morphs)
        if 2 <= morph_count <= 4:
            score += 0.2
        elif morph_count == 1:
            score += 0.1
        elif morph_count > 6:
            score -= 0.1
            
        return max(0.0, min(1.0, score))
        
    except Exception as e:
        logger.warning(f"MeCab 분석 실패: {e}")
        return 0.5

def find_target_span_end_semantic(
    src_unit: str, 
    remaining_tgt: str, 
    embed_func: Callable,
    start_idx: int = 0,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    target_token_count: int = 3
) -> Tuple[int, float]:
    """최적화된 타겟 스팬 종료 지점 찾기 (MeCab 기본 적용)"""
    
    tokens = remaining_tgt.split()
    if len(tokens) == 0:
        return 0, 0.0
    
    max_tokens = min(max_tokens, len(tokens))
    min_tokens = max(1, min(min_tokens, len(tokens)))
    
    # 배치로 후보 스팬들 생성
    candidate_spans = []
    span_lengths = []
    
    for end in range(min_tokens, max_tokens + 1):
        span_text = " ".join(tokens[:end])
        candidate_spans.append(span_text)
        span_lengths.append(end)
    
    if not candidate_spans:
        return min_tokens, 0.0
    
    try:
        src_emb = embed_func([src_unit])[0]
        tgt_embs = embed_func(candidate_spans)
        
        best_end = min_tokens
        best_score = -1.0
        
        for i, (span_text, end, tgt_emb) in enumerate(zip(candidate_spans, span_lengths, tgt_embs)):
            # 의미 유사도
            sim = float(np.dot(src_emb, tgt_emb) / (np.linalg.norm(src_emb) * np.linalg.norm(tgt_emb) + 1e-8))
            
            # 길이 점수
            length_score = calculate_token_count_score(end, target_token_count)
            
            # MeCab 완전성 점수 (기본 적용)
            mecab_score = calculate_mecab_completeness(span_text)
            
            # 통합 점수
            total_score = sim * 0.7 + length_score * 0.2 + mecab_score * 0.1
            
            if total_score > best_score:
                best_score = total_score
                best_end = end
        
        return best_end, best_score
        
    except Exception as e:
        logger.warning(f"스팬 탐색 실패: {e}")
        return min_tokens, 0.0

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