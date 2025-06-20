"""원문(jieba) + 번역문(Kkma) 토크나이저"""

import logging
import numpy as np
import regex
import re
from typing import List, Callable
import jieba  # 원문용
from konlpy.tag import Kkma  # 번역문용

logger = logging.getLogger(__name__)

# 기본 설정값
DEFAULT_MIN_TOKENS = 1
DEFAULT_MAX_TOKENS = 50
DEFAULT_SIMILARITY_THRESHOLD = 0.4

# jieba 초기화 (원문용)
try:
    jieba.initialize()
    logger.info("✅ jieba 초기화 성공")
except Exception as e:
    logger.warning(f"⚠️ jieba 초기화 실패: {e}")

# Kkma 초기화 (번역문용)
try:
    kkma = Kkma()
    logger.info("✅ Kkma 초기화 성공")
except Exception as e:
    logger.warning(f"⚠️ Kkma 초기화 실패: {e}")
    kkma = None

# 미리 컴파일된 정규식
hanja_re = regex.compile(r'\p{Han}+')
hangul_re = regex.compile(r'^\p{Hangul}+$')

def split_src_meaning_units(
    text: str,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    use_advanced: bool = True
) -> List[str]:
    """한문 텍스트를 의미 단위로 분할 - 항상 jieba 사용"""
    
    # jieba_mecab.py와 동일한 로직
    words = text.replace('\n', ' ').replace('：', '： ').split()
    if not words:
        return []
    
    jieba_tokens = list(jieba.cut(text))
    
    units = []
    i = 0
    
    while i < len(words):
        word = words[i]
        
        if hanja_re.search(word):
            units.append(word)
            i += 1
            continue
        
        if hangul_re.match(word):
            group = [word]
            j = i + 1
            
            while j < len(words) and hangul_re.match(words[j]):
                should_group = _should_group_words_by_jieba(group + [words[j]], jieba_tokens)
                if should_group:
                    group.append(words[j])
                    j += 1
                else:
                    break
            
            units.append(' '.join(group))
            i = j
            continue
        
        units.append(word)
        i += 1
    
    return units

def _should_group_words_by_jieba(word_group: List[str], jieba_tokens: List[str]) -> bool:
    """jieba 분석 결과를 참고해서 어절들을 묶을지 결정"""
    combined = ''.join(word_group)
    
    for token in jieba_tokens:
        if token.replace(' ', '') == combined.replace(' ', ''):
            return True
    
    if len(combined) > 10:
        return False
    
    return len(word_group) <= 3

def split_inside_chunk(chunk: str) -> List[str]:
    """번역문 청크를 의미 단위로 분할 - Kkma 사용"""
    
    if not chunk or not chunk.strip():
        return []
    
    words = chunk.split()
    if not words:
        return []
    
    # Kkma 형태소 분석
    morpheme_info = []
    if kkma:
        try:
            pos_tags = kkma.pos(chunk)
            morpheme_info = pos_tags
        except Exception as e:
            logger.warning(f"Kkma 분석 실패: {e}")
    
    # 기본 패턴으로 어절들을 의미 단위로 그룹화
    delimiters = ['을', '를', '이', '가', '은', '는', '에', '에서', '로', '으로',
                  '와', '과', '고', '며', '하고', '때', '의', '도', '만', '때에', '：']
    
    units = []
    current_group = []
    
    for word in words:
        current_group.append(word)
        
        # 기본 패턴: 조사/어미로 끝나면 의미 단위 완성
        should_break = any(word.endswith(delimiter) for delimiter in delimiters)
        
        # Kkma 분석 결과 참고: 품사 정보로 경계 조정
        if morpheme_info and not should_break:
            should_break = _should_break_by_kkma(word, morpheme_info)
        
        if should_break and current_group:
            units.append(' '.join(current_group))
            current_group = []
    
    if current_group:
        units.append(' '.join(current_group))
    
    return [unit.strip() for unit in units if unit.strip()]

def _should_break_by_kkma(word: str, morpheme_info: List[tuple]) -> bool:
    """Kkma 분석 결과를 참고해서 의미 단위 경계 결정"""
    
    # word에 해당하는 형태소들의 품사 확인
    for surface, pos in morpheme_info:
        if surface in word:
            # 조사, 어미, 구두점에서 경계
            if pos in ['JKS', 'JKO', 'JKC', 'JX', 'EF', 'EC', 'ETN', 'SF', 'SP']:
                return True
            # 동사, 형용사 어간 다음에서 경계  
            if pos in ['VV', 'VA', 'VX']:
                return True
    
    return False

# 나머지 함수들은 soynlp.py와 동일한 구조로...
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
        src_emb = embed_func([src_unit])[0]
        
        tgt_tokens = remaining_tgt.split()
        if not tgt_tokens:
            return 0
            
        upper = min(len(tgt_tokens), max_tokens)
        cumulative_lengths = [0]
        current_length = 0
        
        for tok in tgt_tokens:
            current_length += len(tok) + 1
            cumulative_lengths.append(current_length)
            
        candidates = []
        candidate_indices = []
        
        step_size = 1 if upper <= 10 else 2
        
        for end_i in range(min_tokens-1, upper, step_size):
            cand = " ".join(tgt_tokens[:end_i+1])
            candidates.append(cand)
            candidate_indices.append(end_i)
            
        cand_embs = embed_func(candidates)
        
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

    src_embs = embed_func(src_units)

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

    cuts = [T]
    curr = T
    for i in range(N, 0, -1):
        prev = int(back[i, curr])
        cuts.append(prev)
        curr = prev
    cuts = cuts[::-1]
    assert cuts[0] == 0 and cuts[-1] == T and len(cuts) == N + 1

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
    if embed_func is None:
        from sa_embedders import compute_embeddings_with_cache
        embed_func = compute_embeddings_with_cache
        
    src_units = split_src_meaning_units(src_text, min_tokens, max_tokens)

    if use_semantic:
        return split_tgt_by_src_units_semantic(
            src_units,
            tgt_text,
            embed_func=embed_func,
            min_tokens=min_tokens
        )
    else:
        return split_tgt_by_src_units(src_units, tgt_text)

def tokenize_text(text):
    """형태소 분석 및 토큰화 - Kkma 사용"""
    if kkma:
        try:
            return [word for word, _ in kkma.pos(text)]
        except:
            return text.split()
    else:
        return text.split()

def pos_tag_text(text):
    """품사 태깅 - Kkma 사용"""
    if kkma:
        try:
            return kkma.pos(text)
        except:
            return [(word, 'UNKNOWN') for word in text.split()]
    else:
        return [(word, 'UNKNOWN') for word in text.split()]

def sentence_split(text):
    """문장 단위로 분리"""
    sentences = re.split(r'[.!?。！？]+', text)
    return [s.strip() for s in sentences if s.strip()]