"""원문과 번역문을 의미 단위로 분할하는 모듈 - jieba & MeCab 활용"""

import logging
import numpy as np
import regex
import re
import itertools
from typing import List, Callable
import jieba
import MeCab
import os

logger = logging.getLogger(__name__)

# 기본 설정값
DEFAULT_MIN_TOKENS = 1
DEFAULT_MAX_TOKENS = 50
DEFAULT_SIMILARITY_THRESHOLD = 0.7

# jieba와 MeCab 초기화
try:
    # 사용자 사전 경로를 .venv/Scripts/user.dic로 지정
    mecabrc_path = 'C:/Users/junto/Downloads/head-repo/CSP/.venv/Lib/site-packages/mecab_ko_dic/dicdir/mecabrc'
    dicdir_path = 'C:/Users/junto/Downloads/head-repo/CSP/.venv/Lib/site-packages/mecab_ko_dic/dicdir'
    userdic_path = 'C:/Users/junto/Downloads/head-repo/CSP/.venv/Lib/site-packages/mecab_ko_dic/dicdir/user.dic'
    mecab = MeCab.Tagger(f'-r {mecabrc_path} -d {dicdir_path} -u {userdic_path}')
    print("✅ MeCab 초기화 성공")
    logger.info("✅ MeCab 초기화 성공") # -d는 사전 디렉토리, -u는 사용자 사전 경로
except Exception as e:
    print(f"⚠️ MeCab 초기화 실패: {e}")
    logger.warning(f"⚠️ MeCab 초기화 실패: {e}")
    mecab = None

# 미리 컴파일된 정규식
hanja_re = regex.compile(r'\p{Han}+')
hangul_re = regex.compile(r'^\p{Hangul}+$')

def split_src_meaning_units(
    text: str,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    by_space: bool = False,
    **kwargs
):
    """원문(한문+한글)을 jieba와 MeCab으로 의미 단위 분할"""
    
    # 1단계: 어절 단위로 분리 (어절 내부는 절대 쪼개지지 않음)
    # 전각 콜론 뒤에만 공백을 추가하여 "전운(箋云)：" + "갈대는" 형태로 분할
    words = text.replace('\n', ' ').replace('：', '： ').split()
    if not words:
        return []
    
    # 2단계: jieba와 MeCab 분석 결과 준비
    jieba_tokens = list(jieba.cut(text))
    
    # MeCab 분석 (한글 부분용)
    morpheme_info = []
    if mecab:
        result = mecab.parse(text)
        for line in result.split('\n'):
            if line and line != 'EOS':
                parts = line.split('\t')
                if len(parts) >= 2:
                    surface = parts[0]
                    pos = parts[1].split(',')[0]
                    morpheme_info.append((surface, pos))
    
    # 3단계: 어절들을 의미 단위로 그룹화 (jieba + MeCab 정보 활용)
    units = []
    i = 0
    
    while i < len(words):
        word = words[i]
        
        # 한자 포함 어절 처리
        if hanja_re.search(word):
            # 현재 어절이 한자를 포함하면 하나의 의미 단위
            units.append(word)
            i += 1
            continue
        
        # 한글 어절들 처리 - jieba와 MeCab 분석 결과 모두 참고
        if hangul_re.match(word):
            group = [word]
            j = i + 1
            
            # 중세국어 어미나 문법 표지로 경계 판단 (원문용)
            should_break_here = _should_break_by_mecab_src(word, morpheme_info) if morpheme_info else False
            
            # jieba 토큰 연속성도 확인 (경계 신호가 없는 경우만)
            if not should_break_here:
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
        
        # 기타 어절 (숫자, 구두점 등)
        units.append(word)
        i += 1
    
    return units

def _should_group_words_by_jieba(word_group: List[str], jieba_tokens: List[str]) -> bool:
    """jieba 분석 결과를 참고해서 어절들을 묶을지 결정"""
    combined = ''.join(word_group)
    
    # jieba 토큰 중에서 현재 조합과 일치하는 것이 있으면 묶기
    for token in jieba_tokens:
        if token.replace(' ', '') == combined.replace(' ', ''):
            return True
    
    # 길이 제한
    if len(combined) > 10:
        return False
    
    return len(word_group) <= 3

def split_inside_chunk(chunk: str) -> List[str]:
    """번역문 청크를 의미 단위로 분할 - MeCab 분석 참고 (개선된 버전)"""
    
    if not chunk or not chunk.strip():
        return []
    
    # 1단계: 어절 단위로 분리 (어절 내부는 절대 쪼개지지 않음)
    # 전각 콜론 뒤에만 공백을 추가하여 "전운(箋云)：" + "갈대는" 형태로 분할
    words = chunk.replace('：', '： ').split()
    if not words:
        return []
    
    # 2단계: MeCab 분석 결과 참고
    morpheme_info = []
    if mecab:
        result = mecab.parse(chunk)
        for line in result.split('\n'):
            if line and line != 'EOS':
                parts = line.split('\t')
                if len(parts) >= 2:
                    surface = parts[0]
                    pos = parts[1].split(',')[0]
                    morpheme_info.append((surface, pos))
    
    # 3단계: MeCab 분석 결과를 활용한 의미 단위 그룹화
    units = []
    current_group = []
    
    for word in words:
        current_group.append(word)
        
        # 쉼표 분할을 최우선으로 처리 (다른 조건보다 먼저)
        if word.endswith(',') or word.endswith('，'):
            units.append(' '.join(current_group))
            current_group = []
            continue
        
        # 전각 콜론으로 끝나는 단어는 즉시 단위 완성 (하드 경계)
        if word.endswith('：') or word == '：':
            units.append(' '.join(current_group))
            current_group = []
            continue
        
        # MeCab 분석 결과로 경계 판단 (품사 정보 활용)
        should_break = _should_break_by_mecab(word, morpheme_info) if morpheme_info else False
        
        if should_break and current_group:
            units.append(' '.join(current_group))
            current_group = []
    
    if current_group:
        units.append(' '.join(current_group))
    
    return [unit.strip() for unit in units if unit.strip()]

def _should_break_by_mecab(word: str, morpheme_info: List[tuple]) -> bool:
    """MeCab 분석 결과를 참고해서 의미 단위 경계 결정 - 보조사(JX) 강화"""
    
    # MeCab 분석 결과 확인
    for surface, pos in morpheme_info:
        # 단어가 해당 형태소로 끝나는지 확인 (더 정확한 매칭)
        if word.endswith(surface):
            # 강한 경계 신호 - 종결어미, 구두점
            if pos in ['EF', 'SF', 'SP']:
                return True
            
            # 보조사(JX) - 매우 중요한 문법적 표지로 강화 처리
            if pos == 'JX':
                return True  # 모든 보조사에서 분할
            
            # 주요 조사들 - 의미 단위 경계
            if pos in ['JKS', 'JKO', 'JKC', 'JKB', 'JKG', 'JKV', 'JKQ']:
                return True  # 모든 조사에서 분할
            
            # 연결어미(EC) - 문장 연결
            if pos == 'EC':
                return True  # 모든 연결어미에서 분할
            
            # 명사형 전성어미(ETN) - 명사화
            if pos == 'ETN':
                return True
            
            # 관형형 전성어미(ETM) - 관형어화
            if pos == 'ETM':
                return True
            
            # 동사, 형용사 어간 다음에서 경계  
            if pos in ['VV', 'VA', 'VX']:
                return len(surface) >= 1  # 길이 1 이상인 용언 어간에서 분할
            
            # 중요한 부사에서 분할 (MAG, MAJ)
            if pos in ['MAG', 'MAJ'] and len(surface) >= 2:
                return True  # 길이 2 이상인 부사에서 분할
    
    return False

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
    """원문 단위에 따른 번역문 분할 (의미 기반, 전역 매칭)"""
    
    # 1단계: 전각 콜론을 하드 경계로 처리
    if '：' in tgt_text:
        colon_parts = tgt_text.split('：')
        if len(colon_parts) == 2:
            part1 = colon_parts[0].strip() + '：'
            part2 = colon_parts[1].strip()
            result = [part1]
            if len(src_units) > 1:
                remaining_src = src_units[1:]
                remaining_parts = split_tgt_by_src_units_semantic(
                    remaining_src, part2, embed_func, min_tokens
                )
                result.extend(remaining_parts)
            else:
                result.append(part2)
            return result
    
    # 2단계: 번역문을 먼저 자연스러운 단위로 분할
    tgt_chunks = split_inside_chunk(tgt_text)
    if not tgt_chunks or len(src_units) == 0:
        return tgt_chunks if tgt_chunks else []
    
    # 3단계: 의미 기반 전역 매칭
    if len(src_units) == len(tgt_chunks):
        # 1:1 대응인 경우 의미 유사도로 최적 매칭 찾기 (순서 보존)
        return _find_optimal_semantic_matching(src_units, tgt_chunks, embed_func, preserve_order=True)
    elif len(src_units) == 1:
        # 원문이 하나인 경우 - 번역문을 하나로 합치거나 DP 매칭 사용
        if len(tgt_chunks) <= 3:  # 작은 개수면 하나로 합치기
            return [tgt_text.strip()]
        else:
            # 많은 개수면 DP 매칭 사용
            return _dp_semantic_matching(src_units, tgt_text, embed_func, min_tokens)
    elif len(tgt_chunks) == 1:
        # 번역문이 하나인 경우 - 원문 개수만큼 분할 시도
        return _split_single_target_to_multiple(src_units, tgt_chunks[0], embed_func)
    else:
        # 개수가 다른 경우 DP 매칭 사용
        return _dp_semantic_matching(src_units, tgt_text, embed_func, min_tokens)

def _find_optimal_semantic_matching(src_units: List[str], tgt_chunks: List[str], embed_func: Callable, preserve_order: bool = True) -> List[str]:
    """원문과 번역문 청크 간의 최적 의미 매칭 찾기 (순서 보존 옵션 추가)"""
    import itertools
    
    if len(src_units) != len(tgt_chunks):
        return tgt_chunks
    
    # 순서 보존 모드인 경우 재정렬 없이 그대로 반환
    if preserve_order:
        return tgt_chunks
    
    try:
        # 원문과 번역문 임베딩 계산
        normalized_src = [_normalize_for_embedding(src) for src in src_units]
        normalized_tgt = [_normalize_for_embedding(tgt) for tgt in tgt_chunks]
        
        src_embs = embed_func(normalized_src)
        tgt_embs = embed_func(normalized_tgt)
        
        # 모든 가능한 매칭에 대해 종합 유사도 계산
        best_score = -1
        best_permutation = list(range(len(tgt_chunks)))
        
        for perm in itertools.permutations(range(len(tgt_chunks))):
            total_score = 0
            for i, j in enumerate(perm):
                # 1. 의미 유사도 계산 (코사인 유사도)
                sim = float(np.dot(src_embs[i], tgt_embs[j]) / 
                          (np.linalg.norm(src_embs[i]) * np.linalg.norm(tgt_embs[j]) + 1e-8))
                
                # 2. 키워드 매칭 보너스 (한자, 고유명사 등)
                keyword_bonus = _calculate_keyword_bonus(src_units[i], tgt_chunks[j])
                
                # 3. 문법적 경계 보너스
                grammar_bonus = _calculate_grammar_bonus(tgt_chunks[j])
                
                # 4. 구문 구조 매칭 보너스
                structure_bonus = _calculate_structure_bonus(src_units[i], tgt_chunks[j])
                
                # 5. 길이 균형 보너스 (너무 불균형한 매칭 방지)
                length_bonus = _calculate_length_balance_bonus(src_units[i], tgt_chunks[j])
                
                # 6. 순서 보존 보너스 (원래 순서 유지 선호)
                order_bonus = 0.3 if i == j else -0.1  # 원래 순서면 보너스, 아니면 페널티
                
                total_score += (sim * 1.0 + keyword_bonus * 0.8 + grammar_bonus * 0.6 + 
                               structure_bonus * 0.5 + length_bonus * 0.3 + order_bonus)
            
            if total_score > best_score:
                best_score = total_score
                best_permutation = perm
        
        # 최적 매칭 순서로 반환
        return [tgt_chunks[i] for i in best_permutation]
        
    except Exception as e:
        logger.warning(f"의미 매칭 실패, 원본 순서 유지: {e}")
        return tgt_chunks

def _calculate_keyword_bonus(src_unit: str, tgt_chunk: str) -> float:
    """키워드 매칭 보너스 계산 - 단순화된 버전"""
    bonus = 0.0
    
    # 1. 한자 추출
    src_hanja = regex.findall(r'\p{Han}+', src_unit)
    
    # 2. 한자 직접 매칭
    for hanja in src_hanja:
        if hanja in tgt_chunk:
            bonus += 0.5  # 한자 직접 매칭
            if len(hanja) >= 2:
                bonus += 0.2  # 긴 한자어 보너스
    
    # 3. 기본적인 문법 표지 매칭만 유지
    if '者' in src_unit and any(marker in tgt_chunk for marker in ['것', '자', '라는']):
        bonus += 0.3
    
    if '也' in src_unit and any(marker in tgt_chunk for marker in ['다', '이다', '것이다']):
        bonus += 0.3
    
    return min(bonus, 1.5)  # 최대값 제한


def _calculate_structure_bonus(src_unit: str, tgt_chunk: str) -> float:
    """구문 구조 매칭 보너스 계산 - 단순화된 버전"""
    bonus = 0.0
    
    # 1. 구두점 수 매칭만 유지
    src_punct = len(re.findall(r'[,，.。!！?？:：;；]', src_unit))
    tgt_punct = len(re.findall(r'[,，.。!！?？:：;；]', tgt_chunk))
    
    if src_punct == tgt_punct and src_punct > 0:
        bonus += 0.2
    
    # 2. 괄호 구조 매칭
    src_parens = src_unit.count('(') + src_unit.count('（')
    tgt_parens = tgt_chunk.count('(') + tgt_chunk.count('（')
    
    if src_parens == tgt_parens and src_parens > 0:
        bonus += 0.1
    
    return bonus

def _calculate_length_balance_bonus(src_unit: str, tgt_chunk: str) -> float:
    """길이 균형 보너스 계산 (너무 불균형한 매칭 방지)"""
    src_len = len(src_unit.strip())
    tgt_len = len(tgt_chunk.strip())
    
    if src_len == 0 or tgt_len == 0:
        return -0.5  # 빈 문자열 페널티
    
    # 길이 비율 계산
    ratio = min(src_len, tgt_len) / max(src_len, tgt_len)
    
    # 적절한 길이 비율에 보너스 (0.3 ~ 1.0 사이가 적절)
    if ratio >= 0.5:
        return 0.2 * ratio  # 균형 잡힌 길이에 보너스
    elif ratio >= 0.2:
        return 0.1 * ratio  # 약간 불균형한 경우 작은 보너스
    else:
        return -0.1  # 너무 불균형한 경우 페널티

def _dp_semantic_matching(src_units: List[str], tgt_text: str, embed_func: Callable, min_tokens: int) -> List[str]:
    """DP 기반 의미 매칭 (기존 로직)"""
    # 기존 DP 로직 유지 (백업용)
    tgt_tokens = tgt_text.replace('：', '： ').split()
    N, T = len(src_units), len(tgt_tokens)
    if N == 0 or T == 0:
        return []

    dp = np.full((N+1, T+1), -np.inf)
    back = np.zeros((N+1, T+1), dtype=int)
    dp[0, 0] = 0.0

    # 원문 단위들을 정규화하여 임베딩 계산
    normalized_src_units = [_normalize_for_embedding(unit) for unit in src_units]
    src_embs = embed_func(normalized_src_units)

    # 모든 후보 span 수집
    span_map = {}
    all_spans = []
    for i in range(1, N+1):
        for j in range(i*min_tokens, T-(N-i)*min_tokens+1):
            for k in range((i-1)*min_tokens, j-min_tokens+1):
                span = " ".join(tgt_tokens[k:j]).strip()
                key = (k, j)
                if span and key not in span_map:
                    span_map[key] = span
                    all_spans.append(span)
    
    all_spans = list(set(all_spans))

    # 배치 임베딩
    def batch_embed(spans, batch_size=100):
        results = []
        for i in range(0, len(spans), batch_size):
            batch_spans = spans[i:i+batch_size]
            normalized_batch = [_normalize_for_embedding(span) for span in batch_spans]
            results.extend(embed_func(normalized_batch))
        return results
    
    span_embs = batch_embed(all_spans)
    span_emb_dict = {span: emb for span, emb in zip(all_spans, span_embs)}

    # DP 계산 (개선된 의미 유사도 + 다중 보너스)
    for i in range(1, N+1):
        for j in range(i*min_tokens, T-(N-i)*min_tokens+1):
            for k in range((i-1)*min_tokens, j-min_tokens+1):
                span = span_map[(k, j)]
                tgt_emb = span_emb_dict[span]
                
                # 1. 기본 의미 유사성 (코사인 유사도)
                sim = float(np.dot(src_embs[i-1], tgt_emb)/((np.linalg.norm(src_embs[i-1])*np.linalg.norm(tgt_emb))+1e-8))
                
                # 2. 키워드 매칭 보너스
                keyword_bonus = _calculate_keyword_bonus(src_units[i-1], span)
                
                # 3. 문법적 경계 보너스
                grammar_bonus = _calculate_grammar_bonus(span)
                
                # 4. 구문 구조 매칭 보너스
                structure_bonus = _calculate_structure_bonus(src_units[i-1], span)
                
                # 5. 길이 균형 보너스
                length_bonus = _calculate_length_balance_bonus(src_units[i-1], span)
                
                # 가중치 적용한 최종 점수
                final_score = (sim * 1.0 + keyword_bonus * 0.8 + grammar_bonus * 0.6 + 
                              structure_bonus * 0.5 + length_bonus * 0.3)
                
                score = dp[i-1, k] + final_score
                
                if score > dp[i, j]:
                    dp[i, j] = score
                    back[i, j] = k

    # 역추적
    cuts = [T]
    curr = T
    for i in range(N, 0, -1):
        prev = int(back[i, curr])
        cuts.append(prev)
        curr = prev
    cuts = cuts[::-1]

    tgt_spans = []
    for i in range(N):
        span = " ".join(tgt_tokens[cuts[i]:cuts[i+1]]).strip()
        tgt_spans.append(span)
    return tgt_spans

def split_tgt_meaning_units(
    src_text: str,
    tgt_text: str,
    use_semantic: bool = False,  # 기본값을 False로 변경 (순차 모드 우선)
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    embed_func: Callable = None
) -> List[str]:
    """번역문을 의미 단위로 분할 - 순차 방식 우선"""
    
    # 기본적으로 순차 분할 사용 (순서 보장)
    if not use_semantic:
        return split_tgt_meaning_units_sequential(
            src_text, tgt_text, min_tokens, max_tokens
        )
    
    # 기존 의미 기반 방식 (하위 호환용)
    if embed_func is None:
        from sa_embedders import compute_embeddings_with_cache
        embed_func = compute_embeddings_with_cache
        
    src_units = split_src_meaning_units(src_text, min_tokens, max_tokens)
    return split_tgt_by_src_units_semantic(
        src_units, tgt_text, embed_func=embed_func, min_tokens=min_tokens
    )

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
        return [(word, 'UNKNOWN') for word in text.split()]

def sentence_split(text):
    """문장 단위로 분리"""
    sentences = re.split(r'[.!?。！？]+', text)
    return [s.strip() for s in sentences if s.strip()]

def normalize_for_embedding(text: str) -> str:
    """임베딩 계산을 위해 텍스트 정규화 - 전각 콜론 등 구두점 제거"""
    # 전각 콜론과 괄호 등을 제거하여 의미 매칭에 집중
    normalized = text.replace('：', '').replace('(', '').replace(')', '')
    # 연속된 공백을 하나로 정리
    normalized = ' '.join(normalized.split())
    return normalized

def _normalize_for_embedding(text: str) -> str:
    """임베딩 계산을 위한 텍스트 정규화 - 전각 콜론 제거"""
    return text.replace('：', '').strip()

def _calculate_grammar_bonus(span: str) -> float:
    """문법적 경계에 대한 보너스 점수 계산 - MeCab 기반 단순화 버전"""
    span = span.strip()
    bonus = 0.0
    
    # 1. 전각 콜론으로 끝나는 경우 강한 보너스
    if span.endswith('：'):
        return 0.8
    
    # 2. MeCab을 이용한 정확한 어미/조사 분석
    if mecab:
        try:
            result = mecab.parse(span)
            last_pos = None
            for line in result.split('\n'):
                if line and line != 'EOS':
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        pos_detail = parts[1].split(',')
                        last_pos = pos_detail[0]
            
            # 품사별 보너스 - 단순화
            if last_pos == 'EF':  # 종결어미
                bonus = 0.5
            elif last_pos == 'EC':  # 연결어미
                bonus = 0.4
            elif last_pos == 'JX':  # 보조사
                bonus = 0.4
            elif last_pos in ['JKS', 'JKO', 'JKB', 'JKC']:  # 주요 조사
                bonus = 0.3
            elif last_pos in ['ETN', 'ETM']:  # 전성어미
                bonus = 0.3
        except:
            pass
    
    # 3. 기본 구두점 처리
    if span.endswith(('.', '。', '!', '！', '?', '？')):
        bonus = max(bonus, 0.4)
    elif span.endswith((',', '，', ';', '；')):
        bonus = max(bonus, 0.2)
        
    return min(bonus, 1.0)

def _split_single_target_to_multiple(src_units: List[str], single_tgt: str, embed_func: Callable) -> List[str]:
    """단일 번역문을 여러 원문 단위에 맞게 분할"""
    # 원문이 여러 개이고 번역문이 하나인 경우
    # 번역문을 자연스러운 경계에서 분할하여 원문 개수에 맞춤
    
    # 먼저 자연스러운 분할 시도
    natural_splits = split_inside_chunk(single_tgt)
    
    if len(natural_splits) >= len(src_units):
        # 자연 분할이 충분한 경우, 의미적으로 가장 적합한 조합 찾기
        return _merge_splits_to_match_src_count(src_units, natural_splits, embed_func)
    else:
        # 자연 분할이 부족한 경우, 강제 분할
        return _force_split_by_semantic_boundaries(src_units, single_tgt, embed_func)

def _merge_splits_to_match_src_count(src_units: List[str], tgt_splits: List[str], embed_func: Callable) -> List[str]:
    """번역문 분할을 원문 개수에 맞게 병합 - 단순화된 버전"""
    if len(src_units) >= len(tgt_splits):
        return tgt_splits
    
    # 너무 많이 split된 경우 앞에서부터 순차적으로 병합
    current_splits = tgt_splits[:]
    
    while len(current_splits) > len(src_units):
        # 가장 짧은 인접한 두 분할을 병합
        best_merge_idx = 0
        min_combined_length = float('inf')
        
        for i in range(len(current_splits) - 1):
            combined_length = len(current_splits[i]) + len(current_splits[i + 1])
            if combined_length < min_combined_length:
                min_combined_length = combined_length
                best_merge_idx = i
        
        # 병합 실행
        merged_text = current_splits[best_merge_idx] + ' ' + current_splits[best_merge_idx + 1]
        current_splits = (current_splits[:best_merge_idx] + 
                         [merged_text] + 
                         current_splits[best_merge_idx + 2:])
    
    return current_splits

def _force_split_by_semantic_boundaries(src_units: List[str], single_tgt: str, embed_func: Callable) -> List[str]:
    """의미적 경계를 기준으로 강제 분할 - 단순화된 버전"""
    tokens = single_tgt.split()
    if len(tokens) <= len(src_units):
        return [single_tgt]  # 토큰이 부족하면 그대로 반환
    
    # 단순하게 토큰을 거의 균등하게 분할
    tokens_per_unit = len(tokens) // len(src_units)
    remainder = len(tokens) % len(src_units)
    
    result = []
    start = 0
    
    for i in range(len(src_units)):
        # 나머지가 있으면 앞쪽 단위들에 하나씩 더 배분
        current_size = tokens_per_unit + (1 if i < remainder else 0)
        end = start + current_size
        
        if end > len(tokens):
            end = len(tokens)
        
        if start < end:
            segment = ' '.join(tokens[start:end]).strip()
            if segment:
                result.append(segment)
        
        start = end
    
    # 결과가 부족하면 마지막 것을 반환
    if not result:
        result = [single_tgt]
    
    return result

def _should_break_by_mecab_src(word: str, morpheme_info: List[tuple]) -> bool:
    """원문용 - MeCab 분석 결과 + 중세국어 어미 패턴으로 의미 단위 경계 결정"""
    
    # 1. 중세국어 어미 패턴 확인 (원문에만 적용)
    middle_korean_endings = [
        '니라', '노라', '도다', '로다', '가다', '거다',  # 종결어미
        '려니와', '거니와', '로되', '되',              # 연결어미
        '건댄', '건대', '어니', '거니',               # 연결어미
        '하니', '하되', '하여', '하야',               # 연결어미
        '이니', '이로', '이며', '이면',               # 연결어미
        '은즉', '즉', '면', '니',                    # 연결어미
        '라도', '마는', '나마', '려마는'              # 보조사/연결어미
    ]
    
    for ending in middle_korean_endings:
        if word.endswith(ending):
            return True
    
    # 2. 일반적인 MeCab 분석 결과 확인 (번역문과 동일)
    return _should_break_by_mecab(word, morpheme_info)

def split_by_whitespace_and_colon(text: str) -> List[str]:
    """공백 및 전각 콜론 기준 분할 (PA 방식과 동일)"""
    if not text or not text.strip():
        return []
    
    # 1단계: 전각 콜론 기준 분할
    parts = text.split('：')
    if len(parts) > 1:
        # 첫 번째 부분에 콜론 붙이기
        parts[0] = parts[0] + '：'
        # 나머지 부분들은 그대로
    
    # 2단계: 각 부분을 공백 기준으로 분할
    result = []
    for part in parts:
        words = part.strip().split()
        result.extend(words)
    
    return [word for word in result if word.strip()]

def merge_target_by_source_sequential(src_units: List[str], tgt_tokens: List[str]) -> List[str]:
    """원문 단위 기준으로 번역문 토큰을 순차적으로 병합 (PA 방식)"""
    if not src_units or not tgt_tokens:
        return tgt_tokens if tgt_tokens else []
    
    # 원문과 번역문 비율 계산
    src_count = len(src_units)
    tgt_count = len(tgt_tokens)
    
    if src_count == 1:
        # 원문이 하나면 번역문 전체를 하나로
        return [' '.join(tgt_tokens)]
    
    # 번역문 토큰을 원문 개수만큼 분할
    tokens_per_unit = tgt_count // src_count
    remainder = tgt_count % src_count
    
    result = []
    start_idx = 0
    
    for i in range(src_count):
        # 나머지가 있으면 앞쪽 단위들에 하나씩 더 배분
        current_size = tokens_per_unit + (1 if i < remainder else 0)
        end_idx = start_idx + current_size
        
        if end_idx > tgt_count:
            end_idx = tgt_count
        
        if start_idx < end_idx:
            unit_tokens = tgt_tokens[start_idx:end_idx]
            result.append(' '.join(unit_tokens))
        
        start_idx = end_idx
    
    return result

def split_tgt_meaning_units_sequential(
    src_text: str,
    tgt_text: str,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    **kwargs
) -> List[str]:
    """순차적 분할 방식 - 순서 보장, 쉼표 분할 지원"""
    
    # 1단계: 원문 의미 단위 분할
    src_units = split_src_meaning_units(src_text, min_tokens, max_tokens)
    
    # 2단계: 번역문을 자연스럽게 분할 (쉼표, 콜론, MeCab 기준)
    tgt_chunks = split_inside_chunk(tgt_text)
    
    # 3단계: 원문 개수에 맞춰 번역문 조정
    if len(src_units) == len(tgt_chunks):
        # 1:1 매칭 - 순서 그대로 유지
        return tgt_chunks
    elif len(src_units) == 1:
        # 원문 1개 - 번역문 전체 합치기
        return [tgt_text.strip()]
    elif len(tgt_chunks) == 1:
        # 번역문 1개 - 원문 개수만큼 단순 분할
        return _simple_split_by_tokens(tgt_text, len(src_units))
    elif len(tgt_chunks) > len(src_units):
        # 번역문이 많음 - 순차적 병합
        return _merge_target_chunks_sequential(tgt_chunks, len(src_units))
    else:
        # 원문이 많음 - 번역문 단순 분할
        return _simple_split_by_tokens(tgt_text, len(src_units))

def _simple_split_by_tokens(text: str, target_count: int) -> List[str]:
    """텍스트를 토큰 기준으로 단순 분할"""
    tokens = text.split()
    if len(tokens) <= target_count:
        return [text]
    
    tokens_per_unit = len(tokens) // target_count
    remainder = len(tokens) % target_count
    
    result = []
    start = 0
    
    for i in range(target_count):
        current_size = tokens_per_unit + (1 if i < remainder else 0)
        end = start + current_size
        
        if start < len(tokens):
            segment = ' '.join(tokens[start:end]).strip()
            if segment:
                result.append(segment)
        start = end
    
    return result

def _merge_target_chunks_sequential(chunks: List[str], target_count: int) -> List[str]:
    """번역문 청크를 순차적으로 병합"""
    if len(chunks) <= target_count:
        return chunks
    
    result = chunks[:]
    
    while len(result) > target_count:
        # 가장 짧은 인접한 두 청크 병합
        min_length = float('inf')
        merge_idx = 0
        
        for i in range(len(result) - 1):
            combined_length = len(result[i]) + len(result[i + 1])
            if combined_length < min_length:
                min_length = combined_length
                merge_idx = i
        
        # 병합 실행
        merged = result[merge_idx] + ' ' + result[merge_idx + 1]
        result = result[:merge_idx] + [merged] + result[merge_idx + 2:]
    
    return result