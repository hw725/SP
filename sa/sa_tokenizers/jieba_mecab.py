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

# 문법적 경계 표지 (간결한 정의)
BOUNDARY_MARKERS = {
    # 중세국어 어미 (원문 전용 - 표점 기능)
    '호되': ['boundary', 'quotative', 'split_after', 'src_only'],    # 인용 표지 (원문 표점 기능)
    '라': ['boundary', 'imperative', 'src_only'],                    # 명령/청유 표지 (원문 전용)
    '니': ['boundary', 'causal', 'src_only'],                        # 원인/이유 표지 (원문 전용)
    '되': ['boundary', 'adversative', 'src_only'],                   # 대립/전환 표지 (원문 전용)
    '니라': ['boundary', 'declarative', 'src_only'],                 # 서술 표지 (원문 전용)
    '나': ['boundary', 'interrogative', 'src_only'],                 # 의문 표지 (원문 전용)
    
    # 현대어 연결 표지 (번역문에서 사용)
    '므로': ['boundary', 'causal'],                                  # 인과 관계
    '서': ['boundary', 'causal'],                                    # 인과 관계 (단축)
    '면서': ['boundary', 'simultaneous'],                            # 동시 관계
}

# 콤마 경계 패턴 (병렬 제외)
COMMA_BOUNDARY_PATTERNS = [
    r'(?:하고|하며|하면서),',          # 순차적 연결
    r'(?:므로|서|니),',                # 인과 관계  
    r'(?:후|전|때),',                  # 시간 관계
    r'(?:면|거든|면서)',               # 조건/동시 관계
]

# 앞 구에 붙어야 하는 표현들 (새로운 구로 시작하면 안 됨)
ATTACH_TO_PREVIOUS_PATTERNS = [
    # 인용 표지 (라고 + 용언)
    r'^라고\s+(?:하[다시며면었을]|말하[다시며면았을]|생각하[다시며면었을])',  # 라고 하다/말하다/생각하다
    r'^라고\s+(?:한다|했다|하면|하며|하는|할)',                      # 라고 + 용언 활용
    
    # 관형 표지 (라는/라던 + 명사구)
    r'^라는\s+(?:것은|사실은|말은|점은|부분은)',                     # 라는 것은/사실은/말은 등
    r'^라던\s+(?:것은|말은|점은)',                                  # 라던 것은/말은 등
    
    # 기타 연결 표현
    r'^라며\s+',                                                   # 라며
    r'^라면서\s+',                                                 # 라면서
    r'^라고도\s+',                                                 # 라고도
    
    # 격조사로 시작하는 표현들
    r'^(?:이라|라)는\s+',                                          # 이라는/라는
    r'^(?:이라|라)고\s+',                                          # 이라고/라고
]

# 중세국어 어미 상세 정의 (detect_middle_korean_ending 함수용)
MIDDLE_KOREAN_ENDINGS = {
    '호되': {'type': 'connective', 'meaning': '역접/인용', 'split_after': True},
    '라': {'type': 'imperative', 'meaning': '명령/청유', 'split_after': True},
    '니': {'type': 'connective', 'meaning': '인과', 'split_after': True},
    '되': {'type': 'connective', 'meaning': '역접', 'split_after': True},
    '니라': {'type': 'declarative', 'meaning': '서술', 'split_after': True},
    '나': {'type': 'interrogative', 'meaning': '의문', 'split_after': True},
    
    # 현대어 연결 표지도 포함
    '므로': {'type': 'connective', 'meaning': '인과', 'split_after': True},
    '서': {'type': 'connective', 'meaning': '인과', 'split_after': True},
    '면서': {'type': 'connective', 'meaning': '동시', 'split_after': True},
    '면': {'type': 'connective', 'meaning': '조건', 'split_after': False},
    '야': {'type': 'connective', 'meaning': '연속', 'split_after': False},
    
    # 조사 (한문 뒤에 붙는 경우)
    '은': {'type': 'particle', 'meaning': '주제/대조', 'split_after': False},
    '는': {'type': 'particle', 'meaning': '주제/대조', 'split_after': False},
    '이': {'type': 'particle', 'meaning': '주격', 'split_after': False},
    '가': {'type': 'particle', 'meaning': '주격', 'split_after': False},
    '을': {'type': 'particle', 'meaning': '목적격', 'split_after': False},
    '를': {'type': 'particle', 'meaning': '목적격', 'split_after': False},
    '도': {'type': 'particle', 'meaning': '보조사', 'split_after': False},
    '만': {'type': 'particle', 'meaning': '보조사', 'split_after': False},
}

# [REMOVED] 중세국어 어미 처리 함수들 - 더 이상 사용되지 않음
# - detect_middle_korean_ending
# - split_hanja_with_korean_ending
# - enhance_src_split_with_korean_endings

def split_src_meaning_units(
    text: str,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    by_space: bool = False,
    **kwargs
):
    """원문(한문+한글)을 jieba와 MeCab으로 의미 단위 분할"""
    
    # 무결성 검증: 입력 텍스트 보존
    if not text or not text.strip():
        return []
    
    original_text = text
    
    # 1단계: 공백 정규화 및 전각 콜론 처리 (무결성 보장)
    # 연속 공백을 단일 공백로 정규화하고, 전각 콜론 뒤에 공백 추가
    normalized_text = re.sub(r'\s+', ' ', text.replace('\n', ' ')).strip()
    normalized_text = normalized_text.replace('：', '： ')
    
    words = normalized_text.split()
    if not words:
        return []
    
    # 2단계: jieba와 MeCab 분석 결과 준비 (원본 텍스트 기준)
    jieba_tokens = list(jieba.cut(original_text))
    
    # MeCab 분석 (한글 부분용)
    morpheme_info = []
    if mecab:
        result = mecab.parse(original_text)
        for line in result.split('\n'):
            if line and line != 'EOS':
                parts = line.split('\t')
                if len(parts) >= 2:
                    surface = parts[0]
                    pos = parts[1].split(',')[0]
                    morpheme_info.append((surface, pos))
    
    # 3단계: 어절들을 의미 단위로 그룹화 (jieba + MeCab 정보 적극 활용)
    units = []
    i = 0
    
    while i < len(words):
        word = words[i]
        
        # 전각 콜론 처리 - 하드 경계
        if word.endswith('：'):
            units.append(word)
            i += 1
            continue
        
        # 한자 포함 어절 처리
        if hanja_re.search(word):
            # 현재 어절이 한자를 포함하면 하나의 의미 단위
            units.append(word)
            i += 1
            continue
        
        # 한글 어절들 처리 - jieba와 MeCab 분석 결과 적극 활용
        if hangul_re.match(word):
            group = [word]
            j = i + 1
            
            # 중세국어 어미나 문법 표지로 경계 판단 (원문용)
            should_break_here = _should_break_by_mecab_src(word, morpheme_info) if morpheme_info else False
            
            # jieba 토큰 연속성도 적극 확인 (경계 신호가 없는 경우만)
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
    
    # 무결성 검증: 토큰 순서 및 내용 보존 확인
    reconstructed = ' '.join(units).replace(' ', '')
    original_clean = original_text.replace(' ', '').replace('\n', '')
    
    if reconstructed != original_clean:
        logger.warning(f"원문 분할 무결성 경고: 원본과 복원 결과 불일치")
        logger.warning(f"원본: {original_clean[:100]}...")
        logger.warning(f"복원: {reconstructed[:100]}...")
    
    return units

def _should_group_words_by_jieba(word_group: List[str], jieba_tokens: List[str]) -> bool:
    """jieba 분석 결과를 적극 참고해서 어절들을 묶을지 결정 (강화 버전)"""
    combined = ''.join(word_group)
    
    # 1. jieba 토큰과 정확히 일치하는 경우 (우선순위 1)
    for token in jieba_tokens:
        clean_token = token.replace(' ', '').replace('\n', '')
        clean_combined = combined.replace(' ', '')
        if clean_token == clean_combined and len(clean_token) > 1:
            return True
    
    # 2. jieba 토큰에 포함되는 부분 문자열인 경우 (우선순위 2)
    for token in jieba_tokens:
        clean_token = token.replace(' ', '').replace('\n', '')
        clean_combined = combined.replace(' ', '')
        if len(clean_combined) > 1 and clean_combined in clean_token:
            return True
    
    # 3. 길이 제한 및 기본 휴리스틱
    if len(combined) > 15:  # 너무 긴 조합 방지
        return False
    
    if len(word_group) > 4:  # 너무 많은 어절 조합 방지
        return False
    
    # 4. 단음절 어절들의 조합은 신중하게 (길이 3 이하)
    if all(len(word) <= 1 for word in word_group) and len(word_group) > 2:
        return False
    
    return len(word_group) <= 3

def split_inside_chunk(chunk: str) -> List[str]:
    """번역문 청크를 경계 표지 기반으로 분할 (전각 콜론 우선 처리, 무결성 보장)"""
    
    if not chunk or not chunk.strip():
        return []
    
    original_chunk = chunk
    
    # 1단계: 전각 콜론 우선 분할 (재귀적 처리)
    if '：' in chunk:
        colon_parts = chunk.split('：')
        if len(colon_parts) == 2:
            part1 = colon_parts[0].strip() + '：'
            part2 = colon_parts[1].strip()
            result = []
            if part1.strip():
                result.append(part1)
            if part2.strip():
                result.extend(split_inside_chunk(part2))  # 재귀적으로 처리
            
            # 무결성 검증
            reconstructed = ''.join(result).replace(' ', '')
            original_clean = original_chunk.replace(' ', '')
            if reconstructed != original_clean:
                logger.warning(f"청크 분할 무결성 경고: {original_chunk} -> {result}")
                return [original_chunk]  # 실패 시 원본 반환
            
            return result
    
    # 2단계: 공백 정규화 (연속 공백 제거)
    normalized_chunk = re.sub(r'\s+', ' ', chunk.strip())
    words = normalized_chunk.split()
    if not words:
        return []
    
    # 3단계: 경계 표지 기반 분할
    units = []
    current_group = []
    
    for word in words:
        current_group.append(word)
        
        # 경계 표지 확인 (문법적 표지 + 콤마) - 번역문 처리
        if is_boundary_marker(word, is_source=False):
            units.append(' '.join(current_group))
            current_group = []
            continue
    
    # 마지막 그룹 처리
    if current_group:
        units.append(' '.join(current_group))
    
    # 4단계: "라고/라는" 표현 후처리 - 앞 구에 병합
    if len(units) > 1:
        merged_units = []
        i = 0
        
        while i < len(units):
            current_unit = units[i]
            
            # 다음 단위가 "라고/라는" 계열로 시작하면 현재 단위에 병합
            if i + 1 < len(units) and should_attach_to_previous(units[i + 1]):
                merged_unit = current_unit + ' ' + units[i + 1]
                merged_units.append(merged_unit.strip())
                i += 2  # 두 단위를 모두 처리했으므로 2 증가
            else:
                merged_units.append(current_unit.strip())
                i += 1
        
        units = merged_units
    
    result = [unit.strip() for unit in units if unit.strip()]
    
    # 무결성 검증: 전체 내용 보존 확인
    reconstructed = ''.join(result).replace(' ', '')
    original_clean = original_chunk.replace(' ', '')
    
    if reconstructed != original_clean:
        logger.warning(f"청크 분할 무결성 경고: 내용 불일치")
        logger.warning(f"원본: {original_clean}")
        logger.warning(f"복원: {reconstructed}")
        return [original_chunk]  # 실패 시 원본 반환
    
    return result if result else [original_chunk]

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

# [REMOVED] find_target_span_end_semantic - 더 이상 사용되지 않음

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
    embed_func: Callable = None, 
    min_tokens: int = DEFAULT_MIN_TOKENS
) -> List[str]:
    """원문 단위에 따른 번역문 분할 - 전각 콜론 우선 처리"""
    
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
    
    # 2단계: 순차 방식으로 분할
    return split_tgt_meaning_units_sequential(
        ' '.join(src_units), tgt_text, min_tokens, min_tokens * 3
    )

# [REMOVED] Complex semantic matching functions - 더 이상 사용되지 않음
# - _find_optimal_semantic_matching
# - _calculate_keyword_bonus  
# - _calculate_structure_bonus
# - _calculate_length_balance_bonus
# - _dp_semantic_matching

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

# [REMOVED] 임베딩 정규화 함수들 - 더 이상 사용되지 않음
# - normalize_for_embedding  
# - _normalize_for_embedding

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

# [REMOVED] 사용되지 않는 함수들 - 더 이상 필요하지 않음
# - _split_single_target_to_multiple
# - _merge_splits_to_match_src_count  
# - _force_split_by_semantic_boundaries

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
    
    # 2. '호되' 특별 처리 - 인용 표지로 강력한 분할 신호
    if word.endswith('호되'):
        return True  # 다음 어절부터 인용문이므로 확실한 경계
    
    # 3. 일반적인 MeCab 분석 결과 확인 (번역문과 동일)
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
    embed_func: Callable = None,
    use_grammar: bool = True,  # 🆕 문법적 표지 사용 옵션
    **kwargs
) -> List[str]:
    """의미 기반 순차적 분할 - 임베딩과 콤마 분할, 문법적 표지를 적극 활용 (무결성 보장)"""
    
    # 무결성 검증: 입력 검사
    if not tgt_text or not tgt_text.strip():
        return []
    
    original_tgt = tgt_text
    
    # 전각 콜론을 포함한 경우 우선 처리 (하드 경계)
    if '：' in tgt_text:
        colon_parts = tgt_text.split('：')
        if len(colon_parts) == 2:
            part1 = colon_parts[0].strip() + '：'
            part2 = colon_parts[1].strip()
            result = []
            if part1.strip():
                result.append(part1)
            if part2.strip():
                # 나머지 부분을 재귀적으로 처리
                remaining_parts = split_tgt_meaning_units_sequential(
                    src_text, part2, min_tokens, max_tokens, embed_func, use_grammar, **kwargs
                )
                result.extend(remaining_parts)
            
            # 무결성 검증
            reconstructed = ''.join(result).replace(' ', '')
            original_clean = original_tgt.replace(' ', '')
            if reconstructed != original_clean:
                logger.warning(f"타겟 분할 무결성 경고 (콜론): 내용 불일치")
                return [original_tgt]
            
            return result
    
    # 임베딩 함수 준비
    if embed_func is None:
        try:
            from sa_embedders import compute_embeddings_with_cache
            embed_func = compute_embeddings_with_cache
        except ImportError:
            # 임베딩 없으면 기본 순차 분할
            return _fallback_sequential_split(src_text, tgt_text, min_tokens, max_tokens, use_grammar)
    
    # 1단계: 원문 의미 단위 분할
    src_units = split_src_meaning_units(src_text, min_tokens, max_tokens)
    
    # 2단계: 번역문을 자연스럽게 분할 (콤마 우선, MeCab 참고)
    tgt_chunks = split_inside_chunk(tgt_text)
    
    # 3단계: 의미 기반 순차 매칭 (간결화)
    result = _semantic_sequential_matching(src_units, tgt_chunks, embed_func)
    
    # 무결성 검증: 최종 결과 확인
    reconstructed = ''.join(result).replace(' ', '')
    original_clean = original_tgt.replace(' ', '')
    
    if reconstructed != original_clean:
        logger.warning(f"순차 분할 무결성 경고: 내용 불일치")
        logger.warning(f"원본: {original_clean[:100]}...")
        logger.warning(f"복원: {reconstructed[:100]}...")
        # 실패 시 폴백 분할 시도
        return _fallback_sequential_split(src_text, tgt_text, min_tokens, max_tokens, use_grammar)
    
    return result

def _semantic_sequential_matching(
    src_units: List[str], 
    tgt_chunks: List[str], 
    embed_func: Callable
) -> List[str]:
    """의미 기반 순차 매칭 - 순서 보장하면서 의미적으로 최적화"""
    
    if not src_units or not tgt_chunks:
        return tgt_chunks if tgt_chunks else [' '.join(src_units)]
    
    # 임베딩 계산
    try:
        src_embeddings = embed_func(src_units)
        tgt_embeddings = embed_func(tgt_chunks)
        
        # 기본 유사도 행렬 계산
        embed_similarity = _calculate_similarity_matrix(src_embeddings, tgt_embeddings)
        
        # 강화된 유사도 행렬 (키워드 + 구조 정보 결합)
        enhanced_similarity = _enhanced_similarity_matrix(src_units, tgt_chunks, embed_similarity)
        
        # 구조적 보너스 추가
        structure_bonus = _calculate_structure_bonus(src_units, tgt_chunks)
        
        # 순차적 최적 매칭 (강화된 유사도 사용)
        return _optimal_sequential_assignment(
            src_units, tgt_chunks, enhanced_similarity, structure_bonus
        )
        
    except Exception as e:
        logger.warning(f"임베딩 기반 매칭 실패, 폴백: {e}")
        return _fallback_sequential_split(' '.join(src_units), ' '.join(tgt_chunks), 1, 50)

def _calculate_similarity_matrix(src_embeddings, tgt_embeddings):
    """유사도 행렬 계산"""
    import numpy as np
    
    # 코사인 유사도 계산
    src_norm = src_embeddings / (np.linalg.norm(src_embeddings, axis=1, keepdims=True) + 1e-8)
    tgt_norm = tgt_embeddings / (np.linalg.norm(tgt_embeddings, axis=1, keepdims=True) + 1e-8)
    
    similarity = np.dot(src_norm, tgt_norm.T)
    return similarity

def _optimal_sequential_assignment(
    src_units: List[str], 
    tgt_chunks: List[str], 
    similarity_matrix,
    structure_bonus: float = 0.0
) -> List[str]:
    """순차적 최적 할당 - 순서 보장하면서 의미 유사도 최대화"""
    
    n_src = len(src_units)
    n_tgt = len(tgt_chunks)
    
    # 1:1 매칭인 경우
    if n_src == n_tgt:
        return tgt_chunks
    
    # 원문이 더 적은 경우 - 번역문 청크 병합 (구조 보너스 포함)
    if n_src < n_tgt:
        return _merge_chunks_by_similarity(tgt_chunks, similarity_matrix, n_src, structure_bonus)
    
    # 원문이 더 많은 경우 - 번역문 청크 분할 (구조 보너스 포함)
    else:
        return _split_chunks_by_similarity(src_units, tgt_chunks, similarity_matrix, n_src, structure_bonus)

def _merge_chunks_by_similarity(
    tgt_chunks: List[str], 
    similarity_matrix, 
    target_count: int,
    structure_bonus: float = 0.0
) -> List[str]:
    """유사도 기반 청크 병합 - 순서 보장"""
    
    if len(tgt_chunks) <= target_count:
        return tgt_chunks
    
    import numpy as np
    
    # 현재 청크들
    current_chunks = tgt_chunks[:]
    
    # target_count개가 될 때까지 병합
    while len(current_chunks) > target_count:
        
        # 인접한 청크들 중 병합 점수가 가장 높은 쌍 찾기
        best_merge_idx = -1
        best_score = -1
        
        for i in range(len(current_chunks) - 1):
            # 병합 점수 계산 (유사도 + 길이 균형 + 콤마 분할 보존 + 구조 보너스)
            merge_score = _calculate_merge_score(
                current_chunks[i], 
                current_chunks[i + 1],
                i, 
                similarity_matrix,
                len(current_chunks),
                structure_bonus
            )
            
            if merge_score > best_score:
                best_score = merge_score
                best_merge_idx = i
        
        # 최적 쌍 병합
        if best_merge_idx >= 0:
            merged_chunk = current_chunks[best_merge_idx] + ' ' + current_chunks[best_merge_idx + 1]
            current_chunks = (current_chunks[:best_merge_idx] + 
                            [merged_chunk] + 
                            current_chunks[best_merge_idx + 2:])
        else:
            # 더 이상 병합할 수 없으면 마지막 두 청크 병합
            if len(current_chunks) >= 2:
                merged = current_chunks[-2] + ' ' + current_chunks[-1]
                current_chunks = current_chunks[:-2] + [merged]
            else:
                break
    
    return current_chunks

def _calculate_merge_score(
    chunk1: str, 
    chunk2: str, 
    position: int, 
    similarity_matrix, 
    total_chunks: int,
    structure_bonus: float = 0.0
) -> float:
    """청크 병합 점수 계산 - 문법적 표지 고려"""
    
    # 기본 점수 (길이 균형)
    len1, len2 = len(chunk1), len(chunk2)
    length_balance = 1.0 - abs(len1 - len2) / max(len1 + len2, 1)
    
    # 콤마 분할 보존 점수 (콤마가 있는 청크는 병합하지 않음)
    comma_penalty = 0.0
    if (',' in chunk1 and chunk1.endswith(',')) or ('，' in chunk1 and chunk1.endswith('，')):
        comma_penalty = -0.5  # 콤마로 끝나는 청크는 병합 페널티
    if (',' in chunk2 and chunk2.endswith(',')) or ('，' in chunk2 and chunk2.endswith('，')):
        comma_penalty = -0.5
    
    # 🆕 문법적 표지 보존 점수
    grammar_bonus = _calculate_grammar_preservation_score(chunk1, chunk2)
    
    # 의미적 유사도 (임베딩 기반) - 강화된 계산
    semantic_bonus = 0.0
    try:
        if similarity_matrix is not None and position < similarity_matrix.shape[0] - 1:
            # 인접한 원문 단위들과의 유사도 평균 (더 정교한 계산)
            if position < similarity_matrix.shape[1] - 1:
                semantic_bonus = float(similarity_matrix[position, position]) * 0.4
                # 교차 유사도도 고려 (의미적 연관성)
                cross_similarity = float(similarity_matrix[position, position + 1]) * 0.2
                semantic_bonus += cross_similarity
    except:
        pass
    
    # 위치 기반 보너스 (앞쪽 청크들 우선 병합)
    position_bonus = (total_chunks - position) / total_chunks * 0.2
    
    # 구조적 보너스 적용
    final_structure_bonus = structure_bonus * 0.1  # 전체 점수의 10%
    
    return length_balance + semantic_bonus + position_bonus + comma_penalty + grammar_bonus + final_structure_bonus

def _calculate_grammar_preservation_score(chunk1: str, chunk2: str) -> float:
    """문법적 표지 보존 점수 계산"""
    
    # 길이 균형 점수 (간결화)
    len1, len2 = len(chunk1), len(chunk2)
    length_balance = 1.0 - abs(len1 - len2) / max(len1 + len2, 1)
    
    # 콤마 분할 보존 점수 (경계 표지 활용)
    comma_penalty = 0.0
    if is_boundary_marker(chunk1, is_source=False) or is_boundary_marker(chunk2, is_source=False):
        comma_penalty = -0.5  # 경계 표지가 있는 청크는 병합 페널티
    
    # 경계 강도 기반 점수
    boundary_bonus = (get_boundary_strength(chunk1, is_source=False) + get_boundary_strength(chunk2, is_source=False)) * 0.2
    
    return length_balance + comma_penalty + boundary_bonus

def _split_chunks_by_similarity(
    src_units: List[str],
    tgt_chunks: List[str], 
    similarity_matrix,
    target_count: int,
    structure_bonus: float = 0.0
) -> List[str]:
    """유사도 기반 청크 분할"""
    
    if len(tgt_chunks) >= target_count:
        return tgt_chunks[:target_count]  # 잘라서 반환
    
    # 가장 긴 청크를 의미적으로 분할
    result_chunks = tgt_chunks[:]
    
    while len(result_chunks) < target_count:
        # 가장 긴 청크 찾기
        longest_idx = max(range(len(result_chunks)), key=lambda i: len(result_chunks[i]))
        longest_chunk = result_chunks[longest_idx]
        
        # MeCab 기반으로 분할 시도
        split_attempts = split_inside_chunk(longest_chunk)
        
        if len(split_attempts) > 1:
            # 분할 성공
            result_chunks = (result_chunks[:longest_idx] + 
                           split_attempts + 
                           result_chunks[longest_idx + 1:])
        else:
            # 분할 실패 시 단순 토큰 분할
            tokens = longest_chunk.split()
            if len(tokens) >= 2:
                mid = len(tokens) // 2
                part1 = ' '.join(tokens[:mid])
                part2 = ' '.join(tokens[mid:])
                result_chunks = (result_chunks[:longest_idx] + 
                               [part1, part2] + 
                               result_chunks[longest_idx + 1:])
            else:
                break  # 더 이상 분할할 수 없음
    
    return result_chunks[:target_count]

def _fallback_sequential_split(
    src_text: str, 
    tgt_text: str, 
    min_tokens: int, 
    max_tokens: int,
    use_grammar: bool = True
) -> List[str]:
    """임베딩 없을 때 폴백 분할 - 문법적 표지 고려"""
    
    # 원문 분할
    src_units = split_src_meaning_units(src_text, min_tokens, max_tokens)
    
    # 번역문 콤마 기반 분할
    tgt_chunks = split_inside_chunk(tgt_text)
    # 간단한 길이 기반 조정 (간결화)
    if len(src_units) == len(tgt_chunks):
        return tgt_chunks
    elif len(src_units) == 1:
        # 콤마 분할 보존 (경계 표지 기반)
        if len(tgt_chunks) > 1 and any(is_boundary_marker(c, is_source=False) for c in tgt_chunks):
            return tgt_chunks
        else:
            return [tgt_text]
    elif len(tgt_chunks) > len(src_units):
        # 간단한 병합
        return _simple_merge_chunks(tgt_chunks, len(src_units))
    else:
        # 간단한 분할
        return _simple_split_by_tokens(tgt_text, len(src_units))

def _simple_merge_chunks(chunks: List[str], target_count: int) -> List[str]:
    """간단한 청크 병합"""
    if len(chunks) <= target_count:
        return chunks
    
    result = chunks[:]
    while len(result) > target_count:
        # 가장 짧은 인접 쌍 병합
        min_len = float('inf')
        merge_idx = 0
        
        for i in range(len(result) - 1):
            combined_len = len(result[i]) + len(result[i + 1])
            # 콤마로 끝나는 청크는 병합 우선순위 낮춤
            if result[i].endswith(',') or result[i].endswith('，'):
                combined_len += 100  # 페널티
            
            if combined_len < min_len:
                min_len = combined_len
                merge_idx = i
        
        # 병합 실행
        merged = result[merge_idx] + ' ' + result[merge_idx + 1]
        result = result[:merge_idx] + [merged] + result[merge_idx + 2:]
    
    return result

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
    """기존 함수 호환성 유지"""
    return _simple_merge_chunks(chunks, target_count)

# 문법적 경계 표지와 콤마를 간결하게 인식하는 함수 추가
def is_boundary_marker(text: str, is_source: bool = False) -> bool:
    """문법적 경계 표지 또는 콤마 경계인지 확인 (원문/번역문 구분, 발화동사 탐지 통합)"""
    
    # 1. 문법적 표지 확인 (기존 로직)
    for marker, functions in BOUNDARY_MARKERS.items():
        if text.endswith(marker):
            # 원문 전용 표지인지 확인
            if 'src_only' in functions:
                if is_source:
                    # '호되'는 원문에서 표점 기능 - 강력한 분할 신호
                    if marker == '호되':
                        return True  # 인용 경계로 표점 기능
                    return True
                else:
                    # 번역문에서는 원문 전용 표지 무시
                    continue
            else:
                # 일반 표지 (번역문에서 사용)
                return True
    
    # 2. 🆕 발화동사 기반 인용 경계 탐지 (번역문에서만)
    if not is_source and is_quotative_end_pattern(text):
        return True
    
    # 3. 콤마 경계 확인 (번역문에서만, 병렬 제외)
    if not is_source:
        import re
        for pattern in COMMA_BOUNDARY_PATTERNS:
            if re.search(pattern, text):
                return True
        
        # 4. 단순 콤마 (병렬이 아닌 경우)
        if text.endswith(',') or text.endswith('，'):
            # 병렬 제외 로직 (간단한 휴리스틱)
            if not any(conj in text for conj in ['과', '와', '및', '또는', '이나']):
                return True
    
    return False

def get_boundary_strength(text: str, is_source: bool = False) -> float:
    """경계 강도 계산 (0.0-1.0) - 원문/번역문 구분, '호되' 특별 처리, 발화동사 고려"""
    
    # 1. 원문 전용 표지 처리
    if is_source:
        # '호되'는 원문에서 표점 기능 - 매우 강한 경계
        if text.endswith('호되'):
            return 0.95  # 인용 표지로 표점 기능, 거의 절대적 분할점
        
        # 기타 중세국어 어미들
        for marker, functions in BOUNDARY_MARKERS.items():
            if 'src_only' in functions and text.endswith(marker):
                return 0.8  # 원문 전용 어미들은 강한 경계
    
    # 2. 전각 콜론 (원문/번역문 공통)
    if text.endswith('：'):
        return 0.95  # '호되'와 동등한 강도
    
    # 3. 🆕 발화동사 기반 인용 경계 (번역문에서만)
    if not is_source and is_quotative_end_pattern(text):
        return 0.85  # 발화동사 뒤는 강한 분할점 (인용문 경계)
    
    # 4. 일반 문법적 표지 (번역문에서 주로 사용)
    for marker, functions in BOUNDARY_MARKERS.items():
        if 'src_only' not in functions and text.endswith(marker):
            if 'boundary' in functions:
                return 0.8  # 일반적인 경계 표지
            return 0.6
    
    # 5. 콤마 경계 (번역문에서만)
    if not is_source and (text.endswith(',') or text.endswith('，')):
        return 0.5
    
    return 0.0

def _calculate_keyword_similarity(src_unit: str, tgt_chunk: str) -> float:
    """키워드 기반 의미 유사도 계산 - 한문-한글 매칭 강화 (괄호 한자 포함)"""
    
    # 🆕 '호되'-콜론 특별 매칭 (우선순위 1)
    if src_unit.endswith('호되') and tgt_chunk.endswith('：'):
        # 호되로 끝나는 원문과 콜론으로 끝나는 번역문은 강한 유사성
        base_score = 0.85  # 높은 기본 점수
        
        # 나머지 부분의 한자 매칭도 확인
        src_without_ending = src_unit[:-2]  # '호되' 제거
        tgt_without_colon = tgt_chunk[:-1]  # '：' 제거
        
        if src_without_ending and tgt_without_colon:
            content_similarity = _calculate_content_similarity(src_without_ending, tgt_without_colon)
            return min(base_score + (content_similarity * 0.15), 1.0)
        else:
            return base_score
    
    # 🆕 역방향 매칭: 콜론으로 끝나는 원문과 호되로 끝나는 번역문
    elif src_unit.endswith('：') and tgt_chunk.endswith('호되'):
        base_score = 0.85
        src_without_colon = src_unit[:-1]
        tgt_without_ending = tgt_chunk[:-2]
        
        if src_without_colon and tgt_without_ending:
            content_similarity = _calculate_content_similarity(src_without_colon, tgt_without_ending)
            return min(base_score + (content_similarity * 0.15), 1.0)
        else:
            return base_score
    
    # 한자 키워드 추출 (원문에서)
    src_hanja = regex.findall(r'\p{Han}+', src_unit)
    
    if not src_hanja:
        return 0.0
    
    # 🆕 번역문에서 한자 추출 - 괄호 안 한자 우선 고려
    # 1. 괄호 안의 한자 추출 (가장 중요)
    bracket_hanja = regex.findall(r'[（(]\s*(\p{Han}+)\s*[）)]', tgt_chunk)
    
    # 2. 일반 텍스트에서 한자 추출 (보조적)
    direct_hanja = regex.findall(r'\p{Han}+', tgt_chunk)
    
    # 괄호 안 한자를 우선적으로 고려하되, 일반 한자도 포함
    all_tgt_hanja = bracket_hanja + direct_hanja
    
    # 키워드 매칭 점수 계산 (괄호 한자에 가중치 부여)
    keyword_matches = 0.0
    total_keywords = len(src_hanja)
    
    for src_word in src_hanja:
        best_match_score = 0.0
        
        # 1. 괄호 안 한자와의 완전 일치 (최고 점수 + 보너스)
        if src_word in bracket_hanja:
            best_match_score = 1.2  # 괄호 한자 완전 일치 보너스
        
        # 2. 일반 텍스트에서의 완전 일치
        elif src_word in tgt_chunk:
            best_match_score = 1.0
            
        # 3. 괄호 한자와의 부분 일치 (높은 점수)
        else:
            for bracket_word in bracket_hanja:
                if src_word in bracket_word or bracket_word in src_word:
                    partial_score = len(set(src_word) & set(bracket_word)) / len(set(src_word) | set(bracket_word))
                    best_match_score = max(best_match_score, partial_score * 1.0)  # 괄호 내 부분 일치 보너스
            
            # 4. 일반 한자와의 부분 일치 (중간 점수)
            if best_match_score == 0.0:
                for tgt_word in direct_hanja:
                    if src_word in tgt_word or tgt_word in src_word:
                        partial_score = len(set(src_word) & set(tgt_word)) / len(set(src_word) | set(tgt_word))
                        best_match_score = max(best_match_score, partial_score * 0.8)
            
            # 5. 개별 한자 매칭 (가장 낮은 점수)
            if best_match_score == 0.0:
                char_matches = sum(1 for char in src_word if char in tgt_chunk)
                if char_matches > 0:
                    best_match_score = (char_matches / len(src_word)) * 0.3
        
        keyword_matches += min(best_match_score, 1.0)  # 최대 1.0으로 제한
    
    # 키워드 매칭 비율 (괄호 한자 보너스 반영)
    keyword_ratio = keyword_matches / max(total_keywords, 1)
    
    # 🆕 괄호 한자 존재 시 추가 보너스
    bracket_bonus = 0.0
    if bracket_hanja and any(src_word in bracket_hanja for src_word in src_hanja):
        # 원문 한자와 괄호 한자의 일치 비율에 따른 보너스
        matching_bracket_count = sum(1 for src_word in src_hanja if src_word in bracket_hanja)
        bracket_bonus = (matching_bracket_count / len(src_hanja)) * 0.2
    
    # 길이 기반 보정 (개선된 공식)
    src_len = len(src_unit.replace(' ', ''))
    tgt_len = len(tgt_chunk.replace(' ', ''))
    
    if src_len > 0 and tgt_len > 0:
        length_ratio = min(tgt_len / src_len, src_len / tgt_len)
        # 적정 길이 비율 (0.5 ~ 2.0)에서 최고 점수
        if 0.5 <= length_ratio <= 2.0:
            length_factor = 1.0
        else:
            length_factor = max(0.3, length_ratio if length_ratio < 0.5 else 1.0 / length_ratio)
    else:
        length_factor = 0.1
    
    # 구두점 일치 보너스
    punctuation_bonus = 0.0
    src_punct = set(char for char in src_unit if char in '，。；！？：')
    tgt_punct = set(char for char in tgt_chunk if char in '，。；！？：,.')
    if src_punct and tgt_punct:
        punctuation_bonus = len(src_punct & tgt_punct) / max(len(src_punct | tgt_punct), 1) * 0.1
    
    final_score = keyword_ratio * length_factor + bracket_bonus + punctuation_bonus
    return min(final_score, 1.0)

def _calculate_content_similarity(src_content: str, tgt_content: str) -> float:
    """'호되'와 '：'를 제외한 나머지 내용의 유사도 계산 - 괄호 한자 고려"""
    
    # 한자 키워드 추출 (원문)
    src_hanja = regex.findall(r'\p{Han}+', src_content)
    
    if not src_hanja:
        return 0.0
    
    # 🆕 번역문에서 괄호 한자 우선 추출
    bracket_hanja = regex.findall(r'[（(]\s*(\p{Han}+)\s*[）)]', tgt_content)
    direct_hanja = regex.findall(r'\p{Han}+', tgt_content)
    
    # 키워드 매칭 점수 (괄호 한자에 가중치)
    keyword_matches = 0.0
    total_keywords = len(src_hanja)
    
    for src_word in src_hanja:
        # 1. 괄호 안 한자와의 완전 일치 (보너스 점수)
        if src_word in bracket_hanja:
            keyword_matches += 1.2
        # 2. 일반 텍스트에서의 완전 일치
        elif src_word in tgt_content:
            keyword_matches += 1.0
        else:
            # 3. 부분 매칭 (개별 한자)
            char_matches = sum(1 for char in src_word if char in tgt_content)
            if char_matches > 0:
                keyword_matches += (char_matches / len(src_word)) * 0.5
    
    # 결과를 [0, 1] 범위로 정규화
    return min(keyword_matches / max(total_keywords, 1), 1.0)

def _enhanced_similarity_matrix(src_units: List[str], tgt_chunks: List[str], embed_similarity) -> np.ndarray:
    """임베딩 유사도와 키워드 유사도를 결합한 강화된 유사도 행렬"""
    
    n_src = len(src_units)
    n_tgt = len(tgt_chunks)
    
    # 키워드 기반 유사도 계산
    keyword_similarity = np.zeros((n_src, n_tgt))
    
    for i, src_unit in enumerate(src_units):
        for j, tgt_chunk in enumerate(tgt_chunks):
            keyword_similarity[i, j] = _calculate_keyword_similarity(src_unit, tgt_chunk)
    
    # 임베딩 유사도와 키워드 유사도 결합 (개선된 가중치)
    # 한문-한글 번역에서는 키워드 매칭이 더 중요할 수 있음
    combined_similarity = (
        embed_similarity * 0.55 +      # 임베딩 유사도 55%
        keyword_similarity * 0.45      # 키워드 유사도 45% (증가)
    )
    
    return combined_similarity

def _calculate_structure_bonus(src_units: List[str], tgt_chunks: List[str]) -> float:
    """구조적 일치도 보너스 계산 (개선 버전 - '호되'-콜론 매칭 고려)"""
    
    total_bonus = 0.0
    
    # 🆕 '호되'-콜론 구조적 매칭 보너스 (우선순위 1)
    hodeok_colon_bonus = 0.0
    for i, src_unit in enumerate(src_units):
        if src_unit.endswith('호되'):
            # 같은 위치나 인접 위치의 타겟에서 콜론 찾기
            for j in range(max(0, i-1), min(len(tgt_chunks), i+2)):
                if j < len(tgt_chunks) and tgt_chunks[j].endswith('：'):
                    position_match = 1.0 - abs(i - j) / max(len(src_units), len(tgt_chunks), 1)
                    hodeok_colon_bonus = max(hodeok_colon_bonus, position_match * 0.6)
                    break
    
    total_bonus += hodeok_colon_bonus
    
    # 1. 전각 콜론 위치 일치도 (기존 로직, 가중치 조정)
    src_colon_positions = [i for i, unit in enumerate(src_units) if '：' in unit]
    tgt_colon_positions = [i for i, chunk in enumerate(tgt_chunks) if '：' in chunk]
    
    if src_colon_positions and tgt_colon_positions:
        # 첫 번째 콜론 위치 비율 비교
        src_ratio = src_colon_positions[0] / max(len(src_units) - 1, 1)
        tgt_ratio = tgt_colon_positions[0] / max(len(tgt_chunks) - 1, 1)
        position_similarity = 1.0 - abs(src_ratio - tgt_ratio)
        total_bonus += position_similarity * 0.3  # 호되-콜론 매칭이 있으면 가중치 감소
    
    # 2. 단위 수 일치도
    count_similarity = 1.0 - abs(len(src_units) - len(tgt_chunks)) / max(len(src_units), len(tgt_chunks), 1)
    total_bonus += count_similarity * 0.2
    
    # 3. 길이 분포 일치도 (개선된 계산)
    if len(src_units) > 1 and len(tgt_chunks) > 1:
        src_lengths = [len(unit.replace(' ', '')) for unit in src_units]
        tgt_lengths = [len(chunk.replace(' ', '')) for chunk in tgt_chunks]
        
        # 같은 길이인 경우 상관관계 계산
        if len(src_lengths) == len(tgt_lengths):
            try:
                correlation = np.corrcoef(src_lengths, tgt_lengths)[0, 1]
                if not np.isnan(correlation):
                    length_bonus = max(correlation, 0) * 0.25
                else:
                    length_bonus = 0.0
            except:
                length_bonus = 0.0
        else:
            # 길이가 다른 경우 분포의 유사성 계산
            src_avg = np.mean(src_lengths) if src_lengths else 0
            tgt_avg = np.mean(tgt_lengths) if tgt_lengths else 0
            avg_similarity = 1.0 - abs(src_avg - tgt_avg) / max(src_avg + tgt_avg, 1)
            length_bonus = avg_similarity * 0.15
        
        total_bonus += length_bonus
    
    # 4. 구두점 패턴 일치도
    src_punct_pattern = ''.join([char for unit in src_units for char in unit if char in '，。；！？'])
    tgt_punct_pattern = ''.join([char for chunk in tgt_chunks for char in chunk if char in '，。；！？：,.'])
    
    if src_punct_pattern and tgt_punct_pattern:
        # 구두점 순서와 종류의 일치도
        punct_similarity = len(set(src_punct_pattern) & set(tgt_punct_pattern)) / len(set(src_punct_pattern) | set(tgt_punct_pattern))
        total_bonus += punct_similarity * 0.15
    
    return min(total_bonus, 1.0)

def _smart_colon_split(src_text: str, tgt_text: str) -> tuple:
    """전각 콜론 기반 스마트 분할 - 의미 대응 고려 ('호되' 특별 처리)"""
    
    # 🆕 '호되'로 끝나는 원문과 콜론이 있는 번역문 매칭
    if src_text.endswith('호되') and '：' in tgt_text:
        tgt_parts = tgt_text.split('：')
        if len(tgt_parts) == 2:
            # '호되'로 끝나는 원문은 콜론 앞부분과 매칭
            src_parts = [src_text, '']  # 호되는 첫 번째 부분으로 처리
            tgt_result = [tgt_parts[0] + '：', tgt_parts[1].strip()]
            return ([src_text], tgt_result)
    
    # 🆕 콜론이 있는 원문과 '호되'로 끝나는 번역문 매칭 (역방향)
    elif '：' in src_text and tgt_text.endswith('호되'):
        src_parts = src_text.split('：')
        if len(src_parts) == 2:
            src_result = [src_parts[0] + '：', src_parts[1].strip()]
            return (src_result, [tgt_text])
    
    # 기존 로직 계속 실행
    if '：' not in src_text and '：' not in tgt_text:
        return None, None
    
    # 소스에만 콜론이 있는 경우
    if '：' in src_text and '：' not in tgt_text:
        src_parts = src_text.split('：')
        if len(src_parts) == 2:
            # 번역문에서 해당 위치 추정
            src_ratio = len(src_parts[0]) / len(src_text)
            split_point = int(len(tgt_text) * src_ratio)
            
            # 어절 경계에서 분할점 조정
            words = tgt_text[:split_point + 20].split()  # 여유분 포함
            if len(words) > 1:
                adjusted_split = len(' '.join(words[:-1])) + 1
                part1 = tgt_text[:adjusted_split].strip() + '：'
                part2 = tgt_text[adjusted_split:].strip()
                return (src_parts[0] + '：', src_parts[1]), (part1, part2)
    
    # 타겟에만 콜론이 있는 경우
    elif '：' not in src_text and '：' in tgt_text:
        tgt_parts = tgt_text.split('：')
        if len(tgt_parts) == 2:
            # 원문에서 해당 위치 추정
            tgt_ratio = len(tgt_parts[0]) / len(tgt_text)
            split_point = int(len(src_text) * tgt_ratio)
            
            # 한자 경계에서 분할점 조정
            adjusted_split = split_point
            while adjusted_split < len(src_text) and src_text[adjusted_split] not in '，。；':
                adjusted_split += 1
            
            if adjusted_split < len(src_text):
                part1 = src_text[:adjusted_split + 1]
                part2 = src_text[adjusted_split + 1:]
                return (part1, part2), (tgt_parts[0] + '：', tgt_parts[1])
    
    # 둘 다 콜론이 있는 경우 (기본 처리)
    elif '：' in src_text and '：' in tgt_text:
        src_parts = src_text.split('：')
        tgt_parts = tgt_text.split('：')
        if len(src_parts) == 2 and len(tgt_parts) == 2:
            return (src_parts[0] + '：', src_parts[1]), (tgt_parts[0] + '：', tgt_parts[1])
    
    return None, None

def should_attach_to_previous(text: str) -> bool:
    """앞 구에 붙어야 하는 표현인지 확인 (단순화된 패턴 + MeCab 활용)"""
    
    if not text or not text.strip():
        return False
    
    text = text.strip()
    
    # 1. 핵심 패턴들만 확인 (하드코딩 최소화)
    if text.startswith(('라고 ', '라는 ', '라며', '라면서')):
        return True
    
    # 2. MeCab을 이용한 품사 기반 판단
    if mecab:
        try:
            result = mecab.parse(text)
            morphemes = []
            
            for line in result.split('\n'):
                if line and line != 'EOS':
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        surface = parts[0]
                        pos = parts[1].split(',')[0]
                        morphemes.append((surface, pos))
            
            if morphemes:
                # 첫 번째 형태소가 인용 조사인 경우
                first_surface, first_pos = morphemes[0]
                if first_pos == 'JKQ' and first_surface in ['라고', '라는']:  # 인용격조사
                    return True
                
                # 연결어미로 시작하는 경우
                if first_pos == 'EC' and first_surface in ['라며', '라면서']:
                    return True
        
        except Exception as e:
            logger.debug(f"MeCab 분석 중 오류: {e}")
    
    return False

# 발화동사 원형들 (핵심만, 시제는 MeCab으로 처리)
# 핵심 발화동사 원형 (최소한으로 축소)
CORE_QUOTATIVE_LEMMAS = {
    '말하다', '묻다', '답하다', '이르다'  # 가장 기본적인 발화동사만
}

# MeCab 의미 분류를 활용한 발화동사 탐지용 키워드
COMMUNICATION_KEYWORDS = {
    '말', '언', '답', '문', '물', '이야기', '얘기', '논', '설명', '표현', 
    '진술', '서술', '발언', '언급', '평', '칭', '부르'
}

def detect_quotative_boundary_advanced(text: str) -> bool:
    """MeCab 원형 복원 및 의미 추론을 활용한 발화동사 탐지"""
    
    if not mecab or not text.strip():
        return False
    
    try:
        # MeCab 분석 (원형 정보 포함)
        result = mecab.parse(text.strip())
        morphemes = []
        
        for line in result.split('\n'):
            if line and line != 'EOS':
                parts = line.split('\t')
                if len(parts) >= 2:
                    surface = parts[0]
                    features = parts[1].split(',')
                    pos = features[0]
                    
                    # 원형 정보 추출 (MeCab 출력 구조에 따라)
                    lemma = features[6] if len(features) > 6 and features[6] != '*' else surface
                    
                    morphemes.append({
                        'surface': surface,
                        'pos': pos,
                        'lemma': lemma,
                        'features': features
                    })
        
        if not morphemes:
            return False
        
        # 문장 끝부분에서 발화동사 패턴 찾기
        for i in range(max(0, len(morphemes) - 4), len(morphemes)):
            morph = morphemes[i]
            
            # 1. 동사인지 확인
            if morph['pos'] in ['VV', 'VX']:
                lemma = morph['lemma']
                surface = morph['surface']
                
                # 2-1. 핵심 발화동사 원형 확인 (확실한 경우)
                if lemma in CORE_QUOTATIVE_LEMMAS:
                    return True
                
                # 2-2. 의미적 추론: 소통/대화 관련 키워드 포함 여부
                for keyword in COMMUNICATION_KEYWORDS:
                    if keyword in lemma or keyword in surface:
                        # 3. 종결어미나 연결어미가 뒤따르는지 확인
                        if i + 1 < len(morphemes):
                            next_morph = morphemes[i + 1]
                            if next_morph['pos'] in ['EF', 'EC']:  # 종결어미 or 연결어미
                                return True
                        return True  # 동사 자체로도 판단
                
                # 2-3. 어간 패턴 분석 (보조적)
                verb_stem = lemma[:-1] if lemma.endswith('다') else lemma
                if len(verb_stem) >= 2:  # 최소 길이 확인
                    for keyword in COMMUNICATION_KEYWORDS:
                        if verb_stem.startswith(keyword) or verb_stem.endswith(keyword):
                            return True
        
        return False
        
    except Exception as e:
        logger.debug(f"발화동사 탐지 중 오류: {e}")
        return False

def detect_sentence_ending_type(text: str) -> str:
    """문장 종결 유형 탐지 - 발화동사 여부와 관계없이"""
    
    if not mecab or not text.strip():
        return 'unknown'
    
    try:
        result = mecab.parse(text.strip())
        
        # 마지막 몇 개 형태소 확인
        morphemes = []
        for line in result.split('\n'):
            if line and line != 'EOS':
                parts = line.split('\t')
                if len(parts) >= 2:
                    surface = parts[0]
                    features = parts[1].split(',')
                    morphemes.append((surface, features[0], features))
        
        if not morphemes:
            return 'unknown'
        
        # 종결어미 확인
        for surface, pos, features in reversed(morphemes[-3:]):
            if pos == 'EF':  # 종결어미
                if len(features) > 1:
                    # 종결어미 세부 유형
                    ending_type = features[1]
                    if ending_type in ['평서', '의문', '명령', '청유']:
                        return ending_type
                return 'declarative'  # 기본값
        
        return 'unknown'
        
    except Exception as e:
        logger.debug(f"문장 종결 유형 탐지 중 오류: {e}")
        return 'unknown'

def is_quotative_end_pattern(text: str) -> bool:
    """발화동사 기반 인용 끝 탐지 - 원형 복원 활용"""
    
    # 1. 기본적인 문장 종결 확인
    if not text.strip().endswith(('.', '。', '?', '？', '!', '！')):
        return False
    
    # 2. MeCab 기반 발화동사 탐지 (개선된 버전)
    return detect_quotative_boundary_advanced(text)

def is_discourse_marker(text: str) -> bool:
    """담화 표지 탐지 - MeCab 품사 정보 중심의 일반화된 접근"""
    
    if not mecab or not text.strip():
        return False
    
    # 1. 발화동사도 담화 표지의 일종
    if detect_quotative_boundary_advanced(text):
        return True
    
    # 2. MeCab으로 담화 기능 품사 확인
    try:
        result = mecab.parse(text.strip())
        
        for line in result.split('\n'):
            if line and line != 'EOS':
                parts = line.split('\t')
                if len(parts) >= 2:
                    surface = parts[0]
                    features = parts[1].split(',')
                    pos = features[0]
                    
                    # 담화 기능을 하는 품사들
                    if pos in ['MAJ', 'IC', 'JC'] and len(surface) >= 2:  # 접속부사, 감탄사, 접속조사
                        return True
                    
                    # 접속 의미의 부사
                    if pos == 'MAG' and len(surface) >= 3:  # 일반부사 중 긴 것들 (접속 기능 가능성)
                        # 접속 관련 키워드 포함 여부
                        connection_hints = ['그러', '하지', '따라', '그래', '즉', '또', '게다']
                        if any(hint in surface for hint in connection_hints):
                            return True
                            
    except Exception as e:
        logger.debug(f"담화 표지 탐지 중 오류: {e}")
    
    return False