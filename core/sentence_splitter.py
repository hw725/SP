"""PA 문장 분할기 - 번역문은 spaCy, 원문은 의미적 병합만 지원"""
from typing import List, Tuple

# 번역문 분할에만 spaCy 사용
import re
import regex
try:
    import spacy
    nlp_ko = spacy.load("ko_core_news_lg")
except Exception:
    nlp_ko = None
try:
    nlp_zh = spacy.load("zh_core_web_lg")
except Exception:
    nlp_zh = None

def split_target_sentences_advanced(text: str, max_length: int = 150, splitter: str = "punctuation") -> List[str]:
    """
    번역문 분할 - 닫는 따옴표 홀로 분할 방지 및 공백 내부 분리 방지
    """
    # 1차: 기본 종결부호 기준 분할 (단, 구두점 앞뒤가 모두 공백인 경우는 분할하지 않음)
    import re
    # 종결부호 앞뒤가 모두 공백이 아닌 경우만 분할
    pattern = r'(?<=[.!?。？！○])(?=\s+[^ ])'
    sentences = re.split(pattern, text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # 2차: 닫는 따옴표 홀로 분할된 것 병합
    merged = []
    i = 0
    while i < len(sentences):
        current = sentences[i]
        # 닫는 따옴표만 있거나 매우 짧은 경우 이전 문장과 병합
        quote_only = current in ['"', "'", '"', "'", '」', '』', '"']
        short_quote = (len(current.strip()) <= 3 and 
                      re.match(r'^[""\'』」"\"]+', current.strip()))
        
        if i > 0 and (quote_only or short_quote):
            merged[-1] += ' ' + current
        else:
            merged.append(current)
        i += 1
    
    return merged if merged else [text]

def split_with_spacy(text: str, is_target: bool = True) -> List[str]:
    if contains_chinese(text):
        nlp_model = nlp_zh
    else:
        nlp_model = nlp_ko
    if not nlp_model:
        return []
    try:
        doc = nlp_model(text)
        return [sent.text for sent in doc.sents if sent.text]
    except Exception:
        return []

def split_with_smart_punctuation_rules(text: str) -> List[str]:
    pattern = r'(?<=[。？！○])|(?<=[.!?]\s)'
    segments = re.split(pattern, text)
    return [seg for seg in segments if seg]

def apply_legacy_rules(sentences: List[str], max_length: int = 150) -> List[str]:
    length_adjusted = []
    for sent in sentences:
        if len(sent) > max_length:
            length_adjusted.extend(split_long_sentence_semantically(sent, max_length))
        else:
            length_adjusted.append(sent)
    return merge_low_chinese_segments(length_adjusted)

def split_long_sentence_semantically(sentence: str, max_length: int) -> List[str]:
    if len(sentence) <= max_length:
        return [sentence]
    parts = []
    remaining = sentence
    while len(remaining) > max_length:
        split_pos = find_semantic_split_near_position(remaining, max_length)
        if split_pos > 0:
            parts.append(remaining[:split_pos])
            remaining = remaining[split_pos:]
        else:
            break
    if remaining:
        parts.append(remaining)
    return parts

def find_semantic_split_near_position(text: str, target_pos: int) -> int:
    start = max(0, target_pos - 20)
    end = min(len(text), target_pos + 20)
    search_text = text[start:end]
    split_patterns = [
        (r'[。！？○]', 1),
        (r'[.!?]\s', 2),
        (r'[：]', 1),
        (r'[:]\s', 2),
        (r'[，]\s*(?=.{10,})', 1),
        (r'[,]\s+(?=.{10,})', 2),
        (r'\s+', 1),
    ]
    for pattern, offset in split_patterns:
        matches = list(re.finditer(pattern, search_text))
        if matches:
            return start + matches[0].end()
    return target_pos

def merge_low_chinese_segments(sentences: List[str]) -> List[str]:
    if not sentences:
        return []
    merged, buffer = [], ''
    for sent in sentences:
        han_count = len(regex.findall(r'\p{Han}', sent))
        if han_count <= 3:
            buffer += sent
        else:
            if buffer:
                merged.append(buffer)
                buffer = ''
            merged.append(sent)
    if buffer:
        if merged:
            merged[-1] += buffer
        else:
            merged.append(buffer)
    return [s for s in merged if s]

def contains_chinese(text: str) -> bool:
    chinese_count = len(regex.findall(r'\p{Han}', text))
    return chinese_count > len(text) * 0.3

def split_source_by_whitespace_and_align(source: str, target_count: int) -> List[str]:
    """
    원문(한문) 분할: 번역문 분할 개수에 맞춰 순차적으로 분할(병합/패딩), 모든 공백/포맷 100% 보존
    
    주어+발화동사+인용구 병합 시에도 공백이 손상되지 않도록 보장
    """
    if not source.strip():
        return [''] * target_count
    
    # 1. 원문을 의미 단위로 토큰화 (공백 보존)
    # 구두점과 공백을 구분자로 사용하되, 구분자도 보존
    import re
    # 구분자: 중국어 구두점, 공백, 줄바꿈 등
    delimiter_pattern = r'([：。！？；、，\s]+)'
    parts = re.split(delimiter_pattern, source)
    
    # 빈 문자열 제거하지 않고 모든 부분 보존
    tokens = []
    for part in parts:
        if part:  # 빈 문자열이 아닌 경우만
            tokens.append(part)
    
    if not tokens:
        return [''] * target_count
    
    # 2. 번역문 개수에 맞춰 순차적으로 토큰들을 병합
    if len(tokens) <= target_count:
        # 토큰이 적은 경우: 각 토큰을 하나씩 배정하고 나머지는 빈 문자열
        result = tokens + [''] * (target_count - len(tokens))
        return result[:target_count]
    else:
        # 토큰이 많은 경우: 균등하게 분배하여 병합
        chunk_size = len(tokens) // target_count
        remainder = len(tokens) % target_count
        
        result = []
        start = 0
        for i in range(target_count):
            # 나머지가 있으면 앞쪽 청크들에 하나씩 더 배정
            current_chunk_size = chunk_size + (1 if i < remainder else 0)
            end = start + current_chunk_size
            
            # 토큰들을 그대로 연결 (공백/구두점 보존)
            chunk = ''.join(tokens[start:end])
            result.append(chunk)
            start = end
        
        return result

def split_source_by_meaning_units(source: str, target_count: int) -> List[str]:
    """
    원문(한문/한글) 의미 단위 분할: 한글 어미에서 분리되지 않도록 개선하고, 어절 내부 분할을 방지.
    """
    if not source.strip():
        return [''] * target_count
    
    # 1. 의미 단위로 1차 분할 (구두점 기준)
    meaning_pattern = r'([。！？；、，：]+)'
    parts = re.split(meaning_pattern, source)
    
    # 구두점과 앞 텍스트 재결합
    units = []
    i = 0
    while i < len(parts):
        if parts[i].strip():
            current = parts[i]
            if i + 1 < len(parts) and re.match(r'^[。！？；、，：]+$', parts[i + 1]):
                current += parts[i + 1]
                i += 2
            else:
                i += 1
            units.append(current)
        else:
            i += 1
    
    # 한글 어미(다, 니다, 요 등)에서 분리되는 경우 병합
    def is_korean_ending(u):
        return bool(re.search(r'(다|니다|요|라|까|죠|네|군|구나|구요|네요|랍니다|랍니까|라니|라면|라서|라니까|라더라|라더군|라더라고|라더니|라더냐|라더라구요)$', u.strip()))
    # 병합: 어미만 단독으로 분리된 경우 앞 단위와 합침
    merged = []
    for u in units:
        if merged and is_korean_ending(u):
            merged[-1] += u
        else:
            merged.append(u)
    units = merged
    
    if not units:
        return [''] * target_count
    
    # 2. target_count에 맞춰 조정
    if len(units) == target_count:
        return units
    elif len(units) < target_count:
        # 단위가 부족하면 긴 단위를 어절 단위로 분할
        while len(units) < target_count:
            splittable_units = [(i, u) for i, u in enumerate(units) if ' ' in u.strip() and len(u) > 10]
            if not splittable_units:
                break
            
            original_idx, longest_unit = max(splittable_units, key=lambda item: len(item[1]))
            mid = len(longest_unit) // 2
            
            left_pos = longest_unit.rfind(' ', 0, mid)
            right_pos = longest_unit.find(' ', mid)
            
            split_at = -1
            if left_pos != -1 and right_pos != -1:
                split_at = left_pos if mid - left_pos <= right_pos - mid else right_pos
            else:
                split_at = left_pos if left_pos != -1 else right_pos

            if split_at != -1:
                part1 = longest_unit[:split_at].strip()
                part2 = longest_unit[split_at+1:].strip()
                if part1 and part2:
                    units[original_idx:original_idx+1] = [part1, part2]
                else:
                    break
            else:
                break
        
        # 여전히 부족하면 빈 문자열 추가
        units.extend([''] * (target_count - len(units)))
    else:
        # 단위가 많으면 인접한 것들 병합
        while len(units) > target_count:
            min_len = float('inf')
            merge_idx = 0
            for i in range(len(units) - 1):
                combined_len = len(units[i]) + len(units[i + 1])
                if combined_len < min_len:
                    min_len = combined_len
                    merge_idx = i
            units[merge_idx:merge_idx+2] = [units[merge_idx] + units[merge_idx + 1]]
    
    return units[:target_count]
