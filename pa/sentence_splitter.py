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
    번역문 분할 - 반드시 문장 종결부호+공백 기준만 사용 (spaCy 등 의미 단위 분할 완전 제거)
    """
    # 종결부호(한글/한자/영문) + 공백 또는 텍스트 끝 기준 분할
    # 종결부호: . ? ! 。" ？ ！ ○ 등
    pattern = r'(?<=[.!?。？！○])\s+'  # 종결부호 뒤 공백 기준
    sentences = re.split(pattern, text.strip())
    # 빈 문장 제거하고 앞뒤 공백 제거
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

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