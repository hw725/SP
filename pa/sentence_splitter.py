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

def split_target_sentences_advanced(text: str, max_length: int = 150, splitter: str = "spacy") -> List[str]:
    """
    번역문 분할 - spaCy 의미 단위 + 기존 기준 후처리
    splitter: "spacy" (기본)
    """
    semantic_sentences = split_with_spacy(text, is_target=True)
    if not semantic_sentences:
        semantic_sentences = split_with_smart_punctuation_rules(text)
    final_sentences = apply_legacy_rules(semantic_sentences, max_length)
    return final_sentences

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

def split_source_by_whitespace_and_align(src_text: str, target_sentences: List[str], embed_func, similarity_threshold: float = 0.5, max_tokens: int = 50) -> List[str]:
    """
    원문을 공백/구두점 단위로 분할한 뒤, 번역문 문장과 의미적으로 가장 잘 맞는 경계에서 토큰을 합쳐 alignment.
    **수정**: 공백을 보존하여 원본 텍스트 구조 유지
    """
    # 구분자를 보존하면서 토큰화 - 원본 위치 정보 추적
    delimiters = r'([：。！？；、，\s]+)'  # 괄호 추가로 구분자도 함께 캡처
    parts = re.split(delimiters, src_text)
    
    # 빈 문자열 제거하고 토큰과 구분자 분리
    tokens = []
    separators = []
    current_pos = 0
    
    for i, part in enumerate(parts):
        if part and part.strip():  # 비어있지 않은 토큰
            if re.match(r'[：。！？；、，\s]+', part):  # 구분자인 경우
                if tokens:  # 이전 토큰이 있는 경우에만 구분자 저장
                    separators.append(part)
            else:  # 실제 토큰인 경우
                tokens.append(part)
                if i < len(parts) - 1:  # 마지막이 아닌 경우 다음 구분자 찾기
                    next_sep = ""
                    for j in range(i + 1, len(parts)):
                        if parts[j] and re.match(r'[：。！？；、，\s]+', parts[j]):
                            next_sep = parts[j]
                            break
                        elif parts[j]:  # 다음 토큰을 만난 경우
                            break
                    separators.append(next_sep)
    
    # 구분자 수 조정
    while len(separators) < len(tokens):
        separators.append('')
    
    if not tokens or len(tokens) < len(target_sentences):
        # 원본이 부족한 경우 원본 그대로 반환하되 개수 맞춤
        return [src_text] + [''] * (len(target_sentences) - 1) if src_text else [''] * len(target_sentences)
    
    # 의미적 병합 (임베딩 유사도 기반, 순서 보존)
    aligned_chunks = []
    start = 0
    
    for tgt in target_sentences:
        best_end = min(start + max_tokens, len(tokens))
        best_score = -float('inf')
        best_idx = start + 1
        
        for end in range(start + 1, best_end + 1):
            # 토큰과 구분자를 원본 순서대로 결합
            chunk_parts = []
            for i in range(start, end):
                chunk_parts.append(tokens[i])
                if i < len(separators) and i < end - 1:  # 마지막 토큰이 아닌 경우에만 구분자 추가
                    chunk_parts.append(separators[i])
            
            chunk = ''.join(chunk_parts)
            
            try:
                score = float(embed_func([chunk])[0] @ embed_func([tgt])[0])
            except Exception:
                score = 0.0
            
            if score > best_score:
                best_score = score
                best_idx = end
        
        # 최적 청크 생성 (구분자 포함)
        chunk_parts = []
        for i in range(start, best_idx):
            chunk_parts.append(tokens[i])
            if i < len(separators) and i < best_idx - 1:
                chunk_parts.append(separators[i])
        
        aligned_chunks.append(''.join(chunk_parts))
        start = best_idx
        
        if start >= len(tokens):
            break
    
    # 남은 토큰이 있으면 마지막에 합침
    if start < len(tokens):
        remaining_parts = []
        for i in range(start, len(tokens)):
            remaining_parts.append(tokens[i])
            if i < len(separators):
                remaining_parts.append(separators[i])
        aligned_chunks.append(''.join(remaining_parts))
    
    # 결과 길이를 target_sentences와 맞춤
    result = aligned_chunks[:len(target_sentences)]
    while len(result) < len(target_sentences):
        result.append('')
    
    return result