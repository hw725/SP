"""PA 문장 분할기 - 번역문/원문 모두 의미 기반 + 기존 기준 후처리"""

import re
import regex
from typing import List, Tuple
import spacy

# ✅ spaCy 모델 (lg 우선) - 조용히 로드
try:
    nlp_ko = spacy.load("ko_core_news_lg")
except OSError:
    try:
        nlp_ko = spacy.load("ko_core_news_sm")
    except OSError:
        nlp_ko = None

try:
    nlp_zh = spacy.load("zh_core_web_lg")
except OSError:
    try:
        nlp_zh = spacy.load("zh_core_web_sm")
    except OSError:
        nlp_zh = None

# ✅ Stanza 모델 (옵션)
try:
    import stanza
    stanza_ko = stanza.Pipeline(lang='ko', processors='tokenize', tokenize_no_ssplit=False, use_gpu=False)
except Exception:
    stanza_ko = None

try:
    stanza_zh = stanza.Pipeline(lang='zh-hant', processors='tokenize', tokenize_no_ssplit=False, use_gpu=False)
except Exception:
    stanza_zh = None

def split_target_sentences_advanced(text: str, max_length: int = 150, splitter: str = "spacy") -> List[str]:
    """
    번역문 분할 - spaCy/stanza 의미 단위 + 기존 기준 후처리
    splitter: "spacy" (기본), "stanza" (옵션)
    """
    if splitter == "stanza":
        semantic_sentences = split_with_stanza(text)
    else:
        semantic_sentences = split_with_spacy(text, is_target=True)

    if not semantic_sentences:
        semantic_sentences = split_with_smart_punctuation_rules(text)

    final_sentences = apply_legacy_rules(semantic_sentences, max_length)
    return final_sentences

def split_with_stanza(text: str) -> List[str]:
    """Stanza로 문장 분할"""
    if contains_chinese(text):
        nlp_model = stanza_zh
    else:
        nlp_model = stanza_ko

    if not nlp_model:
        return []

    try:
        doc = nlp_model(text)
        sentences = [sentence.text.strip() for sentence in doc.sentences if sentence.text.strip()]
        return sentences
    except Exception:
        return []

def split_with_smart_punctuation_rules(text: str) -> List[str]:
    """지능적 구두점 기반 분할 (콤마 조건부)"""
    primary_splits = split_by_sentence_endings(text)
    final_splits = []
    for segment in primary_splits:
        subsegments = split_by_conditional_comma(segment)
        final_splits.extend(subsegments)
    return final_splits

def split_by_sentence_endings(text: str) -> List[str]:
    """종결부호 기반 분할 (1차)"""
    pattern = r'(?<=[。？！○])|(?<=[.!?]\s)'
    segments = re.split(pattern, text)
    result = []
    for seg in segments:
        seg = seg.strip()
        if seg:
            result.append(seg)
    return result if result else [text.strip()]

def split_by_conditional_comma(text: str) -> List[str]:
    """조건부 콤마/콜론 분할 (2차)"""
    if len(text) < 30:
        return [text]
    colon_splits = split_by_colons(text)
    final_splits = []
    for segment in colon_splits:
        comma_splits = split_by_smart_comma(segment)
        final_splits.extend(comma_splits)
    return final_splits

def split_by_colons(text: str) -> List[str]:
    """콜론 기반 분할"""
    pattern = r'(?<=[：])|(?<=[:]\s)'
    segments = re.split(pattern, text)
    result = []
    for seg in segments:
        seg = seg.strip()
        if seg:
            result.append(seg)
    return result if result else [text]

def split_by_smart_comma(text: str) -> List[str]:
    """지능적 콤마 분할 (의미 경계만)"""
    if '，' not in text and ',' not in text:
        return [text]
    if len(text) < 50:
        return [text]
    meaningful_comma_splits = find_meaningful_comma_splits(text)
    return meaningful_comma_splits

def find_meaningful_comma_splits(text: str) -> List[str]:
    """의미 있는 콤마 분할점 찾기"""
    comma_pattern = r'[，,]'
    comma_matches = list(re.finditer(comma_pattern, text))
    if not comma_matches:
        return [text]
    split_points = []
    for match in comma_matches:
        comma_pos = match.start()
        if is_meaningful_comma_boundary(text, comma_pos):
            split_points.append(comma_pos + 1)
    if not split_points:
        return [text]
    segments = []
    start = 0
    for split_pos in split_points:
        segment = text[start:split_pos].strip()
        if segment:
            segments.append(segment)
        start = split_pos
    if start < len(text):
        segment = text[start:].strip()
        if segment:
            segments.append(segment)
    return segments if segments else [text]

def is_meaningful_comma_boundary(text: str, comma_pos: int) -> bool:
    """콤마가 의미 있는 경계인지 판단"""
    start = max(0, comma_pos - 15)
    end = min(len(text), comma_pos + 15)
    before_comma = text[start:comma_pos].strip()
    after_comma = text[comma_pos + 1:end].strip()
    if is_simple_enumeration(before_comma, after_comma):
        return False
    if has_semantic_transition(before_comma, after_comma):
        return True
    if len(before_comma) > 20 and len(after_comma) > 20:
        return True
    return False

def is_simple_enumeration(before: str, after: str) -> bool:
    """단순 병렬 나열인지 판단"""
    if len(before) < 10 and len(after) < 10:
        return True
    if re.search(r'[0-9①②③④⑤一二三四五]$', before) and re.search(r'^[0-9①②③④⑤一二三四五]', after):
        return True
    simple_patterns = [
        r'[가-힣]{1,3}$',
        r'[A-Za-z]{1,8}$',
        r'\p{Han}{1,3}$',
    ]
    for pattern in simple_patterns:
        if pattern.startswith(r'\p{Han}'):
            if regex.search(pattern, before.strip()) and regex.search(f'^{pattern[:-1]}', after.strip()):
                return True
        else:
            if re.search(pattern, before.strip()) and re.search(f'^{pattern[:-1]}', after.strip()):
                return True
    return False

def has_semantic_transition(before: str, after: str) -> bool:
    """의미 전환이 있는지 판단"""
    transition_patterns = [
        r'(그러나|하지만|그런데|따라서|그래서|또한|또|더욱이|반면|한편)',
        r'(但是|然而|因此|所以|而且|另外|同时|相反)',
        r'(however|but|therefore|also|moreover|meanwhile)',
    ]
    for pattern in transition_patterns:
        if re.search(pattern, after[:10]):
            return True
    if before.endswith(('다', '음', '임')) and after.startswith(('왜', '어떻게', '언제', '무엇')):
        return True
    return False

def split_source_with_spacy(src_text: str, target_sentences, splitter: str = "spacy") -> List[str]:
    """원문 분할 - 번역문 의미 경계와 매칭 (spaCy/stanza 선택)"""
    if isinstance(target_sentences, int):
        target_count = target_sentences
        target_sentences = []
    elif isinstance(target_sentences, list):
        target_count = len(target_sentences)
    else:
        target_count = 1
        target_sentences = []

    if splitter == "stanza":
        source_semantic_units = split_with_stanza(src_text)
    else:
        source_semantic_units = split_with_spacy(src_text, is_target=False)

    if not source_semantic_units:
        source_semantic_units = split_source_with_smart_punctuation(src_text)

    if len(source_semantic_units) == target_count:
        final_chunks = source_semantic_units
    elif len(source_semantic_units) < target_count:
        final_chunks = expand_semantic_chunks(source_semantic_units, target_count, target_sentences)
    else:
        final_chunks = merge_semantic_chunks(source_semantic_units, target_count, target_sentences)

    return final_chunks

def split_source_with_smart_punctuation(src_text: str) -> List[str]:
    """원문 지능적 구두점 분할"""
    return split_with_smart_punctuation_rules(src_text)

def split_with_spacy(text: str, is_target: bool = True) -> List[str]:
    """spaCy로 의미 단위 분할 (최우선)"""
    if contains_chinese(text):
        nlp_model = nlp_zh
    else:
        nlp_model = nlp_ko
    if not nlp_model:
        return []
    try:
        doc = nlp_model(text)
        sentences = []
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if sent_text:
                sentences.append(sent_text)
        if sentences:
            return sentences
    except Exception:
        pass
    return []

def apply_legacy_rules(sentences: List[str], max_length: int = 150) -> List[str]:
    """기존 기준 후처리 (150자 제한 + 한자 3개 이하 병합)"""
    length_adjusted = []
    for sent in sentences:
        if len(sent) > max_length:
            parts = split_long_sentence_semantically(sent, max_length)
            length_adjusted.extend(parts)
        else:
            length_adjusted.append(sent)
    final_sentences = merge_low_chinese_segments(length_adjusted)
    return final_sentences

def split_long_sentence_semantically(sentence: str, max_length: int) -> List[str]:
    """긴 문장을 의미 보존하며 분할"""
    if len(sentence) <= max_length:
        return [sentence]
    parts = []
    remaining = sentence
    while len(remaining) > max_length:
        split_pos = find_semantic_split_near_position(remaining, max_length)
        if split_pos > 0:
            part = remaining[:split_pos].strip()
            if part:
                parts.append(part)
            remaining = remaining[split_pos:].strip()
        else:
            parts.append(remaining[:max_length])
            remaining = remaining[max_length:]
    if remaining.strip():
        parts.append(remaining.strip())
    return parts

def find_semantic_split_near_position(text: str, target_pos: int) -> int:
    """목표 위치 근처에서 의미 있는 분할점 찾기 (우선순위 개선)"""
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
            best_match = min(matches, key=lambda m: abs((start + m.start()) - target_pos))
            return start + best_match.start() + offset
    return target_pos

def merge_low_chinese_segments(sentences: List[str]) -> List[str]:
    """한자 3개 이하 세그먼트 병합 (✅ regex 사용)"""
    if not sentences:
        return []
    merged, buffer = [], ''
    for sent in sentences:
        han_count = len(regex.findall(r'\p{Han}', sent))
        if han_count <= 3:
            if buffer:
                buffer += ' ' + sent
            else:
                buffer = sent
        else:
            if buffer:
                if merged:
                    merged[-1] += ' ' + buffer
                else:
                    merged.append(buffer)
                buffer = ''
            merged.append(sent)
    if buffer:
        if merged:
            merged[-1] += ' ' + buffer
        else:
            merged.append(buffer)
    return [s.strip() for s in merged if s.strip()]

def expand_semantic_chunks(source_chunks: List[str], target_count: int, target_sentences: List[str]) -> List[str]:
    """의미 보존하며 원문 청크 확장"""
    expanded = source_chunks.copy()
    target_idx = 0
    while len(expanded) < target_count and target_idx < len(target_sentences):
        if target_idx < len(expanded):
            current_chunk = expanded[target_idx]
            if len(current_chunk) > 30:
                split_result = split_chunk_semantically(current_chunk, target_sentences[target_idx:target_idx+2])
                if split_result and len(split_result) > 1:
                    expanded[target_idx] = split_result[0]
                    for i, part in enumerate(split_result[1:], 1):
                        expanded.insert(target_idx + i, part)
                    target_idx += len(split_result)
                    continue
        target_idx += 1
        if target_idx >= len(expanded):
            break
    return expanded

def merge_semantic_chunks(source_chunks: List[str], target_count: int, target_sentences: List[str]) -> List[str]:
    """의미 보존하며 원문 청크 병합"""
    merged = source_chunks.copy()
    while len(merged) > target_count:
        best_merge_idx = find_best_semantic_merge(merged, target_sentences)
        merged[best_merge_idx] = merged[best_merge_idx] + merged[best_merge_idx + 1]
        merged.pop(best_merge_idx + 1)
    return merged

def split_chunk_semantically(chunk: str, related_targets: List[str]) -> List[str]:
    """의미 관련성을 고려한 청크 분할"""
    if len(related_targets) < 2:
        return [chunk]
    mid_point = len(chunk) // 2
    split_result = find_semantic_split_near_position(chunk, mid_point)
    if split_result > 5 and split_result < len(chunk) - 5:
        part1 = chunk[:split_result].strip()
        part2 = chunk[split_result:].strip()
        if part1 and part2:
            return [part1, part2]
    return [chunk]

def find_best_semantic_merge(chunks: List[str], target_sentences: List[str]) -> int:
    """의미적 연관성이 높은 병합 후보 찾기"""
    best_score = float('inf')
    best_idx = 0
    for i in range(len(chunks) - 1):
        len_score = len(chunks[i]) + len(chunks[i + 1])
        ratio1 = get_chinese_ratio(chunks[i])
        ratio2 = get_chinese_ratio(chunks[i + 1])
        lang_similarity = abs(ratio1 - ratio2) * 100
        total_score = len_score + lang_similarity
        if total_score < best_score:
            best_score = total_score
            best_idx = i
    return best_idx

def contains_chinese(text: str) -> bool:
    """중국어/한문 포함 여부 확인 (✅ regex 사용)"""
    chinese_count = len(regex.findall(r'\p{Han}', text))
    return chinese_count > len(text) * 0.3

def get_chinese_ratio(text: str) -> float:
    """중국어 문자 비율 (✅ regex 사용)"""
    if not text:
        return 0.0
    chinese_count = len(regex.findall(r'\p{Han}', text))
    return chinese_count / len(text)

# ✅ 기존 호환성을 위한 래퍼 함수
def split_sentences(text: str) -> List[str]:
    """기존 호환성을 위한 래퍼"""
    return split_target_sentences_advanced(text, 150)