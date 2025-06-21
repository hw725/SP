"""PA 정렬기 - 유니코드 기반 정규식, 간결한 fallback, 불필요 코드 삭제"""

import regex
from typing import List, Dict

def split_target_by_punct_space_unicode(text: str, max_length: int = 150) -> List[str]:
    """번역문을 유니코드 종결구두점+공백 기준으로 분할, 150자 초과는 맥락 분할"""
    pattern = r'([\p{Term}])\s+'
    parts = regex.split(pattern, text.strip())
    sentences, current = [], ""
    i = 0
    while i < len(parts):
        part = parts[i].strip()
        if not part:
            i += 1
            continue
        if regex.match(r'[\p{Term}]', part):
            current += part
            if current.strip():
                sentences.append(current.strip())
                current = ""
        else:
            current += part
        i += 1
    if current.strip():
        sentences.append(current.strip())
    # 150자 초과 시 맥락 기반 분할
    final = []
    for sent in sentences:
        if len(sent) <= max_length:
            final.append(sent)
        else:
            final.extend(split_by_context_unicode(sent, max_length))
    return final

def split_by_context_unicode(text: str, max_length: int) -> List[str]:
    """맥락(의미 경계) 기반 분할 (유니코드 정규식)"""
    if len(text) <= max_length:
        return [text]
    context_patterns = [
        r'([,，]\s*(?:그런데|그러나|하지만|따라서|그리하여|그러므로|또한|또|그리고))',
        r'([,，]\s*[\'"])',
        r'(라고\s+[했말])',
        r'(다고\s+[했말])',
        r'([,，]\s*(?:이제|그때|그후|먼저|다음에|이어서))',
        r'([,，]\s*<[^>]+>)',
        r'([,，]\s*)',
        r'(에서\s+)',
        r'(에게\s+)',
        r'(으로\s+)',
        r'(로서\s+)',
    ]
    min_pos = int(max_length * 0.3)
    max_pos = int(min(max_length * 0.7, len(text) - 20))
    candidates = []
    for priority, pattern in enumerate(context_patterns):
        for match in regex.finditer(pattern, text):
            pos = match.end()
            if min_pos <= pos <= max_pos:
                center = max_length * 0.5
                distance_score = 1.0 - abs(pos - center) / center
                priority_score = 1.0 - (priority * 0.1)
                total_score = distance_score * 0.7 + priority_score * 0.3
                candidates.append((pos, total_score))
    if candidates:
        best_pos = max(candidates, key=lambda x: x[1])[0]
        left = text[:best_pos].strip()
        right = text[best_pos:].strip()
        result = []
        if left:
            if len(left) > max_length:
                result.extend(split_by_context_unicode(left, max_length))
            else:
                result.append(left)
        if right:
            if len(right) > max_length:
                result.extend(split_by_context_unicode(right, max_length))
            else:
                result.append(right)
        return result
    else:
        mid = len(text) // 2
        return [text[:mid].strip(), text[mid:].strip()]

def split_source_by_han_boundary(src_text: str, target_count: int) -> List[str]:
    """원문을 한자 경계(유니코드)로 분할, 번역문 개수에 맞춤"""
    if target_count <= 1:
        return [src_text]
    boundary_patterns = [
        r'([\p{Han}](?:也|矣|焉|哉))\s*',
        r'([\p{Han}](?:而|然|則|故|且))\s*',
        r'([\p{Han}](?:曰|云)[:：]\s*)',
        r'([，,]\s*)',
        r'([。.]\s*)',
    ]
    chunks = []
    remaining = src_text.strip()
    while remaining:
        found = False
        for pattern in boundary_patterns:
            match = regex.search(pattern, remaining)
            if match:
                end_pos = match.end()
                chunk = remaining[:end_pos].strip()
                if chunk:
                    chunks.append(chunk)
                remaining = remaining[end_pos:].strip()
                found = True
                break
        if not found:
            if remaining.strip():
                chunks.append(remaining.strip())
            break
    # 개수 조정
    if len(chunks) == target_count:
        return chunks
    elif len(chunks) < target_count:
        return expand_chunks_unicode(chunks, target_count)
    else:
        return merge_chunks_unicode(chunks, target_count)

def expand_chunks_unicode(chunks: List[str], target_count: int) -> List[str]:
    """청크 확장 (유니코드)"""
    expanded = []
    need_expand = target_count - len(chunks)
    chunks_with_length = sorted([(i, chunk, len(chunk)) for i, chunk in enumerate(chunks)], key=lambda x: x[2], reverse=True)
    expand_indices = set(x[0] for x in chunks_with_length[:need_expand])
    for i, chunk in enumerate(chunks):
        if i in expand_indices and len(chunk) > 10:
            mid = len(chunk) // 2
            split_pos = find_split_point_unicode(chunk, mid)
            left = chunk[:split_pos].strip()
            right = chunk[split_pos:].strip()
            if left and right:
                expanded.extend([left, right])
            else:
                expanded.append(chunk)
        else:
            expanded.append(chunk)
    return expanded

def find_split_point_unicode(text: str, target_pos: int) -> int:
    """적절한 분할점 찾기 (유니코드)"""
    search_range = min(10, len(text) // 4)
    for offset in range(search_range):
        pos = target_pos + offset
        if pos < len(text) and is_good_split_char_unicode(text, pos):
            return pos
        pos = target_pos - offset
        if pos > 0 and is_good_split_char_unicode(text, pos):
            return pos
    return target_pos

def is_good_split_char_unicode(text: str, pos: int) -> bool:
    """좋은 분할 지점인지 판단 (유니코드)"""
    if pos <= 0 or pos >= len(text):
        return False
    char = text[pos]
    prev_char = text[pos - 1] if pos > 0 else ''
    if prev_char in ' \t，,。.' or (regex.match(r'\p{Han}', char) and regex.match(r'\p{Han}', prev_char)):
        return True
    return False

def merge_chunks_unicode(chunks: List[str], target_count: int) -> List[str]:
    """청크 병합 (유니코드)"""
    if len(chunks) <= target_count:
        return chunks
    merged = []
    chunks_per_group = len(chunks) / target_count
    current_group = []
    group_size = 0
    for chunk in chunks:
        current_group.append(chunk)
        group_size += 1
        if group_size >= chunks_per_group or chunk == chunks[-1]:
            merged_text = ' '.join(current_group).strip()
            merged.append(merged_text)
            current_group = []
            group_size = 0
    return merged

def align_paragraphs_unicode(
    src_paragraph: str,
    tgt_paragraph: str,
    embed_func,
    max_length: int = 150,
    similarity_threshold: float = 0.3
) -> List[Dict]:
    """유니코드 기반 번역문 우선 정렬"""
    tgt_units = split_target_by_punct_space_unicode(tgt_paragraph, max_length)
    src_units = split_source_by_han_boundary(src_paragraph, len(tgt_units))
    from sklearn.metrics.pairwise import cosine_similarity
    alignments = []
    max_len = max(len(src_units), len(tgt_units))
    while len(src_units) < max_len:
        src_units.append("")
    while len(tgt_units) < max_len:
        tgt_units.append("")
    try:
        valid_src = [s for s in src_units if s.strip()]
        valid_tgt = [t for t in tgt_units if t.strip()]
        if valid_src and valid_tgt:
            src_embeddings = embed_func(valid_src)
            tgt_embeddings = embed_func(valid_tgt)
            similarities = []
            for i in range(max_len):
                if i < len(valid_src) and i < len(valid_tgt):
                    sim = cosine_similarity([tgt_embeddings[i]], [src_embeddings[i]])[0][0]
                    similarities.append(sim)
                else:
                    similarities.append(0.0)
        else:
            similarities = [0.0] * max_len
    except Exception:
        similarities = [0.0] * max_len
    for i in range(max_len):
        alignments.append({
            '원문': src_units[i] if i < len(src_units) else "",
            '번역문': tgt_units[i] if i < len(tgt_units) else "",
            'similarity': float(similarities[i]) if i < len(similarities) else 0.0,
            'split_method': 'unicode_punct',
            'align_method': 'unicode_1to1'
        })