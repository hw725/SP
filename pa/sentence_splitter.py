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
    번역문 분할 - 반각 종결부호+공백 → 길이 조정만 (spaCy는 긴 문장 분할시에만 사용)
    인용문 패턴("잘 시행된다."고 말한다.) 보호를 위해 반각 종결부호+공백 기준으로 기본 분할
    """
    print(f"\n=== 순차적 분할 시작 ===")
    print(f"원본 텍스트: {text}")
    
    # 1단계: 반각 종결부호+공백 기준 분할 (인용문 패턴 보호)
    initial_sentences = split_with_smart_punctuation_rules(text)
    print(f"1단계 (구두점+공백): {len(initial_sentences)}개 - {initial_sentences}")
    
    # 2단계: 150자 초과시만 spaCy를 사용한 추가 분할 (인용문 보호 포함)
    final_sentences = apply_legacy_rules(initial_sentences, max_length)
    print(f"2단계 (길이 조정): {len(final_sentences)}개 - {final_sentences}")
    
    # 3단계는 제거 - spaCy는 오직 긴 문장 분할에만 사용
    print(f"최종 결과: {len(final_sentences)}개 - {final_sentences}")
    return final_sentences

def split_with_spacy(text: str, is_target: bool = True) -> List[str]:
    if contains_chinese(text):
        nlp_model = nlp_zh
    else:
        nlp_model = nlp_ko
    if not nlp_model:
        print(f"spaCy 모델 없음: {'중국어' if contains_chinese(text) else '한국어'}")
        return []
    try:
        doc = nlp_model(text)
        sentences = [sent.text for sent in doc.sents if sent.text]
        print(f"spaCy 분할 결과: {len(sentences)}개 - {sentences}")
        return sentences
    except Exception as e:
        print(f"spaCy 오류: {e}")
        return []

def split_with_smart_punctuation_rules(text: str) -> List[str]:
    pattern = r'(?<=[。？！○])|(?<=[.!?])\s+'
    segments = re.split(pattern, text)
    result = [seg.strip() for seg in segments if seg.strip()]
    print(f"구두점 분할 결과: {len(result)}개 - {result}")
    return result

def apply_legacy_rules(sentences: List[str], max_length: int = 150) -> List[str]:
    """2단계: 길이 조정 - spaCy를 사용한 맥락적 분할"""
    length_adjusted = []
    for sent in sentences:
        if len(sent) > max_length:
            # 통합된 의미적 분할 함수 사용 (spaCy 우선, fallback 포함)
            semantic_chunks = split_long_sentence_semantically(sent, max_length)
            length_adjusted.extend(semantic_chunks)
        else:
            length_adjusted.append(sent)
    # 번역문 분할에서는 한자 병합 로직 사용하지 않음
    return length_adjusted



def split_long_sentence_semantically(sentence: str, max_length: int) -> List[str]:
    """spaCy를 사용해서 맥락에 맞게 긴 문장을 분할"""
    if len(sentence) <= max_length:
        return [sentence]
    
    # spaCy로 먼저 시도
    spacy_chunks = split_long_sentence_with_spacy(sentence, max_length)
    if spacy_chunks and len(spacy_chunks) > 1:
        print(f"spaCy 맥락 분할 성공: {len(spacy_chunks)}개 - {spacy_chunks}")
        return spacy_chunks
    
    # spaCy 실패시 기존 패턴 기반 분할
    print(f"spaCy 분할 실패, 패턴 기반 분할 사용")
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

def split_long_sentence_with_spacy(sentence: str, max_length: int) -> List[str]:
    """spaCy를 사용해서 맥락에 맞게 긴 문장을 분할 (인용문 패턴 보호)"""
    if len(sentence) <= max_length:
        return [sentence]
    
    # 인용문 패턴 보호를 위한 마스킹 (유니코드 따옴표 포함)
    protected_patterns = [
        # 일반 따옴표
        (r'"[^"]*"고\s+(말한다|한다)', '인용문_패턴'),
        (r'"[^"]*"라고\s+(말한다|한다)', '인용문_패턴'),
        (r'"[^"]*"\s*고\s+(말한다|한다)', '인용문_패턴'),
        (r'"[^"]*"\s*라고\s+(말한다|한다)', '인용문_패턴'),
        # 유니코드 따옴표 " " (U+201C, U+201D)
        (r'\u201c[^\u201d]*\u201d고\s+(말한다|한다)', '인용문_패턴'),
        (r'\u201c[^\u201d]*\u201d라고\s+(말한다|한다)', '인용문_패턴'),
        (r'\u201c[^\u201d]*\u201d\s*고\s+(말한다|한다)', '인용문_패턴'),
        (r'\u201c[^\u201d]*\u201d\s*라고\s+(말한다|한다)', '인용문_패턴'),
        # 유니코드 따옴표 「 」 (U+300C, U+300D)
        (r'\u300c[^\u300d]*\u300d고\s+(말한다|한다)', '인용문_패턴'),
        (r'\u300c[^\u300d]*\u300d라고\s+(말한다|한다)', '인용문_패턴'),
        (r'\u300c[^\u300d]*\u300d\s*고\s+(말한다|한다)', '인용문_패턴'),
        (r'\u300c[^\u300d]*\u300d\s*라고\s+(말한다|한다)', '인용문_패턴'),
        # 유니코드 따옴표 ' ' (U+2018, U+2019)
        (r'\u2018[^\u2019]*\u2019고\s+(말한다|한다)', '인용문_패턴'),
        (r'\u2018[^\u2019]*\u2019라고\s+(말한다|한다)', '인용문_패턴'),
        (r'\u2018[^\u2019]*\u2019\s*고\s+(말한다|한다)', '인용문_패턴'),
        (r'\u2018[^\u2019]*\u2019\s*라고\s+(말한다|한다)', '인용문_패턴'),
        # 혼합 패턴 (구형 호환성)
        (r'["""][^"""]*["""]고\s+(말한다|한다)', '인용문_패턴'),
        (r'["""][^"""]*["""]라고\s+(말한다|한다)', '인용문_패턴'),
    ]
    
    masked_sentence = sentence
    mask_mappings = {}
    mask_counter = 0
    
    for pattern, mask_type in protected_patterns:
        matches = list(re.finditer(pattern, masked_sentence))
        for match in reversed(matches):  # 뒤에서부터 처리해서 인덱스 충돌 방지
            mask_id = f"__MASK_{mask_counter}__"
            mask_mappings[mask_id] = match.group()
            masked_sentence = masked_sentence[:match.start()] + mask_id + masked_sentence[match.end():]
            mask_counter += 1
    
    # 적절한 spaCy 모델 선택
    if contains_chinese(masked_sentence):
        nlp_model = nlp_zh
        model_name = "중국어"
    else:
        nlp_model = nlp_ko
        model_name = "한국어"
    
    if not nlp_model:
        print(f"spaCy {model_name} 모델 없음")
        return []
    
    try:
        doc = nlp_model(masked_sentence)
        
        # 문장이 너무 긴 경우 토큰 단위로 의미적 청크 생성
        chunks = []
        current_chunk = ""
        current_length = 0
        
        for token in doc:
            token_text = token.text_with_ws  # 공백 포함
            token_len = len(token_text)
            
            # 현재 청크에 토큰을 추가했을 때 길이 확인
            if current_length + token_len <= max_length:
                current_chunk += token_text
                current_length += token_len
            else:
                # 현재 청크가 있으면 저장
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # 새 청크 시작
                current_chunk = token_text
                current_length = token_len
                
                # 단일 토큰이 max_length를 초과하는 경우
                if current_length > max_length:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_length = 0
        
        # 마지막 청크 추가
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # 마스킹 해제
        restored_chunks = []
        for chunk in chunks:
            restored_chunk = chunk
            for mask_id, original_text in mask_mappings.items():
                restored_chunk = restored_chunk.replace(mask_id, original_text)
            restored_chunks.append(restored_chunk)
        
        # 결과 검증
        if restored_chunks and len(restored_chunks) > 1:
            print(f"spaCy 토큰 기반 분할 (인용문 보호): {len(restored_chunks)}개")
            for i, chunk in enumerate(restored_chunks):
                print(f"  청크 {i+1} ({len(chunk)}자): {chunk}")
            return restored_chunks
        else:
            return []
            
    except Exception as e:
        print(f"spaCy 맥락 분할 오류: {e}")
        return []

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