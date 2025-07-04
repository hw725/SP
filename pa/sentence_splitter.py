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

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mecab_jieba_utils import has_speaker_and_speech_verb_ko, has_speech_verb_ko, has_speech_verb_zh, is_korean, is_chinese

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
    """발화동사+인용문 보호를 위한 간단한 분할"""
    
    # 1단계: 기본 종결부호+공백으로 분할
    pattern = r'(?<=[。？！○])|(?<=[.!?])\s+'
    segments = re.split(pattern, text)
    initial_result = [seg.strip() for seg in segments if seg.strip()]
    
    # 2단계: 발화동사로 끝나는 세그먼트와 다음 인용문 세그먼트를 병합하되, 새로운 발화자가 나타나면 분할
    merged_result = []
    i = 0
    
    while i < len(initial_result):
        current_seg = initial_result[i]
        
        # 기존 하드코딩 패턴 제거, mecab/jieba 기반만 사용
        if is_speech_boundary(current_seg):
            is_speech_verb = True
        else:
            is_speech_verb = False
        
        if is_speech_verb:
            # 발화동사 세그먼트
            speaker_text = current_seg
            i += 1
            
            # 인용문들을 연속으로 병합 (새로운 발화자가 나타날 때까지)
            while i < len(initial_result):
                next_seg = initial_result[i]
                
                # 다음 세그먼트가 인용문으로 시작하는지 확인
                if next_seg.strip().startswith(('"', '"', '\u201c', '\u300c', '\u2018')):
                    # 다음 세그먼트에 새로운 발화자가 포함되어 있는지 확인
                    if has_new_speaker(next_seg):
                        # 새로운 발화자가 나타났으므로 현재 그룹을 끝냄
                        # 인용문과 새로운 발화자를 분리
                        quote_part, remaining_part = split_quote_and_speaker(next_seg)
                        if quote_part:
                            speaker_text += ' ' + quote_part
                        
                        # 현재 발화자 그룹을 저장하고 종료
                        merged_result.append(speaker_text)
                        
                        # 남은 부분(새로운 발화자)은 다음 반복에서 처리
                        if remaining_part:
                            initial_result[i] = remaining_part
                        else:
                            i += 1
                        break  # 현재 발화자 그룹 끝
                    else:
                        # 새로운 발화자가 없으면 계속 병합
                        speaker_text += ' ' + next_seg
                        i += 1
                else:
                    # 인용문이 아닌 경우 현재 발화자 그룹 끝
                    merged_result.append(speaker_text)
                    break
            else:
                # while 루프가 정상 종료된 경우 (더 이상 세그먼트가 없음)
                merged_result.append(speaker_text)
        else:
            # 일반 세그먼트 (발화동사가 아닌 경우)
            merged_result.append(current_seg)
            i += 1
    
    # 3단계: 특수 패턴 병합 - 콤마나 닫는 패턴, "라고" 패턴을 앞 문장에 병합
    final_result = []
    for i, seg in enumerate(merged_result):
        # "라고/고 [발화자]" 패턴은 이전 세그먼트에 병합 (일반적인 패턴)
        rago_merge_patterns = [
            r'^(?:라고|고|하고|며)\s+[가-힣]{1,4}(?:公|仲|子|왕|님|씨)?(?:이|가|은|는|께서)?\s*(?:말하[였다가는다]|대답하[였다가는다]|답하[였다가는다]|물[었다어본다]|묻[는다았다])',
        ]
        
        should_merge = False
        if i > 0:
            # 라고 패턴 확인
            for pattern in rago_merge_patterns:
                if re.match(pattern, seg.strip()):
                    should_merge = True
                    break
            
            # 콤마나 닫는 패턴 확인
            if not should_merge and seg.startswith((',', '，', '〉', ')', '】', '』', '」', ':', '：')):
                should_merge = True
        
        if should_merge and final_result:
            final_result[-1] += ' ' + seg
        else:
            final_result.append(seg)
    
    print(f"구두점 분할 결과: {len(final_result)}개 - {final_result}")
    return final_result

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
    """한자가 3자 이하인 세그먼트를 인접한 세그먼트와 병합 (공백 보존)"""
    if not sentences:
        return []
    
    merged = []
    i = 0
    
    while i < len(sentences):
        current_sent = sentences[i]
        han_count = len(regex.findall(r'\p{Han}', current_sent))
        
        if han_count <= 3 and i < len(sentences) - 1:
            # 다음 세그먼트와 병합 (공백 보존)
            next_sent = sentences[i + 1]
            # 현재 세그먼트가 공백으로 끝나지 않고 다음 세그먼트가 공백으로 시작하지 않으면 공백 추가
            if not current_sent.endswith(' ') and not next_sent.startswith(' '):
                merged.append(current_sent + ' ' + next_sent)
            else:
                merged.append(current_sent + next_sent)
            i += 2  # 두 세그먼트를 모두 처리했으므로 2 증가
        elif han_count <= 3 and i > 0:
            # 마지막 세그먼트이고 한자가 적으면 이전 세그먼트에 병합 (공백 보존)
            if not merged[-1].endswith(' ') and not current_sent.startswith(' '):
                merged[-1] += ' ' + current_sent
            else:
                merged[-1] += current_sent
            i += 1
        else:
            # 한자가 많거나 첫 번째 세그먼트인 경우 그대로 추가
            merged.append(current_sent)
            i += 1
    
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
    
    # 원문의 한자 3자 이하 세그먼트 병합과 번역문 동시 병합
    merged_source, merged_target = merge_aligned_segments_by_chinese_count(result, target_sentences)
    
    return merged_source

def merge_aligned_segments_by_chinese_count(source_segments: List[str], target_segments: List[str]) -> Tuple[List[str], List[str]]:
    """원문과 번역문을 동시에 병합 - 한자 3자 이하 규칙 적용"""
    if not source_segments or not target_segments:
        return source_segments, target_segments
    
    # 길이를 맞춤
    min_len = min(len(source_segments), len(target_segments))
    source_segments = source_segments[:min_len]
    target_segments = target_segments[:min_len]
    
    merged_source = []
    merged_target = []
    i = 0
    
    while i < len(source_segments):
        current_src = source_segments[i]
        current_tgt = target_segments[i]
        han_count = len(regex.findall(r'\p{Han}', current_src))
        
        if han_count <= 3 and i < len(source_segments) - 1:
            # 다음 세그먼트와 병합 (원문과 번역문 모두)
            next_src = source_segments[i + 1]
            next_tgt = target_segments[i + 1]
            
            # 원문 병합 (공백 보존)
            if not current_src.endswith(' ') and not next_src.startswith(' '):
                merged_src = current_src + ' ' + next_src
            else:
                merged_src = current_src + next_src
            
            # 번역문 병합 (공백 보존)
            if not current_tgt.endswith(' ') and not next_tgt.startswith(' '):
                merged_tgt = current_tgt + ' ' + next_tgt
            else:
                merged_tgt = current_tgt + next_tgt
            
            merged_source.append(merged_src)
            merged_target.append(merged_tgt)
            i += 2  # 두 세그먼트를 모두 처리했으므로 2 증가
            
        elif han_count <= 3 and i > 0:
            # 마지막 세그먼트이고 한자가 적으면 이전 세그먼트에 병합
            # 원문 병합
            if not merged_source[-1].endswith(' ') and not current_src.startswith(' '):
                merged_source[-1] += ' ' + current_src
            else:
                merged_source[-1] += current_src
            
            # 번역문 병합
            if not merged_target[-1].endswith(' ') and not current_tgt.startswith(' '):
                merged_target[-1] += ' ' + current_tgt
            else:
                merged_target[-1] += current_tgt
            i += 1
            
        else:
            # 한자가 많거나 첫 번째 세그먼트인 경우 그대로 추가
            merged_source.append(current_src)
            merged_target.append(current_tgt)
            i += 1
    
    return [s for s in merged_source if s], [s for s in merged_target if s]

def split_and_align_with_chinese_merge(src_text: str, target_sentences: List[str], embed_func, similarity_threshold: float = 0.5, max_tokens: int = 50) -> Tuple[List[str], List[str]]:
    """
    원문과 번역문을 정렬한 후 한자 3자 이하 규칙으로 동시 병합
    Returns: (aligned_sources, merged_targets)
    """
    # 먼저 기본 정렬 수행
    aligned_src_chunks = split_source_by_whitespace_and_align(src_text, target_sentences, embed_func, similarity_threshold, max_tokens)
    
    # 원문과 번역문을 동시에 병합
    merged_source, merged_target = merge_aligned_segments_by_chinese_count(aligned_src_chunks, target_sentences)
    
    return merged_source, merged_target

def has_new_speaker(text: str) -> bool:
    """텍스트에 새로운 발화자가 포함되어 있는지 확인 - 일반적인 언어학적 패턴 기반"""
    
    # 1. 발화동사 패턴 (확장 가능한 접근)
    speech_verbs = [
        # 기본 발화동사
        r'말하[였다가는다]', r'대답하[였다가는다]', r'답하[였다가는다]', 
        r'물[었다어본다]', r'묻[는다았다]', r'이야기하[였다가는다]',
        # 추가 발화동사들
        r'설명하[였다가는다]', r'외치[였다가는다]', r'소리치[였다가는다]',
        r'속삭이[였다가는다]', r'웃으며\s*말하[였다가는다]', r'울며\s*말하[였다가는다]'
    ]
    
    # 2. 인명 패턴 (더 일반적)
    name_patterns = [
        r'[가-힣]{1,4}(?:公|仲|子|왕|님|씨)?',  # 1-4글자 한국어 이름 + 존칭/호칭
        r'[가-힣]+(?:이|가|은|는|께서)?',      # 이름 + 조사
    ]
    
    # 3. 화법 표지 패턴
    speech_markers = [
        r'(?:라고|고|하고|며|면서)',
        r'(?:라며|하며|면서)',
        r'(?:이라고|라고|다고)'
    ]
    
    # 조합 패턴 생성 (동적)
    combined_patterns = []
    
    # 패턴 1: [이름] + [조사?] + [발화동사]
    for name_pat in name_patterns:
        for verb_pat in speech_verbs:
            combined_patterns.append(f'{name_pat}\\s*{verb_pat}')
    
    # 패턴 2: [화법표지] + [이름] + [발화동사]  
    for marker in speech_markers:
        for name_pat in name_patterns:
            for verb_pat in speech_verbs:
                combined_patterns.append(f'{marker}\\s+{name_pat}\\s*{verb_pat}')
    
    # 패턴 매칭
    for pattern in combined_patterns:
        try:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        except re.error:
            # 잘못된 정규식 패턴은 무시
            continue
    
    return False

def split_quote_and_speaker(text: str) -> tuple:
    """인용문과 새로운 발화자 부분을 분리 - 일반적인 접근"""
    text = text.strip()
    
    # 여는 따옴표로 시작하는지 확인
    opening_quotes = ['"', '"', '\u201c', '\u300c', '\u2018', "'", "'"]
    if not any(text.startswith(q) for q in opening_quotes):
        return "", text
    
    # 따옴표 쌍 매핑 (더 포괄적)
    quote_pairs = {
        '"': ['"', '"'],  # 일반 따옴표는 여러 닫는 형태 가능
        '"': ['"'],
        '\u201c': ['\u201d'], 
        '\u300c': ['\u300d'],
        '\u2018': ['\u2019'],
        "'": ["'"],
        "'": ["'"]
    }
    
    opening = text[0]
    possible_closings = quote_pairs.get(opening, [opening])
    
    # 가장 가까운 닫는 따옴표 찾기
    close_pos = -1
    for closing in possible_closings:
        pos = text.find(closing, 1)
        if pos != -1 and (close_pos == -1 or pos < close_pos):
            close_pos = pos
    
    if close_pos != -1:
        quote_part = text[:close_pos + 1]
        remaining_part = text[close_pos + 1:].strip()
        return quote_part, remaining_part
    else:
        # 닫는 따옴표가 없으면 발화자 패턴으로 분리
        # has_new_speaker와 동일한 로직 사용하되 위치 찾기
        return find_speaker_split_position(text)

def find_speaker_split_position(text: str) -> tuple:
    """발화자 패턴을 찾아서 분리 위치 결정"""
    
    # 발화동사 패턴
    speech_verbs = [
        r'말하[였다가는다]', r'대답하[였다가는다]', r'답하[였다가는다]', 
        r'물[었다어본다]', r'묻[는다았다]', r'이야기하[였다가는다]',
        r'설명하[였다가는다]', r'외치[였다가는다]', r'소리치[였다가는다]',
        r'속삭이[였다가는다]', r'웃으며\s*말하[였다가는다]', r'울며\s*말하[였다가는다]'
    ]
    
    # 인명 패턴
    name_patterns = [
        r'[가-힣]{1,4}(?:公|仲|子|왕|님|씨)?(?:이|가|은|는|께서)?'
    ]
    
    # 화법 표지 패턴
    speech_markers = [r'(?:라고|고|하고|며|면서)', r'(?:라며|하며|면서)', r'(?:이라고|라고|다고)']
    
    # 발화자 패턴들을 조합해서 찾기
    all_patterns = []
    
    # [이름] + [발화동사] 패턴
    for name_pat in name_patterns:
        for verb_pat in speech_verbs:
            all_patterns.append(f'({name_pat}\\s*{verb_pat})')
    
    # [화법표지] + [이름] + [발화동사] 패턴
    for marker in speech_markers:
        for name_pat in name_patterns:
            for verb_pat in speech_verbs:
                all_patterns.append(f'({marker}\\s+{name_pat}\\s*{verb_pat})')
    
    # 가장 먼저 나타나는 패턴 찾기
    earliest_match = None
    earliest_pos = len(text)
    
    for pattern in all_patterns:
        try:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and match.start() < earliest_pos:
                earliest_match = match
                earliest_pos = match.start()
        except re.error:
            continue
    
    if earliest_match:
        split_pos = earliest_match.start()
        quote_part = text[:split_pos].strip()
        remaining_part = text[split_pos:].strip()
        return quote_part, remaining_part
    else:
        # 발화자 패턴이 없으면 전체를 인용문으로 간주
        return text, ""

def is_speech_boundary(text):
    """mecab/jieba 기반 발화동사/주어/부사 패턴 탐지 (한국어/중국어 모두)"""
    if is_korean(text):
        return has_speaker_and_speech_verb_ko(text) or has_speech_verb_ko(text)
    elif is_chinese(text):
        return has_speech_verb_zh(text)
    return False