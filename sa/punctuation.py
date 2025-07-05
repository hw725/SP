"""괄호 및 구두점 처리 모듈"""

import logging
import regex  # 🆕 유니코드 속성 정규식
import re  # 괄호 추출 및 패턴 컴파일용
import numpy as np  # 임베딩 계산용
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# 마스킹 템플릿
MASK_TEMPLATE = '[MASK{}]'

# 괄호 종류별 분류
HALF_WIDTH_BRACKETS = [
    ('(', ')'),
    ('[', ']'),
]

FULL_WIDTH_BRACKETS = [
    ('（', '）'),
    ('［', '］'),
]

TRANS_BRACKETS = [
    ('<', '>'),
    ('《', '》'),
    ('〈', '〉'),
    ('「', '」'),
    ('『', '』'),
    ('〔', '〕'),
    ('【', '】'),
    ('〖', '〗'),
    ('〘', '〙'),
    ('〚', '〛'),
]

def mask_brackets(text: str, text_type: str) -> Tuple[str, List[str]]:
    """Mask content within brackets according to rules."""
    if text_type not in {'source', 'target'}:
        raise ValueError("text_type must be 'source' or 'target'")

    masks: List[str] = []
    mask_id = [0]

    def safe_sub(pattern, repl, s):
        def safe_replacer(m):
            if '[MASK' in m.group(0):
                return m.group(0)
            return repl(m)
        return pattern.sub(safe_replacer, s)

    patterns: List[Tuple[regex.Pattern, bool]] = []

    if text_type == 'source':
        for left, right in HALF_WIDTH_BRACKETS:
            patterns.append((regex.compile(re.escape(left) + r'[^' + re.escape(left + right) + r']*?' + re.escape(right)), True))
        for left, right in FULL_WIDTH_BRACKETS:
            patterns.append((regex.compile(re.escape(left)), False))
            patterns.append((regex.compile(re.escape(right)), False))
    elif text_type == 'target':
        for left, right in HALF_WIDTH_BRACKETS + FULL_WIDTH_BRACKETS:
            patterns.append((regex.compile(re.escape(left) + r'[^' + re.escape(left + right) + r']*?' + re.escape(right)), True))
        for left, right in TRANS_BRACKETS:
            patterns.append((regex.compile(re.escape(left)), False))
            patterns.append((regex.compile(re.escape(right)), False))

    def mask_content(s: str, pattern: regex.Pattern, content_mask: bool) -> str:
        def replacer(match: regex.Match) -> str:
            token = MASK_TEMPLATE.format(mask_id[0])
            masks.append(match.group())
            mask_id[0] += 1
            return token
        return safe_sub(pattern, replacer, s)

    for pattern, content_mask in patterns:
        if content_mask:
            text = mask_content(text, pattern, content_mask)
    for pattern, content_mask in patterns:
        if not content_mask:
            text = mask_content(text, pattern, content_mask)

    return text, masks

def restore_brackets(text: str, masks: List[str]) -> str:
    """Restore masked tokens to their original content."""
    for i, original in enumerate(masks):
        text = text.replace(MASK_TEMPLATE.format(i), original)
    return text

def extract_punctuation_with_han(text: str) -> Tuple[List[str], List[int]]:
    """한자/한글 구두점 경계 추출 (전각 콜론 등 경계 강화, 기존 기능 보존)"""
    # 1. 반각 종결부호+공백 기준 분할(따옴표 문제 방지)
    # 2. 전각 콜론(：)은 공백 없이도 경계로 처리
    result = []
    positions = []
    # 1단계: 반각 종결부호+공백 기준 분할
    pattern1 = r'([.!?])\s+'
    parts = regex.split(pattern1, text)
    temp = []
    for i in range(0, len(parts)-1, 2):
        temp.append(parts[i] + parts[i+1])
    if len(parts) % 2 == 1 and parts[-1]:
        temp.append(parts[-1])
    # 2단계: 각 파트에서 전각 콜론(：) 기준 추가 분할
    for part in temp:
        subparts = regex.split(r'(：)', part)
        for i in range(0, len(subparts)-1, 2):
            seg = subparts[i] + subparts[i+1]
            result.append(seg)
            positions.append(text.find(seg))
        if len(subparts) % 2 == 1 and subparts[-1]:
            result.append(subparts[-1])
            positions.append(text.find(subparts[-1]))
    return result, positions

def is_han_punctuation(char: str) -> bool:
    """한자 구두점 여부 - 전각 콜론은 분할 경계로 사용하므로 제외"""
    han_punctuation = ['。', '！', '？', '，', '；']  # '：' 제거
    return char in han_punctuation

def is_hangul_boundary(text: str, pos: int) -> bool:
    """한글 경계 여부"""
    if pos >= len(text):
        return False
    
    return bool(regex.match(r'\p{Hangul}', text[pos]))

def process_punctuation(alignments: List[Dict[str, Any]], src_units: List[str], tgt_units: List[str]) -> List[Dict[str, Any]]:
    """괄호 및 구두점 처리 - processor.py 호환용
    
    정렬된 원문-번역문 쌍들을 받아서 괄호 매칭 정보를 추가하는 후처리 함수
    
    Args:
        alignments: 정렬 결과 리스트 [{'src_idx': int, 'tgt_idx': int, 'src': str, 'tgt': str, 'score': float}]
        src_units: 원문 의미 단위 리스트 (현재 미사용)
        tgt_units: 번역문 의미 단위 리스트 (현재 미사용)
    
    Returns:
        괄호 매칭 정보가 추가된 정렬 결과 리스트
    """
    
    if not alignments:
        return alignments
    
    # 괄호 매칭 분석을 통해 정렬 결과에 메타데이터 추가
    return process_bracket_alignments(alignments, src_units, tgt_units)

def process_bracket_alignments(alignments: List[Dict[str, Any]], src_units: List[str], tgt_units: List[str]) -> List[Dict[str, Any]]:
    """괄호 정렬 처리"""
    
    processed_alignments = []
    
    for alignment in alignments:
        # 기본적으로 그대로 유지
        processed_alignment = alignment.copy()
        
        # 괄호 처리 로직 (필요시 구현)
        src_text = alignment.get('src', '')
        tgt_text = alignment.get('tgt', '')
        
        # 괄호 쌍 매칭
        if '(' in src_text and ')' in src_text:
            if '<' in tgt_text and '>' in tgt_text:
                processed_alignment['bracket_type'] = 'matched'
            else:
                processed_alignment['bracket_type'] = 'unmatched'
        
        processed_alignments.append(processed_alignment)
    
    return processed_alignments

def handle_parentheses(text: str) -> str:
    """괄호 처리"""
    # 기본 괄호 정규화
    text = text.replace('（', '(').replace('）', ')')
    text = text.replace('〈', '<').replace('〉', '>')
    return text

def extract_brackets(text: str) -> List[str]:
    """괄호 내용 추출"""
    
    brackets = []
    
    # 소괄호 ()
    paren_matches = re.findall(r'\(([^)]+)\)', text)
    brackets.extend(paren_matches)
    
    # 꺾쇠괄호 <>
    angle_matches = re.findall(r'<([^>]+)>', text)
    brackets.extend(angle_matches)
    
    # 대괄호 []
    square_matches = re.findall(r'\[([^\]]+)\]', text)
    brackets.extend(square_matches)
    
    return brackets

# 별칭 함수 (하위 호환성)
restore_masks = restore_brackets

# 모듈 export 함수 목록
__all__ = [
    'mask_brackets',
    'restore_brackets', 
    'process_punctuation'
]

if __name__ == "__main__":
    # 테스트
    test_alignments = [
        {'src': '興也라', 'tgt': '興이다.', 'score': 0.9},
        {'src': '蒹은 薕(렴)이요', 'tgt': '蒹은 물억새<라고>이고', 'score': 0.8}
    ]
    
    test_src = ['興也라', '蒹은 薕(렴)이요']
    test_tgt = ['興이다.', '蒹은 물억새<라고>이고']
    
    result = process_punctuation(test_alignments, test_src, test_tgt)
    
    print("괄호 처리 테스트:")
    for r in result:
        print(f"  {r}")