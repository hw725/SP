"""괄호 및 구두점 처리 모듈"""

import logging
import regex  # 🆕 유니코드 속성 정규식
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
    """한자 고려한 구두점 추출"""
    
    # 🆕 한자/한글 구두점 패턴
    pattern = r'([\p{Han}\p{Hangul}]*[.!?。！？,，;：:]+[\p{Han}\p{Hangul}]*)'
    
    matches = list(regex.finditer(pattern, text))
    punctuations = [match.group() for match in matches]
    positions = [match.start() for match in matches]
    
    return punctuations, positions

def is_han_punctuation(char: str) -> bool:
    """한자 구두점 여부"""
    han_punctuation = ['。', '！', '？', '，', '：', '；']
    return char in han_punctuation

def is_hangul_boundary(text: str, pos: int) -> bool:
    """한글 경계 여부"""
    if pos >= len(text):
        return False
    
    return bool(regex.match(r'\p{Hangul}', text[pos]))

def process_punctuation(alignments: List[Dict[str, Any]], src_units: List[str], tgt_units: List[str]) -> List[Dict[str, Any]]:
    """괄호 및 구두점 처리 - processor.py 호환용"""
    
    if not alignments:
        return alignments
    
    try:
        # 기존 함수가 있다면 그것을 활용, 없다면 기본 처리
        return process_bracket_alignments(alignments, src_units, tgt_units)
    except NameError:
        # process_bracket_alignments 함수가 없다면 기본 반환
        logger.warning("⚠️ process_bracket_alignments 함수가 없습니다. 기본 처리합니다.")
        return alignments

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
    import re
    
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