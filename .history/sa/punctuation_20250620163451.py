"""괄호 및 구두점 처리 모듈 - regex 지원"""

import regex  # 🆕 유니코드 속성 정규식
from typing import List, Tuple

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

# 별칭 함수 (하위 호환성)
restore_masks = restore_brackets