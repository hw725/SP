# punctuation.py

import regex as re
from typing import List, Tuple

# 마스킹 토큰 템플릿
MASK_TEMPLATE = '[MASK{}]'

# 괄호 정의
HALF_WIDTH_BRACKETS = [  # 반각 괄호
    ('(', ')'),
    ('[', ']'),
]
FULL_WIDTH_BRACKETS = [  # 전각 괄호
    ('（', '）'),
    ('［', '］'),
]
TRANS_BRACKETS = [       # 번역문 전용 괄호
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

# 전체 괄호 집합
ALL_BRACKETS = HALF_WIDTH_BRACKETS + FULL_WIDTH_BRACKETS + TRANS_BRACKETS


def mask_brackets(text: str, text_type: str) -> Tuple[str, List[str]]:
    """
    괄호 안의 내용을 규칙에 따라 마스킹한다.

    이미 [MASK...]로 치환된 부분은 절대 다시 마스킹 처리하지 않는다.

    Args:
        text (str): 입력 문자열
        text_type (str): 'source' 또는 'target'

    Returns:
        Tuple[str, List[str]]: 마스킹된 문자열, 마스킹된 내용 목록
    """
    assert text_type in {'source', 'target'}, "text_type must be 'source' or 'target'"

    masks: List[str] = []
    mask_id = [0]

    # 이미 마스킹된 토큰([] 또는 [MASK0], [MASK1] ...)은 한 번만 마스킹 되게 한다.
    def safe_sub(pattern, repl, s):
        # 이미 [MASK]가 포함된 문자열에는 패턴 적용하지 않음
        def safe_replacer(m):
            if '[MASK' in m.group(0):
                return m.group(0)
            return repl(m)
        return pattern.sub(safe_replacer, s)

    # 괄호별 규칙에 따라 정규식 패턴 구성
    patterns: List[Tuple[re.Pattern, bool]] = []

    if text_type == 'source':
        for left, right in HALF_WIDTH_BRACKETS:
            patterns.append((re.compile(re.escape(left) + r'[^' + re.escape(left + right) + r']*?' + re.escape(right)), True))
        for left, right in FULL_WIDTH_BRACKETS:
            patterns.append((re.compile(re.escape(left)), False))
            patterns.append((re.compile(re.escape(right)), False))
    elif text_type == 'target':
        for left, right in HALF_WIDTH_BRACKETS + FULL_WIDTH_BRACKETS:
            patterns.append((re.compile(re.escape(left) + r'[^' + re.escape(left + right) + r']*?' + re.escape(right)), True))
        for left, right in TRANS_BRACKETS:
            patterns.append((re.compile(re.escape(left)), False))
            patterns.append((re.compile(re.escape(right)), False))

    def mask_content(s: str, pattern: re.Pattern, content_mask: bool) -> str:
        def replacer(match: re.Match) -> str:
            token = MASK_TEMPLATE.format(mask_id[0])
            masks.append(match.group())
            mask_id[0] += 1
            return token
        # 이미 [MASK...]가 있으면 그 부분은 손대지 않음
        return safe_sub(pattern, replacer, s)

    # 내용 마스킹 패턴 먼저 적용
    for pattern, content_mask in patterns:
        if content_mask:
            text = mask_content(text, pattern, content_mask)
    # 괄호만 마스킹
    for pattern, content_mask in patterns:
        if not content_mask:
            text = mask_content(text, pattern, content_mask)

    return text, masks


def restore_masks(text: str, masks: List[str]) -> str:
    """
    마스킹된 토큰을 원래 내용으로 복원

    Args:
        text (str): 마스킹된 문자열
        masks (List[str]): 마스크된 원본 내용

    Returns:
        str: 복원된 문자열
    """
    for i, original in enumerate(masks):
        text = text.replace(MASK_TEMPLATE.format(i), original)
    return text
