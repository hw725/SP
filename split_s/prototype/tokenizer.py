# tokenizer.py - 원문/번역문을 구 단위로 분할하는 토크나이저 모듈

import regex as re
from typing import List

def split_src_meaning_units(text: str) -> List[str]:
    """
    원문(한문+한국어 조사)을 의미 단위로 분할해서 반환.
    - \p{Han} (한자) 뒤에 연달아 나오는 한글 토큰을 붙여서 하나의 의미 단위로 묶음.
    """
    # 줄바꿈 제거, 콜론 뒤 공백 처리
    text = text.replace('\n', ' ').replace('：', '： ')
    # 공백 기준으로 우선 토큰화
    tokens = re.findall(r"\S+", text)
    hanja_pattern = r'\p{Han}+'
    units = []
    i = 0

    while i < len(tokens):
        token = tokens[i]
        if re.search(hanja_pattern, token):
            # 한자 포함 토큰: 뒤에 연속된 한글만 붙여준다
            unit = token
            j = i + 1
            while j < len(tokens) and re.match(r'^\p{Hangul}+$', tokens[j]):
                unit += tokens[j]
                j += 1
            units.append(unit)
            i = j
        else:
            # 한자가 없으면 그냥 그대로
            units.append(token)
            i += 1

    return units


def split_tgt_meaning_units(text: str) -> List[str]:
    """
    번역문(현대어)을 공백 기준으로 토큰화해서 반환.
    - 문장 부호(콤마, 마침표) 등이 붙어 있으면 그대로 단어에 포함됨.
    """
    text = text.replace('\n', ' ').replace('：', '： ')
    return re.findall(r"\S+", text)