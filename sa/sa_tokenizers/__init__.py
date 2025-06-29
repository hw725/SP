"""SA 토크나이저 모듈 패키지"""

from .jieba_mecab import (
    split_tgt_meaning_units,  # 번역문(mecab)
    tokenize_text,
    pos_tag_text,
    sentence_split,
    split_tgt_by_src_units_semantic
)
from .bert_tokenizer import split_src_meaning_units as bert_split_src_meaning_units, split_src_sentences as bert_split_src_sentences

def split_src_meaning_units(text: str, *args, **kwargs):
    """원문(한문)은 무조건 jieba로 의미 단위 분할 (tokenizer 인자 무시)"""
    from .jieba_mecab import split_src_meaning_units as jieba_split
    return jieba_split(text, *args, **kwargs)

__all__ = [
    'split_src_meaning_units',
    'split_tgt_meaning_units',
    'split_tgt_by_src_units_semantic',
    'tokenize_text',
    'pos_tag_text',
    'sentence_split'
]