"""고전 한문 BERT 계열 토크나이저 래퍼 - 의존성 충돌로 사용하지 않음"""
from typing import List, Optional
from transformers import AutoTokenizer

DEFAULT_BERT_MODEL = "hfl/chinese-roberta-wwm-ext"

class BertTokenizerWrapper:
    def __init__(self, model_name: str = DEFAULT_BERT_MODEL):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(self, text: str, by_space: bool = False) -> List[str]:
        """
        by_space=True: 공백 단위 토큰화(띄어쓰기 기준),
        by_space=False: BERT subword 토큰화
        """
        if by_space:
            return [tok for tok in text.strip().split() if tok]
        return self.tokenizer.tokenize(text)

    def split_sentences(self, text: str) -> List[str]:
        # 문장 단위 분할(간단히 마침표 등으로 분할, 필요시 개선)
        import re
        sents = re.split(r'[。！？.!?\n]', text)
        return [s.strip() for s in sents if s.strip()]

# 전역 인스턴스 (기본 모델)
_bert_tokenizer = None

def split_src_meaning_units(text: str, model: str = DEFAULT_BERT_MODEL, by_space: bool = False) -> List[str]:
    global _bert_tokenizer
    if _bert_tokenizer is None or _bert_tokenizer.model_name != model:
        _bert_tokenizer = BertTokenizerWrapper(model_name=model)
    return _bert_tokenizer.tokenize(text, by_space=by_space)

def split_src_sentences(text: str, model: str = DEFAULT_BERT_MODEL) -> List[str]:
    global _bert_tokenizer
    if _bert_tokenizer is None or _bert_tokenizer.model_name != model:
        _bert_tokenizer = BertTokenizerWrapper(model_name=model)
    return _bert_tokenizer.split_sentences(text)
