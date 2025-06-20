"""Prototype02 핵심 로직 직접 이식"""
import logging
import numpy as np
import regex
from typing import List, Callable, Dict, Tuple

logger = logging.getLogger(__name__)

# Prototype02 원본 로직 그대로 복사
def split_src_meaning_units_original(text: str) -> List[str]:
    """Prototype02 원본 분할 로직 그대로"""
    # SoyNLP 토크나이저
    try:
        from soynlp.tokenizer import LTokenizer
        tokenizer = LTokenizer()
    except ImportError:
        logger.warning("SoyNLP 없음, 기본 분할 사용")
        tokenizer = None
    
    # 정규식 패턴 (Prototype02 그대로)
    hanja_re = regex.compile(r'\p{Han}+')
    hangul_re = regex.compile(r'^\p{Hangul}+$')
    combined_re = regex.compile(
        r'(\p{Han}+)+(?:\p{Hangul}+)(?:은|는|이|가|을|를|에|에서|으로|로|와|과|도|만|며|고|하고|의|때)?'
    )
    
    text = text.replace('\n', ' ').replace('：', '： ')
    tokens = regex.findall(r'\S+', text)
    units = []
    i = 0

    while i < len(tokens):
        tok = tokens[i]

        # 1) 한자+한글+조사 어미 복합패턴 우선 매칭
        m = combined_re.match(tok)
        if m:
            units.append(m.group(0))
            i += 1
            continue

        # 2) 순수 한자 토큰
        if hanja_re.search(tok):
            unit = tok
            j = i + 1
            # 뒤따르는 순수 한글 토큰이 있으면 묶기
            while j < len(tokens) and hangul_re.match(tokens[j]):
                unit += tokens[j]
                j += 1
            units.append(unit)
            i = j
            continue

        # 3) 순수 한글 토큰: SoyNLP LTokenizer 사용
        if hangul_re.match(tok) and tokenizer:
            korean_tokens = tokenizer.tokenize(tok)
            units.extend(korean_tokens)
            i += 1
            continue

        # 4) 기타 토큰 그대로 보존
        units.append(tok)
        i += 1

    return units

def split_tgt_by_src_units_semantic_original(
    src_units: List[str], 
    tgt_text: str, 
    embed_func: Callable,
    min_tokens: int = 1
) -> List[str]:
    """Prototype02 원본 DP 알고리즘 그대로"""
    tgt_tokens = tgt_text.split()
    N, T = len(src_units), len(tgt_tokens)
    
    if N == 0 or T == 0:
        return []

    # DP 테이블 초기화 (Prototype02 그대로)
    dp = np.full((N+1, T+1), -np.inf)
    back = np.zeros((N+1, T+1), dtype=int)
    dp[0, 0] = 0.0

    # 원문 임베딩 계산 (한 번만)
    try:
        src_embs = embed_func(src_units)
    except Exception as e:
        logger.error(f"원문 임베딩 실패: {e}")
        return [""] * N

    # DP 테이블 채우기 (Prototype02 그대로)
    for i in range(1, N+1):
        for j in range(i*min_tokens, T-(N-i)*min_tokens+1):
            for k in range((i-1)*min_tokens, j-min_tokens+1):
                if k >= j:
                    continue
                    
                span = " ".join(tgt_tokens[k:j])
                if not span.strip():
                    continue
                    
                try:
                    tgt_emb = embed_func([span])[0]
                    
                    # 코사인 유사도 계산
                    sim = float(np.dot(src_embs[i-1], tgt_emb) / 
                               (np.linalg.norm(src_embs[i-1]) * np.linalg.norm(tgt_emb) + 1e-8))
                    
                    score = dp[i-1, k] + sim
                    if score > dp[i, j]:
                        dp[i, j] = score
                        back[i, j] = k
                except Exception as e:
                    logger.warning(f"임베딩 계산 실패: {e}")
                    continue

    # 역추적 (Prototype02 그대로)
    cuts = [T]
    curr = T
    for i in range(N, 0, -1):
        if curr < len(back[0]) and i < len(back):
            prev = int(back[i, curr])
            cuts.append(prev)
            curr = prev
        else:
            cuts.append(0)
    cuts = cuts[::-1]

    # 실제 스팬 구성
    tgt_spans = []
    for i in range(N):
        if i < len(cuts) - 1:
            start_idx = cuts[i]
            end_idx = cuts[i+1]
            span = " ".join(tgt_tokens[start_idx:end_idx]).strip()
            tgt_spans.append(span)
        else:
            tgt_spans.append("")
    
    return tgt_spans

def mask_brackets_original(text: str, text_type: str) -> Tuple[str, List[str]]:
    """Prototype02 원본 마스킹 로직 그대로"""
    masks = []
    mask_id = [0]
    
    def mask_content(pattern_str: str) -> str:
        def replacer(match):
            token = f'[MASK{mask_id[0]}]'
            masks.append(match.group())
            mask_id[0] += 1
            return token
        return regex.sub(pattern_str, replacer, text)
    
    # Prototype02 원본 패턴 그대로
    if text_type == 'source':
        text = mask_content(r'\([^)]*\)')
    elif text_type == 'target':
        text = mask_content(r'\([^)]*\)')
        text = mask_content(r'\[[^\]]*\]')
    
    return text, masks

def restore_masks_original(text: str, masks: List[str]) -> str:
    """Prototype02 원본 복원 로직 그대로"""
    for i, original in enumerate(masks):
        text = text.replace(f'[MASK{i}]', original)
    return text

# 현재 시스템과 호환되는 래퍼 함수들
def split_src_meaning_units(text: str) -> List[str]:
    """현재 시스템 호환용 래퍼"""
    return split_src_meaning_units_original(text)

def split_tgt_meaning_units(
    src_units: List[str],
    masked_tgt: str,
    embed_func: Callable,
    source_analyzer=None,
    target_analyzer=None,
    target_tokenizer=None
) -> List[str]:
    """현재 시스템 호환용 래퍼"""
    return split_tgt_by_src_units_semantic_original(src_units, masked_tgt, embed_func)

class TextMasker:
    """현재 시스템 호환용 래퍼"""
    def __init__(self, **kwargs):
        pass

    def mask(self, text: str, text_type: str = "source"):
        return mask_brackets_original(text, text_type)

    def restore(self, text: str, masks):
        return restore_masks_original(text, masks)