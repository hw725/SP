"""Prototype02 핵심 로직을 현재 구조에 통합"""
import logging
import numpy as np
import regex
from typing import List, Callable, Dict, Tuple

logger = logging.getLogger(__name__)

class TextMasker:
    """Prototype02의 마스킹 로직 - 복원 문제 해결"""
    
    def __init__(self, **kwargs):
        self.mask_template = '[MASK{}]'
    
    def mask(self, text: str, text_type: str = "source") -> Tuple[str, Dict[str, str]]:
        """Prototype02 방식의 괄호 마스킹 - Dict 반환으로 수정"""
        masks = {}  # *** List -> Dict로 변경 ***
        mask_id = [0]
        
        def mask_content(pattern_str: str) -> str:
            def replacer(match):
                token = self.mask_template.format(mask_id[0])
                masks[token] = match.group()  # *** Dict에 저장 ***
                mask_id[0] += 1
                return token
                
            nonlocal text
            return regex.sub(pattern_str, replacer, text)
        
        if text_type == 'source':
            text = mask_content(r'\([^)]*\)')
        elif text_type == 'target':
            text = mask_content(r'\([^)]*\)')
            text = mask_content(r'\[[^\]]*\]')
        
        return text, masks
    
    def restore(self, text: str, masks: Dict[str, str]) -> str:
        """마스크 복원 - Dict 사용"""
        for mask_token, original in masks.items():
            text = text.replace(mask_token, original)
        return text

# Prototype02 핵심 함수들 (그대로 유지)
def split_src_meaning_units(text: str) -> List[str]:
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

def split_tgt_meaning_units(
    src_units: List[str],
    masked_tgt: str,
    embed_func: Callable,
    source_analyzer=None,  # 호환성 위해 유지하되 사용 안 함
    target_analyzer=None,  # 호환성 위해 유지하되 사용 안 함
    target_tokenizer=None,  # 호환성 위해 유지하되 사용 안 함
    min_tokens: int = 1
) -> List[str]:
    """Prototype02 원본 DP 알고리즘 그대로"""
    tgt_tokens = masked_tgt.split()
    N, T = len(src_units), len(tgt_tokens)
    
    if N == 0 or T == 0:
        return [""] * N if N > 0 else []

    # DP 테이블 초기화 (Prototype02 그대로)
    dp = np.full((N+1, T+1), -np.inf)
    back = np.zeros((N+1, T+1), dtype=int)
    dp[0, 0] = 0.0

    # 원문 임베딩 계산 (한 번만)
    try:
        src_embs = embed_func(src_units)
        if not isinstance(src_embs, np.ndarray):
            src_embs = np.array(src_embs)
    except Exception as e:
        logger.error(f"원문 임베딩 실패: {e}")
        return [""] * N

    # DP 테이블 채우기 (Prototype02 그대로)
    for i in range(1, N+1):
        for j in range(i*min_tokens, min(T+1, T-(N-i)*min_tokens+1)):
            for k in range((i-1)*min_tokens, j-min_tokens+1):
                if k >= j:
                    continue
                    
                span = " ".join(tgt_tokens[k:j])
                if not span.strip():
                    continue
                    
                try:
                    tgt_emb = embed_func([span])
                    if isinstance(tgt_emb, list) and len(tgt_emb) > 0:
                        tgt_emb = tgt_emb[0]
                    elif isinstance(tgt_emb, np.ndarray) and len(tgt_emb) > 0:
                        tgt_emb = tgt_emb[0]
                    
                    # 코사인 유사도 계산
                    src_norm = np.linalg.norm(src_embs[i-1])
                    tgt_norm = np.linalg.norm(tgt_emb)
                    
                    if src_norm > 1e-8 and tgt_norm > 1e-8:
                        sim = float(np.dot(src_embs[i-1], tgt_emb) / (src_norm * tgt_norm))
                    else:
                        sim = 0.0
                    
                    score = dp[i-1, k] + sim
                    if score > dp[i, j]:
                        dp[i, j] = score
                        back[i, j] = k
                        
                except Exception as e:
                    logger.warning(f"임베딩 계산 실패 (span: '{span}'): {e}")
                    continue

    # 역추적 (Prototype02 그대로)
    cuts = [T]
    curr = T
    for i in range(N, 0, -1):
        if curr < back.shape[1] and i < back.shape[0]:
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
            if start_idx < len(tgt_tokens) and end_idx <= len(tgt_tokens):
                span = " ".join(tgt_tokens[start_idx:end_idx]).strip()
                tgt_spans.append(span)
            else:
                tgt_spans.append("")
        else:
            tgt_spans.append("")
    
    return tgt_spans

# 하위 호환성을 위한 함수들
def mask_brackets(text: str, text_type: str = "source") -> Tuple[str, Dict[str, str]]:
    """하위 호환성 함수"""
    masker = TextMasker()
    return masker.mask(text, text_type)

def restore_masks(text: str, masks: Dict[str, str]) -> str:
    """하위 호환성 함수"""
    masker = TextMasker()
    return masker.restore(text, masks)