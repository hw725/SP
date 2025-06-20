"""Prototype02의 실제 코드를 클래스 구조에 완전 통합"""
import logging
import numpy as np
import regex
from typing import List, Callable, Dict, Tuple
from soynlp.tokenizer import LTokenizer

logger = logging.getLogger(__name__)

# Prototype02 상수들 그대로 복사
MASK_TEMPLATE = '[MASK{}]'
DEFAULT_MIN_TOKENS = 1

class TextMasker:
    """Prototype02 punctuation.py의 mask_brackets/restore_masks를 클래스로 통합"""
    
    def __init__(self, **kwargs):
        # Prototype02 괄호 정의 그대로
        self.HALF_WIDTH_BRACKETS = [('(', ')'), ('[', ']')]
        self.FULL_WIDTH_BRACKETS = [('（', '）'), ('［', '］')]
        self.TRANS_BRACKETS = [
            ('<', '>'), ('《', '》'), ('〈', '〉'), ('「', '」'), ('『', '』'),
            ('〔', '〕'), ('【', '】'), ('〖', '〗'), ('〘', '〙'), ('〚', '〛')
        ]
    
    def mask(self, text: str, text_type: str) -> Tuple[str, List[str]]:
        """Prototype02 mask_brackets 함수 그대로"""
        assert text_type in {'source', 'target'}, "text_type must be 'source' or 'target'"

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
            for left, right in self.HALF_WIDTH_BRACKETS:
                patterns.append((regex.compile(regex.escape(left) + r'[^' + regex.escape(left + right) + r']*?' + regex.escape(right)), True))
            for left, right in self.FULL_WIDTH_BRACKETS:
                patterns.append((regex.compile(regex.escape(left)), False))
                patterns.append((regex.compile(regex.escape(right)), False))
        elif text_type == 'target':
            for left, right in self.HALF_WIDTH_BRACKETS + self.FULL_WIDTH_BRACKETS:
                patterns.append((regex.compile(regex.escape(left) + r'[^' + regex.escape(left + right) + r']*?' + regex.escape(right)), True))
            for left, right in self.TRANS_BRACKETS:
                patterns.append((regex.compile(regex.escape(left)), False))
                patterns.append((regex.compile(regex.escape(right)), False))

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
    
    def restore(self, text: str, masks: List[str]) -> str:
        """Prototype02 restore_masks 함수 그대로"""
        for i, original in enumerate(masks):
            text = text.replace(MASK_TEMPLATE.format(i), original)
        return text

class SourceTextSplitter:
    """Prototype02 tokenizer.py의 split_src_meaning_units를 클래스로 통합"""
    
    def __init__(self, **kwargs):
        # Prototype02 토크나이저 초기화 그대로
        self.tokenizer = LTokenizer()
        
        # Prototype02 정규식 그대로
        self.hanja_re = regex.compile(r'\p{Han}+')
        self.hangul_re = regex.compile(r'^\p{Hangul}+$')
        self.combined_re = regex.compile(
            r'(\p{Han}+)+(?:\p{Hangul}+)(?:은|는|이|가|을|를|에|에서|으로|로|와|과|도|만|며|고|하고|의|때)?'
        )
    
    def split(self, text: str) -> List[str]:
        """Prototype02 split_src_meaning_units 함수 그대로"""
        text = text.replace('\n', ' ').replace('：', '： ')
        tokens = regex.findall(r'\S+', text)
        units: List[str] = []
        i = 0

        while i < len(tokens):
            tok = tokens[i]

            # 1) 한자+한글+조사 어미 복합패턴 우선 매칭
            m = self.combined_re.match(tok)
            if m:
                units.append(m.group(0))
                i += 1
                continue

            # 2) 순수 한자 토큰
            if self.hanja_re.search(tok):
                unit = tok
                j = i + 1
                # 뒤따르는 순수 한글 토큰이 있으면 묶기
                while j < len(tokens) and self.hangul_re.match(tokens[j]):
                    unit += tokens[j]
                    j += 1
                units.append(unit)
                i = j
                continue

            # 3) 순수 한글 토큰: SoyNLP LTokenizer 사용
            if self.hangul_re.match(tok):
                korean_tokens = self.tokenizer.tokenize(tok)
                units.extend(korean_tokens)
                i += 1
                continue

            # 4) 기타 토큰 (숫자, 로마자 등) 그대로 보존
            units.append(tok)
            i += 1

        return units

class TargetTextAligner:
    """Prototype02 tokenizer.py의 split_tgt_by_src_units_semantic을 클래스로 통합"""
    
    def __init__(self, min_tokens: int = DEFAULT_MIN_TOKENS, **kwargs):
        self.min_tokens = min_tokens
    
    def align(self, src_units: List[str], tgt_text: str, embed_func: Callable) -> List[str]:
        """Prototype02 split_tgt_by_src_units_semantic 함수 그대로"""
        tgt_tokens = tgt_text.split()
        N, T = len(src_units), len(tgt_tokens)
        if N == 0 or T == 0:
            return []

        dp = np.full((N+1, T+1), -np.inf)
        back = np.zeros((N+1, T+1), dtype=int)
        dp[0, 0] = 0.0

        # 원문 임베딩 계산 (Prototype02 그대로)
        src_embs = embed_func(src_units)

        # DP 테이블 채우기 (Prototype02 그대로)
        for i in range(1, N+1):
            for j in range(i*self.min_tokens, T-(N-i)*self.min_tokens+1):
                for k in range((i-1)*self.min_tokens, j-self.min_tokens+1):
                    span = " ".join(tgt_tokens[k:j])
                    tgt_emb = embed_func([span])[0]
                    sim = float(np.dot(src_embs[i-1], tgt_emb)/((np.linalg.norm(src_embs[i-1])*np.linalg.norm(tgt_emb))+1e-8))
                    score = dp[i-1, k] + sim
                    if score > dp[i, j]:
                        dp[i, j] = score
                        back[i, j] = k

        # Traceback (Prototype02 그대로)
        cuts = [T]
        curr = T
        for i in range(N, 0, -1):
            prev = int(back[i, curr])
            cuts.append(prev)
            curr = prev
        cuts = cuts[::-1]
        assert cuts[0] == 0 and cuts[-1] == T and len(cuts) == N + 1

        # Build actual spans (Prototype02 그대로)
        tgt_spans = []
        for i in range(N):
            span = " ".join(tgt_tokens[cuts[i]:cuts[i+1]]).strip()
            tgt_spans.append(span)
        return tgt_spans

class TextAligner:
    """Prototype02 aligner.py의 align_src_tgt를 클래스로 통합"""
    
    def __init__(self, **kwargs):
        pass
    
    def align(self, src_units: List[str], tgt_units: List[str], embed_func: Callable) -> List[Tuple[str, str]]:
        """Prototype02 align_src_tgt 함수 그대로"""
        logger.info(f"Source units: {len(src_units)} items, Target units: {len(tgt_units)} items")

        if len(src_units) != len(tgt_units):
            try:
                # Prototype02 재정렬 로직 그대로
                target_aligner = TargetTextAligner(min_tokens=1)
                flatten_tgt = " ".join(tgt_units)
                new_tgt_units = target_aligner.align(src_units, flatten_tgt, embed_func)
                
                if len(new_tgt_units) == len(src_units):
                    logger.info("Semantic re-alignment successful")
                    return list(zip(src_units, new_tgt_units))
                else:
                    logger.warning(f"Length mismatch after re-alignment: Source={len(src_units)}, Target={len(new_tgt_units)}")
            except Exception as e:
                logger.error(f"Error during semantic re-alignment: {e}")

            # 길이가 맞지 않으면 패딩 (Prototype02 그대로)
            if len(src_units) > len(tgt_units):
                tgt_units.extend([""] * (len(src_units) - len(tgt_units)))
            else:
                src_units.extend([""] * (len(tgt_units) - len(src_units)))

        return list(zip(src_units, tgt_units))

# *** 현재 시스템과의 호환성을 위한 함수형 인터페이스 ***
def split_src_meaning_units(text: str) -> List[str]:
    """현재 시스템 호환용"""
    splitter = SourceTextSplitter()
    return splitter.split(text)

def split_tgt_meaning_units(
    src_units: List[str],
    masked_tgt: str,
    embed_func: Callable,
    source_analyzer=None,
    target_analyzer=None, 
    target_tokenizer=None,
    min_tokens: int = DEFAULT_MIN_TOKENS
) -> List[str]:
    """현재 시스템 호환용"""
    aligner = TargetTextAligner(min_tokens=min_tokens)
    return aligner.align(src_units, masked_tgt, embed_func)

def mask_brackets(text: str, text_type: str = "source") -> Tuple[str, List[str]]:
    """현재 시스템 호환용"""
    masker = TextMasker()
    return masker.mask(text, text_type)

def restore_masks(text: str, masks: List[str]) -> str:
    """현재 시스템 호환용"""
    masker = TextMasker()
    return masker.restore(text, masks)