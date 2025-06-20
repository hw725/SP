"""Prototype02 완전 통합 - 누락 부분 보완"""
import logging
import numpy as np
import regex
from typing import List, Callable, Dict, Tuple, Optional
import os

logger = logging.getLogger(__name__)

# SoyNLP 안전 임포트
try:
    from soynlp.tokenizer import LTokenizer
    SOYNLP_AVAILABLE = True
except ImportError:
    logger.warning("SoyNLP를 찾을 수 없습니다. 기본 토크나이저를 사용합니다.")
    SOYNLP_AVAILABLE = False
    LTokenizer = None

class TextMasker:
    """Prototype02 punctuation.py 완전 복제"""
    
    def __init__(self, **kwargs):
        # Prototype02 괄호 정의 완전 복사
        self.HALF_WIDTH_BRACKETS = [('(', ')'), ('[', ']')]
        self.FULL_WIDTH_BRACKETS = [('（', '）'), ('［', '］')]
        self.TRANS_BRACKETS = [
            ('<', '>'), ('《', '》'), ('〈', '〉'), ('「', '」'), ('『', '』'),
            ('〔', '〕'), ('【', '】'), ('〖', '〗'), ('〘', '〙'), ('〚', '〛')
        ]
        self.MASK_TEMPLATE = '[MASK{}]'
    
    def mask(self, text: str, text_type: str) -> Tuple[str, List[str]]:
        """Prototype02 mask_brackets 함수 완전 복사"""
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
            for left, right in self.HALF_WIDTH_BRACKETS:
                pattern = regex.compile(regex.escape(left) + r'[^' + regex.escape(left + right) + r']*?' + regex.escape(right))
                patterns.append((pattern, True))
            for left, right in self.FULL_WIDTH_BRACKETS:
                patterns.append((regex.compile(regex.escape(left)), False))
                patterns.append((regex.compile(regex.escape(right)), False))
                
        elif text_type == 'target':
            for left, right in self.HALF_WIDTH_BRACKETS + self.FULL_WIDTH_BRACKETS:
                pattern = regex.compile(regex.escape(left) + r'[^' + regex.escape(left + right) + r']*?' + regex.escape(right))
                patterns.append((pattern, True))
            for left, right in self.TRANS_BRACKETS:
                patterns.append((regex.compile(regex.escape(left)), False))
                patterns.append((regex.compile(regex.escape(right)), False))

        def mask_content(s: str, pattern: regex.Pattern, content_mask: bool) -> str:
            def replacer(match: regex.Match) -> str:
                token = self.MASK_TEMPLATE.format(mask_id[0])
                masks.append(match.group())
                mask_id[0] += 1
                return token
            return safe_sub(pattern, replacer, s)

        # Prototype02 마스킹 순서 보장
        for pattern, content_mask in patterns:
            if content_mask:
                text = mask_content(text, pattern, content_mask)
        for pattern, content_mask in patterns:
            if not content_mask:
                text = mask_content(text, pattern, content_mask)

        return text, masks
    
    def restore(self, text: str, masks: List[str]) -> str:
        """Prototype02 restore_masks 함수 완전 복사"""
        for i, original in enumerate(masks):
            text = text.replace(self.MASK_TEMPLATE.format(i), original)
        return text

class SourceTextSplitter:
    """Prototype02 tokenizer.py split_src_meaning_units 완전 복제"""
    
    def __init__(self, **kwargs):
        # SoyNLP 토크나이저 초기화
        self.tokenizer = None
        if SOYNLP_AVAILABLE:
            try:
                self.tokenizer = LTokenizer()
                logger.debug("SoyNLP LTokenizer 초기화 성공")
            except Exception as e:
                logger.warning(f"SoyNLP LTokenizer 초기화 실패: {e}")
        
        # Prototype02 정규식 완전 복사
        self.hanja_re = regex.compile(r'\p{Han}+')
        self.hangul_re = regex.compile(r'^\p{Hangul}+$')
        self.combined_re = regex.compile(
            r'(\p{Han}+)+(?:\p{Hangul}+)(?:은|는|이|가|을|를|에|에서|으로|로|와|과|도|만|며|고|하고|의|때)?'
        )
    
    def split(self, text: str) -> List[str]:
        """Prototype02 split_src_meaning_units 완전 복사"""
        if not text or not text.strip():
            return []
            
        # Prototype02 전처리 완전 복사
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
            if self.hangul_re.match(tok) and self.tokenizer:
                try:
                    korean_tokens = self.tokenizer.tokenize(tok)
                    if korean_tokens:
                        units.extend(korean_tokens)
                    else:
                        units.append(tok)
                except Exception as e:
                    logger.debug(f"SoyNLP 토크나이징 실패: {e}")
                    units.append(tok)
                i += 1
                continue

            # 4) 기타 토큰 그대로 보존
            units.append(tok)
            i += 1

        return units

class TargetTextAligner:
    """Prototype02 tokenizer.py split_tgt_by_src_units_semantic 완전 복제"""
    
    def __init__(self, min_tokens: int = 1, **kwargs):
        self.min_tokens = max(1, min_tokens)  # 최소값 보장
    
    def align(self, src_units: List[str], tgt_text: str, embed_func: Callable) -> List[str]:
        """Prototype02 split_tgt_by_src_units_semantic 완전 복사"""
        if not src_units or not tgt_text or not tgt_text.strip():
            return [""] * len(src_units) if src_units else []
            
        tgt_tokens = tgt_text.split()
        N, T = len(src_units), len(tgt_tokens)
        
        if N == 0 or T == 0:
            return [""] * N

        # DP 테이블 초기화 (Prototype02 완전 복사)
        dp = np.full((N+1, T+1), -np.inf, dtype=np.float64)
        back = np.zeros((N+1, T+1), dtype=np.int32)
        dp[0, 0] = 0.0

        # 원문 임베딩 계산
        try:
            src_embs = embed_func(src_units)
            if isinstance(src_embs, list):
                src_embs = np.array(src_embs, dtype=np.float32)
            logger.debug(f"원문 임베딩 완료: {src_embs.shape}")
        except Exception as e:
            logger.error(f"원문 임베딩 실패: {e}")
            return [""] * N

        # DP 테이블 채우기 (Prototype02 완전 복사)
        for i in range(1, N+1):
            start_j = max(i * self.min_tokens, 1)
            end_j = min(T + 1, T - (N - i) * self.min_tokens + 1)
            
            for j in range(start_j, end_j):
                start_k = max((i-1) * self.min_tokens, 0)
                end_k = min(j, j - self.min_tokens + 1)
                
                for k in range(start_k, end_k):
                    if k >= j:
                        continue
                        
                    span = " ".join(tgt_tokens[k:j])
                    if not span.strip():
                        continue
                        
                    try:
                        tgt_emb_result = embed_func([span])
                        
                        # 임베딩 결과 정규화
                        if isinstance(tgt_emb_result, list) and len(tgt_emb_result) > 0:
                            tgt_emb = np.array(tgt_emb_result[0], dtype=np.float32)
                        elif isinstance(tgt_emb_result, np.ndarray) and len(tgt_emb_result) > 0:
                            tgt_emb = tgt_emb_result[0].astype(np.float32)
                        else:
                            continue
                        
                        # 코사인 유사도 계산 (Prototype02 완전 복사)
                        src_vec = src_embs[i-1].astype(np.float32)
                        
                        src_norm = np.linalg.norm(src_vec)
                        tgt_norm = np.linalg.norm(tgt_emb)
                        
                        if src_norm > 1e-8 and tgt_norm > 1e-8:
                            sim = float(np.dot(src_vec, tgt_emb) / (src_norm * tgt_norm))
                        else:
                            sim = 0.0
                        
                        score = dp[i-1, k] + sim
                        if score > dp[i, j]:
                            dp[i, j] = score
                            back[i, j] = k
                            
                    except Exception as e:
                        logger.debug(f"임베딩 계산 실패 (span: '{span}'): {e}")
                        continue

        # Traceback (Prototype02 완전 복사)
        cuts = [T]
        curr = T
        for i in range(N, 0, -1):
            if 0 <= i < back.shape[0] and 0 <= curr < back.shape[1]:
                prev = int(back[i, curr])
                cuts.append(prev)
                curr = prev
            else:
                cuts.append(0)
        cuts = cuts[::-1]

        # Build actual spans (Prototype02 완전 복사)
        tgt_spans = []
        for i in range(N):
            if i < len(cuts) - 1:
                start_idx = cuts[i]
                end_idx = cuts[i+1]
                if 0 <= start_idx < len(tgt_tokens) and start_idx < end_idx <= len(tgt_tokens):
                    span = " ".join(tgt_tokens[start_idx:end_idx]).strip()
                else:
                    span = ""
            else:
                span = ""
            tgt_spans.append(span)
        
        return tgt_spans

class TextAlignmentProcessor:
    """Prototype02 전체 파이프라인 통합"""
    
    def __init__(self, min_tokens: int = 1, **kwargs):
        self.text_masker = TextMasker()
        self.src_splitter = SourceTextSplitter()
        self.tgt_aligner = TargetTextAligner(min_tokens=min_tokens)
        logger.debug("TextAlignmentProcessor 초기화 완료")
    
    def process(self, src_text: str, tgt_text: str, embed_func: Callable) -> Tuple[str, str, Dict]:
        """Prototype02 전체 파이프라인 실행"""
        processing_info = {'algorithm': 'prototype02_complete_integration'}
        
        if not src_text or not tgt_text:
            return src_text, tgt_text, {'error': 'Empty input text'}
        
        try:
            # 1. 마스킹 (Prototype02 순서 보장)
            masked_src, src_masks = self.text_masker.mask(src_text, text_type="source")
            masked_tgt, tgt_masks = self.text_masker.mask(tgt_text, text_type="target")
            
            processing_info.update({
                'src_masks': len(src_masks),
                'tgt_masks': len(tgt_masks)
            })

            # 2. 원문 의미 단위 분할
            src_units = self.src_splitter.split(masked_src)
            if not src_units:
                src_units = [masked_src]

            # 3. 번역문 의미 단위 분할
            tgt_units = self.tgt_aligner.align(src_units, masked_tgt, embed_func)

            # 길이 검증 및 조정
            if len(src_units) != len(tgt_units):
                logger.warning(f"길이 불일치: 원문 {len(src_units)}, 번역문 {len(tgt_units)}")
                min_len = min(len(src_units), len(tgt_units))
                if min_len > 0:
                    src_units = src_units[:min_len]
                    tgt_units = tgt_units[:min_len]
                else:
                    return src_text, tgt_text, {'error': 'No valid units after splitting'}

            # 4. 언마스킹 (Prototype02 순서 보장)
            restored_src_units = []
            restored_tgt_units = []
            
            for src_unit in src_units:
                restored_src_units.append(self.text_masker.restore(src_unit, src_masks))
            
            for tgt_unit in tgt_units:
                restored_tgt_units.append(self.text_masker.restore(tgt_unit, tgt_masks))

            # 5. 결과 필터링 및 조합
            filtered_pairs = []
            for src_unit, tgt_unit in zip(restored_src_units, restored_tgt_units):
                if src_unit.strip() or tgt_unit.strip():  # 빈 문자열도 허용 (Prototype02 호환)
                    filtered_pairs.append((src_unit, tgt_unit))
            
            if not filtered_pairs:
                return src_text, tgt_text, {'error': 'No valid pairs after processing'}

            final_src_parts, final_tgt_parts = zip(*filtered_pairs)
            final_source = ' | '.join(final_src_parts)
            final_target = ' | '.join(final_tgt_parts)
            
            processing_info.update({
                'status': 'success',
                'units_count': len(filtered_pairs),
                'src_units': len(src_units),
                'tgt_units': len(tgt_units)
            })

            return final_source, final_target, processing_info

        except Exception as e:
            logger.error(f"텍스트 정렬 처리 실패: {e}", exc_info=True)
            return src_text, tgt_text, {'error': str(e)}

# *** 하위 호환성을 위한 함수형 인터페이스 (완전 보장) ***
def split_src_meaning_units(text: str) -> List[str]:
    """현재 시스템 호환용 - Prototype02 동일 결과 보장"""
    splitter = SourceTextSplitter()
    return splitter.split(text)

def split_tgt_meaning_units(
    src_units: List[str],
    masked_tgt: str,
    embed_func: Callable,
    source_analyzer=None,  # 하위 호환성
    target_analyzer=None,  # 하위 호환성
    target_tokenizer=None, # 하위 호환성
    min_tokens: int = 1
) -> List[str]:
    """현재 시스템 호환용 - Prototype02 동일 결과 보장"""
    aligner = TargetTextAligner(min_tokens=min_tokens)
    return aligner.align(src_units, masked_tgt, embed_func)

def mask_brackets(text: str, text_type: str = "source") -> Tuple[str, List[str]]:
    """현재 시스템 호환용 - Prototype02 동일 결과 보장"""
    masker = TextMasker()
    return masker.mask(text, text_type)

def restore_masks(text: str, masks: List[str]) -> str:
    """현재 시스템 호환용 - Prototype02 동일 결과 보장"""
    masker = TextMasker()
    return masker.restore(text, masks)