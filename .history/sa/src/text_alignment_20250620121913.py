"""Prototype02 완전 통합 - 분석 정보 제대로 활용하는 버전"""
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
    """Prototype02 punctuation.py 완전 복제 (변경 없음)"""
    
    def __init__(self, **kwargs):
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

class AnalyzerAwareSourceTextSplitter:
    """분석기 정보를 적극적으로 활용하는 소스 텍스트 분할기"""
    
    def __init__(self, analyzer=None, **kwargs):
        self.analyzer = analyzer
        
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
        
        logger.debug(f"분석기 인식 분할기 초기화: {type(analyzer).__name__ if analyzer else 'None'}")
    
    def split(self, text: str) -> List[str]:
        """분석기 정보를 적극 활용한 스마트 분할"""
        if not text or not text.strip():
            return []
        
        logger.debug(f"분석기 분할 시작: '{text}'")
        
        # *** 1단계: 분석기가 있으면 분석 우선 수행 ***
        analysis_info = []
        if self.analyzer:
            try:
                analysis_info = self.analyzer.analyze(text)
                logger.debug(f"분석 정보 획득: {len(analysis_info)}개 형태소")
                
                # 분석 결과 로깅
                for i, morph in enumerate(analysis_info[:5]):  # 처음 5개만
                    logger.debug(f"  {i}: {morph.get('surface', '')} ({morph.get('pos', '')})")
                    
            except Exception as e:
                logger.debug(f"분석기 실행 실패: {e}")
        
        # *** 2단계: 분석 정보가 있으면 그것을 우선 사용 ***
        if analysis_info:
            return self._split_by_analysis(text, analysis_info)
        
        # *** 3단계: 분석 정보가 없으면 기존 Prototype02 방식 ***
        return self._split_by_prototype02(text)
    
    def _split_by_analysis(self, text: str, analysis_info: List[Dict]) -> List[str]:
        """분석 정보 기반 분할 (메인 로직)"""
        
        logger.debug("분석 정보 기반 분할 시작")
        
        # 의미 있는 형태소들만 추출
        meaningful_morphemes = []
        for morph in analysis_info:
            surface = morph.get('surface', '').strip()
            pos = morph.get('pos', '')
            
            if not surface:
                continue
            
            # 품사별 의미도 판단
            is_meaningful = self._is_meaningful_morpheme(pos, surface)
            
            if is_meaningful:
                meaningful_morphemes.append({
                    'surface': surface,
                    'pos': pos,
                    'original': morph
                })
        
        logger.debug(f"의미 있는 형태소 {len(meaningful_morphemes)}개 추출")
        
        # 의미 있는 형태소들을 결합하여 의미 단위 생성
        if meaningful_morphemes:
            units = self._combine_meaningful_morphemes(meaningful_morphemes)
            logger.debug(f"최종 의미 단위: {units}")
            return units
        else:
            # 분석 실패시 기존 방식 폴백
            logger.debug("의미 있는 형태소 없음, Prototype02 방식으로 폴백")
            return self._split_by_prototype02(text)
    
    def _is_meaningful_morpheme(self, pos: str, surface: str) -> bool:
        """형태소가 의미 있는지 판단"""
        
        # 한국어 품사 (MeCab, KoNLPy)
        korean_meaningful_pos = {
            # 명사류
            'NNG', 'NNP', 'NNB', 'NR', 'NP',
            # 동사, 형용사
            'VV', 'VA', 'VX', 'VCP', 'VCN',
            # 관형사, 부사
            'MM', 'MAG', 'MAJ',
            # 한자어
            'SH', 'SL', 'SN',
            # 영어
            'SL'
        }
        
        # 중국어 품사 (Jieba)
        chinese_meaningful_pos = {
            'n', 'nr', 'ns', 'nt', 'nw', 'nz',  # 명사류
            'v', 'vd', 'vn',  # 동사류
            'a', 'ad', 'an',  # 형용사류
            'f', 'i', 'j', 'l',  # 방위사, 관용어 등
            'x'  # 기타 의미 있는 단어
        }
        
        # 품사 확인
        pos_upper = pos.upper()
        pos_lower = pos.lower()
        
        # 한국어 품사 확인
        if any(pos_upper.startswith(meaningful_pos) for meaningful_pos in korean_meaningful_pos):
            return True
        
        # 중국어 품사 확인  
        if pos_lower in chinese_meaningful_pos:
            return True
        
        # 길이 기반 필터링 (의미 있는 단어는 보통 1자 이상)
        if len(surface) >= 1 and surface not in {'은', '는', '이', '가', '을', '를', '에', '의', '로', '와', '과', '도', '만'}:
            return True
        
        return False
    
    def _combine_meaningful_morphemes(self, morphemes: List[Dict]) -> List[str]:
        """의미 있는 형태소들을 적절히 결합"""
        
        if not morphemes:
            return []
        
        units = []
        current_unit = ""
        
        for i, morph in enumerate(morphemes):
            surface = morph['surface']
            pos = morph['pos']
            
            # 단독으로 의미 단위가 되는 경우
            if self._should_be_separate_unit(pos, surface):
                # 이전 단위 마무리
                if current_unit:
                    units.append(current_unit.strip())
                    current_unit = ""
                
                # 현재 형태소를 독립 단위로
                units.append(surface)
            
            # 이전 단위와 결합하는 경우
            else:
                if current_unit:
                    current_unit += surface
                else:
                    current_unit = surface
        
        # 마지막 단위 추가
        if current_unit:
            units.append(current_unit.strip())
        
        # 빈 단위 제거
        units = [unit for unit in units if unit.strip()]
        
        return units if units else [morphemes[0]['surface']]  # 최소 1개는 보장
    
    def _should_be_separate_unit(self, pos: str, surface: str) -> bool:
        """독립적인 의미 단위가 될지 판단"""
        
        # 한자어는 보통 독립 단위
        if regex.search(r'\p{Han}', surface):
            return True
        
        # 길이가 긴 단어는 독립 단위
        if len(surface) >= 3:
            return True
        
        # 명사는 독립 단위 경향
        if pos.upper().startswith(('NNG', 'NNP', 'NR')):
            return True
        
        # 중국어 명사
        if pos.lower() in {'n', 'nr', 'ns', 'nt', 'nw', 'nz'}:
            return True
        
        return False
    
    def _split_by_prototype02(self, text: str) -> List[str]:
        """기존 Prototype02 분할 방식 (폴백용)"""
        
        logger.debug("Prototype02 기본 분할 방식 사용")
        
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

            # 3) 순수 한글 토큰: SoyNLP 사용
            if self.hangul_re.match(tok):
                if self.tokenizer:
                    try:
                        korean_tokens = self.tokenizer.tokenize(tok)
                        if korean_tokens:
                            units.extend(korean_tokens)
                        else:
                            units.append(tok)
                    except Exception as e:
                        logger.debug(f"SoyNLP 토크나이징 실패: {e}")
                        units.append(tok)
                else:
                    units.append(tok)
                i += 1
                continue

            # 4) 기타 토큰 그대로 보존
            units.append(tok)
            i += 1

        return units

# 기존 SourceTextSplitter 호환성을 위해 유지
class SourceTextSplitter(AnalyzerAwareSourceTextSplitter):
    """호환성을 위한 기존 클래스"""
    
    def __init__(self, **kwargs):
        super().__init__(analyzer=None, **kwargs)

# 나머지 클래스들은 변경 없음 (TargetTextAligner, TextAligner, TextAlignmentProcessor 등)
class TargetTextAligner:
    """Prototype02 tokenizer.py split_tgt_by_src_units_semantic 완전 복제"""
    
    def __init__(self, min_tokens: int = 1, **kwargs):
        self.min_tokens = max(1, min_tokens)
    
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

class TextAligner:
    """Prototype02 aligner.py의 align_src_tgt 완전 복제"""
    
    def __init__(self, **kwargs):
        pass
    
    def align(self, src_units: List[str], tgt_units: List[str], embed_func: Callable) -> List[Tuple[str, str]]:
        """Prototype02 align_src_tgt 함수 완전 복사"""
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
                tgt_units = tgt_units + [""] * (len(src_units) - len(tgt_units))
            else:
                src_units = src_units + [""] * (len(tgt_units) - len(src_units))

        return list(zip(src_units, tgt_units))

class TextAlignmentProcessor:
    """분석기 정보를 적극 활용한 텍스트 정렬 프로세서"""
    
    def __init__(self, source_analyzer=None, target_analyzer=None, min_tokens: int = 1, **kwargs):
        self.text_masker = TextMasker()
        
        # *** 분석기를 활용한 스마트 분할기 사용 ***
        self.src_splitter = AnalyzerAwareSourceTextSplitter(analyzer=source_analyzer)
        
        self.tgt_aligner = TargetTextAligner(min_tokens=min_tokens)
        self.text_aligner = TextAligner()
        
        self.source_analyzer = source_analyzer
        self.target_analyzer = target_analyzer
        
        logger.info(f"TextAlignmentProcessor 초기화 완료 (분석기: {type(source_analyzer).__name__ if source_analyzer else 'None'})")
    
    def process(self, src_text: str, tgt_text: str, embed_func: Callable) -> Tuple[str, str, Dict]:
        """분석기 정보를 적극 활용한 텍스트 정렬"""
        processing_info = {
            'algorithm': 'prototype02_with_active_analyzer',
            'source_analyzer': type(self.source_analyzer).__name__ if self.source_analyzer else 'None',
            'target_analyzer': type(self.target_analyzer).__name__ if self.target_analyzer else 'None'
        }
        
        if not src_text or not tgt_text:
            return src_text, tgt_text, {'error': 'Empty input text'}
        
        try:
            logger.debug(f"처리 시작: 원문='{src_text[:50]}...', 번역문='{tgt_text[:50]}...'")
            
            # 1. 마스킹 (Prototype02 순서 보장)
            masked_src, src_masks = self.text_masker.mask(src_text, text_type="source")
            masked_tgt, tgt_masks = self.text_masker.mask(tgt_text, text_type="target")
            
            processing_info.update({
                'src_masks': len(src_masks),
                'tgt_masks': len(tgt_masks)
            })

            # 2. *** 분석기 정보를 적극 활용한 원문 분할 ***
            src_units = self.src_splitter.split(masked_src)
            if not src_units:
                src_units = [masked_src]
            
            logger.debug(f"원문 분할 결과: {src_units}")

            # 3. 번역문 의미 단위 분할 (DP 알고리즘)
            tgt_units = self.tgt_aligner.align(src_units, masked_tgt, embed_func)
            
            logger.debug(f"번역문 분할 결과: {tgt_units}")

            # 4. 언마스킹
            restored_src_units = []
            restored_tgt_units = []
            
            for src_unit in src_units:
                restored_src_units.append(self.text_masker.restore(src_unit, src_masks))
            
            for tgt_unit in tgt_units:
                restored_tgt_units.append(self.text_masker.restore(tgt_unit, tgt_masks))

            # 5. 정렬 수행
            aligned_pairs = self.text_aligner.align(restored_src_units, restored_tgt_units, embed_func)

            if not aligned_pairs:
                return src_text, tgt_text, {'error': 'Empty alignment result'}

            # 6. 결과 조합
            filtered_pairs = []
            for src_unit, tgt_unit in aligned_pairs:
                if src_unit.strip() or tgt_unit.strip():
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
            
            logger.debug(f"최종 결과: 원문='{final_source}', 번역문='{final_target}'")

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
    source_analyzer=None,
    target_analyzer=None,
    target_tokenizer=None,
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