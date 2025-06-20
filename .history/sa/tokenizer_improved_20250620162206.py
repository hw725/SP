"""개선된 토크나이저 - 공백 보존 및 의미 단위 최적화"""

import jieba
import MeCab
import logging
import re
from typing import List, Optional, Callable
import numpy as np

logger = logging.getLogger(__name__)

# MeCab 초기화
try:
    mecab = MeCab.Tagger('-Owakati')  # 공백 분할 모드
    logger.info("MeCab 초기화 성공")
except Exception as e:
    logger.error(f"MeCab 초기화 실패: {e}")
    mecab = None

def split_src_meaning_units(
    text: str, 
    min_tokens: int = 1, 
    max_tokens: int = 10,
    use_advanced: bool = True
) -> List[str]:
    """
    개선된 원문 분할 - 의미 단위 보존
    """
    if not text or not text.strip():
        return []
    
    try:
        logger.debug(f"원문 분할 시작: {text[:50]}...")
        
        # 1단계: 구문 경계 기반 분할
        units = syntactic_src_split(text)
        
        # 2단계: 의미 단위 조정
        if use_advanced:
            units = adjust_semantic_units(units, min_tokens, max_tokens)
        
        logger.debug(f"원문 분할 완료: {len(units)}개 단위")
        return units
        
    except Exception as e:
        logger.error(f"원문 분할 실패: {e}")
        return [text]

def syntactic_src_split(text: str) -> List[str]:
    """구문 경계 기반 원문 분할"""
    
    # 주요 구문 경계 표시어들
    primary_delimiters = [
        '然後에', '然後',     # 시간 접속
        '則', '而', '且',     # 한문 접속사
        '이요', '이며',       # 병렬 접속
        '라가', '라서',       # 전환 접속
        '면', '이면', '하면'  # 조건 접속
    ]
    
    # 보조 경계 표시어들
    secondary_delimiters = [
        '云', '曰',          # 인용
        '者', '之',          # 관계사
        '以', '於'           # 전치사류
    ]
    
    units = [text]
    
    # 1차: 주요 구분자로 분할
    for delimiter in primary_delimiters:
        new_units = []
        for unit in units:
            if delimiter in unit:
                parts = split_preserving_delimiter(unit, delimiter)
                new_units.extend(parts)
            else:
                new_units.append(unit)
        units = new_units
    
    # 2차: 너무 긴 단위는 보조 구분자로 추가 분할
    final_units = []
    for unit in units:
        if len(unit) > 30:  # 긴 단위만 추가 분할
            sub_units = secondary_split(unit, secondary_delimiters)
            final_units.extend(sub_units)
        else:
            final_units.append(unit)
    
    return [u.strip() for u in final_units if u.strip()]

def split_preserving_delimiter(text: str, delimiter: str) -> List[str]:
    """구분자를 포함하여 분할 (의미 보존)"""
    if delimiter not in text:
        return [text]
    
    parts = text.split(delimiter)
    result = []
    
    for i, part in enumerate(parts[:-1]):  # 마지막 제외
        if part.strip():
            result.append(part + delimiter)
    
    # 마지막 부분 처리
    if parts[-1].strip():
        result.append(parts[-1])
    
    return result

def secondary_split(text: str, delimiters: List[str]) -> List[str]:
    """보조 구분자로 추가 분할"""
    
    # 한자어 블록 + 조사 패턴 감지
    hanja_pattern = r'([\u4e00-\u9fff]{2,}(?:[\uac00-\ud7af]{1,2})?)'
    blocks = re.findall(hanja_pattern, text)
    
    if len(blocks) >= 2:
        # 패턴 기반 분할
        result = []
        remaining = text
        
        for block in blocks:
            if block in remaining:
                idx = remaining.find(block)
                if idx > 0:
                    result.append(remaining[:idx + len(block)])
                else:
                    result.append(block)
                remaining = remaining[idx + len(block):]
        
        if remaining.strip():
            result.append(remaining)
        
        return [r.strip() for r in result if r.strip()]
    
    return [text]

def split_tgt_meaning_units(
    src_text: str,
    tgt_text: str,
    embed_func: Optional[Callable] = None,
    use_semantic: bool = True,
    min_tokens: int = 1,
    max_tokens: int = 10,
    similarity_threshold: float = 0.3
) -> List[str]:
    """
    개선된 번역문 분할 - 적응적 분할
    """
    if not tgt_text or not tgt_text.strip():
        return []
    
    try:
        logger.debug(f"번역문 분할 시작: {tgt_text[:50]}...")
        
        # 원문 단위 분석
        src_units = split_src_meaning_units(src_text, min_tokens, max_tokens)
        
        if use_semantic and embed_func is not None:
            # 적응적 의미 기반 분할
            units = adaptive_tgt_split(
                src_units, tgt_text, embed_func, 
                min_tokens, max_tokens, similarity_threshold
            )
        else:
            # 개선된 구문 기반 분할
            units = improved_tgt_split(tgt_text, len(src_units), min_tokens, max_tokens)
        
        logger.debug(f"번역문 분할 완료: {len(units)}개 단위")
        return units
        
    except Exception as e:
        logger.error(f"번역문 분할 실패: {e}")
        return [tgt_text]

def improved_tgt_split(
    tgt_text: str, 
    src_unit_count: int,
    min_tokens: int = 1, 
    max_tokens: int = 10
) -> List[str]:
    """개선된 MeCab 기반 번역문 분할"""
    
    if mecab is None:
        return basic_punctuation_split(tgt_text, src_unit_count)
    
    try:
        # MeCab으로 형태소 분석
        result = mecab.parse(tgt_text).strip()
        tokens = result.split(' ') if result else []
        
        if not tokens:
            return [tgt_text]
        
        # 의미 단위별 그룹화 (개선된 방식)
        units = improved_morpheme_grouping(
            tokens, tgt_text, src_unit_count, min_tokens, max_tokens
        )
        
        return units
        
    except Exception as e:
        logger.error(f"MeCab 분할 실패: {e}")
        return basic_punctuation_split(tgt_text, src_unit_count)

def improved_morpheme_grouping(
    tokens: List[str], 
    original_text: str,
    src_unit_count: int,
    min_tokens: int = 1, 
    max_tokens: int = 10
) -> List[str]:
    """개선된 형태소 그룹화 - 공백 보존"""
    
    if not tokens:
        return [original_text]
    
    # 목표 분할 수 계산 (적응적)
    target_segments = calculate_target_segments(
        len(original_text), src_unit_count, len(tokens)
    )
    
    # 의미 경계 감지
    boundaries = detect_semantic_boundaries(tokens, original_text)
    
    # 최적 분할점 선택
    optimal_boundaries = select_optimal_boundaries(
        boundaries, target_segments, len(tokens)
    )
    
    # 실제 분할 수행 (공백 보존)
    units = split_by_boundaries(tokens, original_text, optimal_boundaries)
    
    return units

def calculate_target_segments(text_length: int, src_count: int, token_count: int) -> int:
    """적응적 목표 분할 수 계산"""
    
    # 기본 전략: 원문 단위 수를 기준으로 하되 텍스트 특성 고려
    base_target = max(1, src_count)
    
    # 텍스트 길이 보정
    if text_length > 100:  # 매우 긴 텍스트
        length_factor = 1.5
    elif text_length > 50:  # 긴 텍스트
        length_factor = 1.2
    else:  # 일반 텍스트
        length_factor = 1.0
    
    # 토큰 밀도 보정
    density = text_length / max(1, token_count)
    if density > 3:  # 긴 토큰들 (복합어 많음)
        density_factor = 0.8
    else:  # 짧은 토큰들
        density_factor = 1.0
    
    target = int(base_target * length_factor * density_factor)
    
    # 합리적 범위로 제한
    return max(1, min(target, token_count // 2))

def detect_semantic_boundaries(tokens: List[str], original_text: str) -> List[int]:
    """의미 경계 감지"""
    boundaries = []
    
    # 구두점 경계
    for i, token in enumerate(tokens):
        if re.search(r'[.!?。！？,，;：:]', token):
            boundaries.append(i + 1)
    
    # 접속 표현 경계
    connector_patterns = [
        r'그런데|하지만|따라서|그러므로|즉|또한|그리고',
        r'을|를|이|가|은|는|에서|으로|와|과',  # 주요 조사
        r'했다가|되면|때문에|하여|므로'  # 연결 어미
    ]
    
    for i, token in enumerate(tokens):
        for pattern in connector_patterns:
            if re.search(pattern, token):
                boundaries.append(i + 1)
                break
    
    # 한자어 블록 경계
    for i in range(len(tokens) - 1):
        current_has_hanja = bool(re.search(r'[\u4e00-\u9fff]', tokens[i]))
        next_has_hanja = bool(re.search(r'[\u4e00-\u9fff]', tokens[i + 1]))
        
        # 한자어 → 한글어 또는 그 반대
        if current_has_hanja != next_has_hanja:
            boundaries.append(i + 1)
    
    return sorted(set(boundaries))

def select_optimal_boundaries(
    boundaries: List[int], 
    target_segments: int, 
    total_tokens: int
) -> List[int]:
    """최적 분할점 선택"""
    
    if not boundaries or target_segments <= 1:
        return [total_tokens]
    
    # 경계점들을 균등 분포에 가깝게 선택
    if len(boundaries) <= target_segments - 1:
        return boundaries + [total_tokens]
    
    # 너무 많은 경계점이 있는 경우, 균등하게 선택
    selected = []
    interval = len(boundaries) / (target_segments - 1)
    
    for i in range(target_segments - 1):
        idx = int(i * interval)
        selected.append(boundaries[idx])
    
    selected.append(total_tokens)
    return sorted(set(selected))

def split_by_boundaries(
    tokens: List[str], 
    original_text: str, 
    boundaries: List[int]
) -> List[str]:
    """경계점을 기준으로 분할 (공백 보존)"""
    
    if not boundaries:
        return [original_text]
    
    units = []
    start = 0
    
    for boundary in boundaries:
        if boundary > start:
            # 토큰 범위의 원본 텍스트 추출
            segment_tokens = tokens[start:boundary]
            if segment_tokens:
                # 원본 텍스트에서 해당 부분 찾기
                segment_text = reconstruct_segment(segment_tokens, original_text, start)
                if segment_text.strip():
                    units.append(segment_text.strip())
            start = boundary
    
    return units

def reconstruct_segment(
    segment_tokens: List[str], 
    original_text: str, 
    token_start_idx: int
) -> str:
    """토큰들로부터 원본 텍스트 세그먼트 재구성"""
    
    if not segment_tokens:
        return ""
    
    # 단순 결합 시도
    simple_join = ''.join(segment_tokens)
    
    # 원본 텍스트에서 해당 부분 찾기
    if simple_join in original_text:
        start_pos = original_text.find(simple_join)
        if start_pos != -1:
            return original_text[start_pos:start_pos + len(simple_join)]
    
    # 부분 매칭으로 재구성
    result = ""
    remaining_text = original_text
    
    for token in segment_tokens:
        if token in remaining_text:
            pos = remaining_text.find(token)
            # 토큰 앞의 공백까지 포함
            if pos > 0 and remaining_text[pos-1] == ' ':
                result += remaining_text[:pos + len(token)]
            else:
                result += remaining_text[:pos + len(token)]
            remaining_text = remaining_text[pos + len(token):]
        else:
            result += token
    
    return result

def adaptive_tgt_split(
    src_units: List[str], 
    tgt_text: str, 
    embed_func: Callable,
    min_tokens: int,
    max_tokens: int,
    similarity_threshold: float = 0.3
) -> List[str]:
    """적응적 의미 기반 번역문 분할"""
    
    try:
        # 1. 기본 분할 수행
        base_units = improved_tgt_split(
            tgt_text, len(src_units), min_tokens, max_tokens
        )
        
        # 2. 의미 유사도 기반 재조합
        if len(src_units) > 1 and len(base_units) > 1:
            optimized_units = semantic_optimization(
                src_units, base_units, embed_func, similarity_threshold
            )
            return optimized_units
        
        return base_units
        
    except Exception as e:
        logger.error(f"적응적 분할 실패: {e}")
        return improved_tgt_split(tgt_text, len(src_units), min_tokens, max_tokens)

def semantic_optimization(
    src_units: List[str], 
    tgt_candidates: List[str], 
    embed_func: Callable,
    threshold: float = 0.3
) -> List[str]:
    """의미 유사도 기반 최적화"""
    
    try:
        # 임베딩 계산
        all_texts = src_units + tgt_candidates
        embeddings = embed_func(all_texts)
        
        if len(embeddings) != len(all_texts):
            return tgt_candidates
        
        src_embeddings = embeddings[:len(src_units)]
        tgt_embeddings = embeddings[len(src_units):]
        
        # 유사도 매트릭스
        similarity_matrix = compute_similarity_matrix(src_embeddings, tgt_embeddings)
        
        # 동적 프로그래밍으로 최적 분할 찾기
        optimal_grouping = find_optimal_tgt_grouping(
            tgt_candidates, similarity_matrix, len(src_units), threshold
        )
        
        return optimal_grouping
        
    except Exception as e:
        logger.error(f"의미 최적화 실패: {e}")
        return tgt_candidates

def compute_similarity_matrix(src_embeddings: List, tgt_embeddings: List) -> np.ndarray:
    """코사인 유사도 매트릭스 계산"""
    
    try:
        src_matrix = np.array(src_embeddings)
        tgt_matrix = np.array(tgt_embeddings)
        
        # 정규화
        src_norm = src_matrix / np.linalg.norm(src_matrix, axis=1, keepdims=True)
        tgt_norm = tgt_matrix / np.linalg.norm(tgt_matrix, axis=1, keepdims=True)
        
        # 코사인 유사도
        similarity = np.dot(src_norm, tgt_norm.T)
        
        # 0~1 범위로 변환
        return (similarity + 1) / 2
        
    except Exception as e:
        logger.error(f"유사도 계산 실패: {e}")
        return np.zeros((len(src_embeddings), len(tgt_embeddings)))

def find_optimal_tgt_grouping(
    tgt_candidates: List[str], 
    similarity_matrix: np.ndarray,
    target_count: int,
    threshold: float = 0.3
) -> List[str]:
    """최적 타겟 그룹화 (동적 프로그래밍)"""
    
    try:
        n = len(tgt_candidates)
        if n <= target_count:
            return tgt_candidates
        
        # DP 테이블: dp[i][j] = i번째까지 j개 그룹으로 분할했을 때 최대 점수
        dp = np.full((n + 1, target_count + 1), -np.inf)
        backtrack = np.zeros((n + 1, target_count + 1, 2), dtype=int)
        
        dp[0][0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, min(i, target_count) + 1):
                # k부터 i-1까지를 하나의 그룹으로 만드는 경우
                for k in range(j - 1, i):
                    if dp[k][j-1] == -np.inf:
                        continue
                    
                    # 그룹 [k:i]의 점수 계산
                    group_score = calculate_group_score(
                        tgt_candidates[k:i], similarity_matrix, j-1
                    )
                    
                    total_score = dp[k][j-1] + group_score
                    
                    if total_score > dp[i][j]:
                        dp[i][j] = total_score
                        backtrack[i][j] = [k, j-1]
        
        # 백트래킹으로 최적 분할 찾기
        groups = []
        i, j = n, target_count
        
        while j > 0:
            k, prev_j = backtrack[i][j]
            if k < i:
                group = ''.join(tgt_candidates[k:i])
                groups.append(group)
            i, j = k, prev_j
        
        groups.reverse()
        return groups if groups else tgt_candidates
        
    except Exception as e:
        logger.error(f"최적 그룹화 실패: {e}")
        return tgt_candidates

def calculate_group_score(
    group_candidates: List[str],
    similarity_matrix: np.ndarray,
    src_idx: int
) -> float:
    """그룹 점수 계산"""
    
    if not group_candidates or src_idx >= similarity_matrix.shape[0]:
        return 0.0
    
    # 그룹 내 후보들의 평균 유사도
    scores = []
    for i, candidate in enumerate(group_candidates):
        if src_idx < similarity_matrix.shape[0] and i < similarity_matrix.shape[1]:
            scores.append(similarity_matrix[src_idx, i])
    
    return np.mean(scores) if scores else 0.0

def basic_punctuation_split(text: str, target_count: int) -> List[str]:
    """기본 구두점 기반 분할 (백업용)"""
    
    # 구두점으로 1차 분할
    delimiters = ['.', '!', '?', '。', '！', '？', ',', '，', ';', '：', ':']
    
    units = [text]
    for delimiter in delimiters:
        new_units = []
        for unit in units:
            if delimiter in unit:
                parts = unit.split(delimiter)
                for i, part in enumerate(parts[:-1]):
                    if part.strip():
                        new_units.append(part + delimiter)
                if parts[-1].strip():
                    new_units.append(parts[-1])
            else:
                new_units.append(unit)
        units = new_units
    
    # 목표 개수에 맞춰 조정
    units = [u.strip() for u in units if u.strip()]
    
    if len(units) > target_count and target_count > 1:
        # 너무 많으면 병합
        merged = []
        chunk_size = len(units) // target_count
        for i in range(0, len(units), chunk_size):
            chunk = units[i:i + chunk_size]
            merged.append(' '.join(chunk))
        units = merged
    
    return units

def adjust_semantic_units(
    units: List[str], 
    min_tokens: int, 
    max_tokens: int
) -> List[str]:
    """의미 단위 길이 조정"""
    
    adjusted = []
    
    for unit in units:
        if len(unit) > max_tokens * 4:  # 너무 긴 단위 분할
            # 중간 지점에서 적절한 분할점 찾기
            mid = len(unit) // 2
            for i in range(mid - 5, mid + 5):
                if i > 0 and i < len(unit) and unit[i] in ' ，,、':
                    adjusted.append(unit[:i+1].strip())
                    adjusted.append(unit[i+1:].strip())
                    break
            else:
                # 적절한 분할점이 없으면 그대로
                adjusted.append(unit)
        else:
            adjusted.append(unit)
    
    # 너무 짧은 단위들 병합
    if min_tokens > 1:
        merged = []
        temp = ""
        
        for unit in adjusted:
            if len(temp + unit) < min_tokens * 2:
                temp += unit
            else:
                if temp:
                    merged.append(temp.strip())
                temp = unit
        
        if temp:
            merged.append(temp.strip())
        
        adjusted = merged
    
    return [u for u in adjusted if u.strip()]

if __name__ == "__main__":
    # 테스트
    logging.basicConfig(level=logging.DEBUG)
    
    print("🧪 개선된 토크나이저 테스트")
    
    test_cases = [
        ("興也라", "興이다."),
        ("蒹은 薕(렴)이요 葭는 蘆也라", "蒹은 물억새이고 葭는 갈대이다."),
        ("白露凝戾爲霜然後에 歲事成이요 國家待禮然後興이라", 
         "白露가 얼어 서리가 된 뒤에야 歲事가 이루어지고 國家는 禮가 행해진 뒤에야 흥성한다."),
    ]
    
    for i, (src, tgt) in enumerate(test_cases, 1):
        print(f"\n=== 테스트 케이스 {i} ===")
        print(f"원문: {src}")
        print(f"번역: {tgt}")
        
        # 원문 분할
        src_units = split_src_meaning_units(src, min_tokens=1, max_tokens=15)
        print(f"✅ 개선된 원문 분할: {src_units}")
        
        # 번역문 분할 (구문 기반)
        tgt_units = split_tgt_meaning_units(
            src, tgt, 
            embed_func=None, 
            use_semantic=False,
            min_tokens=1, 
            max_tokens=15
        )
        print(f"✅ 개선된 번역 분할: {tgt_units}")
        
        # 의미 기반 분할 (더미 임베딩)
        def dummy_embed(texts):
            return [np.random.randn(100) for _ in texts]
        
        tgt_units_semantic = split_tgt_meaning_units(
            src, tgt,
            embed_func=dummy_embed,
            use_semantic=True,
            min_tokens=1,
            max_tokens=15
        )
        print(f"🔗 의미 기반 분할: {tgt_units_semantic}")