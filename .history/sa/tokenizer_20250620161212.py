"""텍스트 토크나이징 모듈 - 매개변수 추가"""

import jieba
import MeCab
import logging
import re
from typing import List, Optional, Callable
import numpy as np

# 로깅 설정
logger = logging.getLogger(__name__)

# MeCab 초기화
try:
    mecab = MeCab.Tagger()
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
    원문을 의미 단위로 분할
    
    Args:
        text: 원문 텍스트
        min_tokens: 최소 토큰 수
        max_tokens: 최대 토큰 수  
        use_advanced: 고급 분할 기법 사용 여부
    
    Returns:
        List[str]: 분할된 의미 단위 리스트
    """
    if not text or not text.strip():
        return []
    
    try:
        logger.debug(f"원문 분할 시작: {text[:50]}...")
        
        # 1단계: 기본 문법 구조 분할
        units = basic_src_split(text)
        
        # 2단계: 고급 분할 (옵션)
        if use_advanced:
            units = advanced_src_split(units)
        
        # 3단계: 길이 제한 적용
        units = apply_length_constraints(units, min_tokens, max_tokens, is_src=True)
        
        logger.debug(f"원문 분할 완료: {len(units)}개 단위")
        return units
        
    except Exception as e:
        logger.error(f"원문 분할 실패: {e}")
        return [text]  # 실패 시 원본 반환

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
    번역문을 의미 단위로 분할
    
    Args:
        src_text: 원문
        tgt_text: 번역문
        embed_func: 임베딩 함수
        use_semantic: 의미 기반 분할 사용 여부
        min_tokens: 최소 토큰 수
        max_tokens: 최대 토큰 수
        similarity_threshold: 유사도 임계값
    
    Returns:
        List[str]: 분할된 의미 단위 리스트
    """
    if not tgt_text or not tgt_text.strip():
        return []
    
    try:
        logger.debug(f"번역문 분할 시작: {tgt_text[:50]}...")
        
        if use_semantic and embed_func is not None:
            # 의미 기반 분할
            units = semantic_tgt_split(
                src_text, tgt_text, embed_func, 
                similarity_threshold, min_tokens, max_tokens
            )
        else:
            # 단순 분할
            units = simple_tgt_split(tgt_text, min_tokens, max_tokens)
        
        logger.debug(f"번역문 분할 완료: {len(units)}개 단위")
        return units
        
    except Exception as e:
        logger.error(f"번역문 분할 실패: {e}")
        return [tgt_text]  # 실패 시 원본 반환

def basic_src_split(text: str) -> List[str]:
    """기본 원문 분할"""
    
    # 1. 명확한 구분자로 분할
    delimiters = [
        '然後에',  # 시간 접속
        '然後',
        '이요',    # 병렬 접속
        '이며',
        '이고',
        '라가',    # 전환
        '라서',
        '면',      # 조건
        '이면',
        '하면',
        '則',      # 한문 접속사
        '而',
        '且',
        '또',
        '그리고',
        '하지만',
        '그러나'
    ]
    
    units = [text]
    
    for delimiter in delimiters:
        new_units = []
        for unit in units:
            parts = re.split(f'({re.escape(delimiter)})', unit)
            current = ""
            
            for part in parts:
                if part == delimiter:
                    if current:
                        new_units.append(current + part)
                        current = ""
                else:
                    current += part
            
            if current:
                new_units.append(current)
        
        units = [u.strip() for u in new_units if u.strip()]
    
    return units

def advanced_src_split(units: List[str]) -> List[str]:
    """고급 원문 분할"""
    
    advanced_units = []
    
    for unit in units:
        # 너무 긴 단위는 추가 분할
        if len(unit) > 30:
            # 한자어 + 조사 패턴으로 분할
            pattern = r'([\u4e00-\u9fff]+[\uac00-\ud7af]*)'
            parts = re.findall(pattern, unit)
            
            if len(parts) > 1:
                # 패턴 기반 분할 성공
                start = 0
                for part in parts:
                    pos = unit.find(part, start)
                    if pos > start:
                        # 패턴 사이의 텍스트도 포함
                        advanced_units.append(unit[start:pos + len(part)])
                    else:
                        advanced_units.append(part)
                    start = pos + len(part)
                
                if start < len(unit):
                    advanced_units.append(unit[start:])
            else:
                # 패턴 분할 실패 시 원본 유지
                advanced_units.append(unit)
        else:
            advanced_units.append(unit)
    
    return [u.strip() for u in advanced_units if u.strip()]

def simple_tgt_split(text: str, min_tokens: int = 1, max_tokens: int = 10) -> List[str]:
    """단순 번역문 분할 (MeCab 기반)"""
    
    if mecab is None:
        # MeCab 없으면 기본 분할
        return basic_text_split(text, min_tokens, max_tokens)
    
    try:
        # MeCab으로 형태소 분석
        result = mecab.parse(text)
        morphemes = []
        
        for line in result.split('\n'):
            if line and line != 'EOS':
                parts = line.split('\t')
                if len(parts) >= 2:
                    surface = parts[0]
                    features = parts[1].split(',')
                    pos = features[0] if features else ''
                    
                    morphemes.append({
                        'surface': surface,
                        'pos': pos,
                        'features': features
                    })
        
        # 의미 단위로 그룹화
        units = group_morphemes_by_meaning(morphemes, min_tokens, max_tokens)
        
        return units
        
    except Exception as e:
        logger.error(f"MeCab 분할 실패: {e}")
        return basic_text_split(text, min_tokens, max_tokens)

def semantic_tgt_split(
    src_text: str, 
    tgt_text: str, 
    embed_func: Callable,
    similarity_threshold: float = 0.3,
    min_tokens: int = 1,
    max_tokens: int = 10
) -> List[str]:
    """의미 기반 번역문 분할"""
    
    try:
        # 1. 원문 단위 먼저 분할
        src_units = split_src_meaning_units(src_text, min_tokens, max_tokens)
        
        # 2. 번역문 기본 분할
        tgt_candidates = simple_tgt_split(tgt_text, min_tokens, max_tokens)
        
        # 3. 의미 유사도 기반 재조합
        if len(src_units) > 1 and len(tgt_candidates) > 1:
            tgt_units = semantic_regrouping(
                src_units, tgt_candidates, embed_func, similarity_threshold
            )
        else:
            tgt_units = tgt_candidates
        
        return tgt_units
        
    except Exception as e:
        logger.error(f"의미 기반 분할 실패: {e}")
        return simple_tgt_split(tgt_text, min_tokens, max_tokens)

def semantic_regrouping(
    src_units: List[str], 
    tgt_candidates: List[str], 
    embed_func: Callable,
    similarity_threshold: float = 0.3
) -> List[str]:
    """의미 유사도 기반 재조합"""
    
    try:
        # 모든 텍스트 임베딩
        all_texts = src_units + tgt_candidates
        embeddings = embed_func(all_texts)
        
        if len(embeddings) != len(all_texts):
            logger.warning("임베딩 수 불일치, 기본 분할 사용")
            return tgt_candidates
        
        src_embeddings = embeddings[:len(src_units)]
        tgt_embeddings = embeddings[len(src_units):]
        
        # 유사도 매트릭스 계산
        similarity_matrix = calculate_similarity_matrix(src_embeddings, tgt_embeddings)
        
        # 최적 그룹화 찾기
        tgt_groups = find_optimal_grouping(
            tgt_candidates, similarity_matrix, similarity_threshold
        )
        
        return tgt_groups
        
    except Exception as e:
        logger.error(f"의미 재조합 실패: {e}")
        return tgt_candidates

def group_morphemes_by_meaning(
    morphemes: List[dict], 
    min_tokens: int = 1, 
    max_tokens: int = 10
) -> List[str]:
    """형태소를 의미 단위로 그룹화"""
    
    if not morphemes:
        return []
    
    units = []
    current_unit = ""
    current_count = 0
    
    # 주요 품사 기준 그룹화
    boundary_pos = ['SF', 'SP', 'SS', 'VCP', 'VCN', 'EC', 'EF']  # 문장 경계 품사
    
    for morph in morphemes:
        surface = morph['surface']
        pos = morph['pos']
        
        current_unit += surface
        current_count += 1
        
        # 경계 조건 확인
        is_boundary = (
            pos in boundary_pos or  # 품사 경계
            current_count >= max_tokens or  # 최대 길이
            (current_count >= min_tokens and pos in ['NNG', 'NNP', 'VV', 'VA'])  # 의미 완료
        )
        
        if is_boundary and current_count >= min_tokens:
            units.append(current_unit.strip())
            current_unit = ""
            current_count = 0
    
    # 마지막 단위 처리
    if current_unit.strip():
        if units and current_count < min_tokens:
            # 너무 짧으면 이전 단위와 합치기
            units[-1] += current_unit
        else:
            units.append(current_unit.strip())
    
    return [u for u in units if u]

def basic_text_split(text: str, min_tokens: int = 1, max_tokens: int = 10) -> List[str]:
    """기본 텍스트 분할 (백업용)"""
    
    # 구두점 기준 분할
    delimiters = ['.', '!', '?', '。', '！', '？', ',', '，', ';', '：', ':']
    
    units = [text]
    
    for delimiter in delimiters:
        new_units = []
        for unit in units:
            if delimiter in unit:
                parts = unit.split(delimiter)
                for i, part in enumerate(parts):
                    if i < len(parts) - 1:  # 마지막이 아니면 구분자 포함
                        new_units.append(part + delimiter)
                    else:
                        if part.strip():  # 마지막 부분이 비어있지 않으면
                            new_units.append(part)
            else:
                new_units.append(unit)
        units = [u.strip() for u in new_units if u.strip()]
    
    return apply_length_constraints(units, min_tokens, max_tokens, is_src=False)

def apply_length_constraints(
    units: List[str], 
    min_tokens: int, 
    max_tokens: int, 
    is_src: bool = True
) -> List[str]:
    """길이 제한 적용"""
    
    if min_tokens <= 1 and max_tokens >= 50:
        return units  # 제한이 느슨하면 그대로 반환
    
    constrained_units = []
    
    for unit in units:
        unit_len = len(unit)
        
        if unit_len > max_tokens * 3:  # 대략적인 글자 수 기준
            # 너무 긴 단위는 분할
            mid = len(unit) // 2
            # 적절한 분할점 찾기
            for i in range(mid - 5, mid + 5):
                if i > 0 and i < len(unit) and unit[i] in ' ，,、':
                    constrained_units.append(unit[:i+1].strip())
                    constrained_units.append(unit[i+1:].strip())
                    break
            else:
                # 적절한 분할점 없으면 중간에서 분할
                constrained_units.append(unit[:mid].strip())
                constrained_units.append(unit[mid:].strip())
        else:
            constrained_units.append(unit)
    
    # 너무 짧은 단위는 합치기
    if min_tokens > 1:
        merged_units = []
        temp_unit = ""
        
        for unit in constrained_units:
            if len(temp_unit + unit) < min_tokens * 2:  # 대략적인 기준
                temp_unit += unit
            else:
                if temp_unit:
                    merged_units.append(temp_unit.strip())
                temp_unit = unit
        
        if temp_unit:
            merged_units.append(temp_unit.strip())
        
        constrained_units = merged_units
    
    return [u for u in constrained_units if u.strip()]

def calculate_similarity_matrix(embeddings1: List, embeddings2: List) -> np.ndarray:
    """유사도 매트릭스 계산"""
    
    try:
        emb1 = np.array(embeddings1)
        emb2 = np.array(embeddings2)
        
        # 정규화
        emb1_norm = emb1 / np.linalg.norm(emb1, axis=1, keepdims=True)
        emb2_norm = emb2 / np.linalg.norm(emb2, axis=1, keepdims=True)
        
        # 코사인 유사도
        similarity = np.dot(emb1_norm, emb2_norm.T)
        
        # 0~1 범위로 정규화
        similarity = (similarity + 1) / 2
        
        return similarity
        
    except Exception as e:
        logger.error(f"유사도 계산 실패: {e}")
        return np.zeros((len(embeddings1), len(embeddings2)))

def find_optimal_grouping(
    candidates: List[str], 
    similarity_matrix: np.ndarray,
    threshold: float = 0.3
) -> List[str]:
    """최적 그룹화 찾기"""
    
    try:
        # 단순 그리디 그룹화
        groups = []
        used = set()
        
        # 유사도가 높은 순서로 그룹화
        for i in range(len(candidates)):
            if i not in used:
                current_group = [candidates[i]]
                used.add(i)
                
                # 유사한 후보들 찾기
                for j in range(i + 1, len(candidates)):
                    if j not in used:
                        # 그룹 내 평균 유사도 계산
                        avg_sim = np.mean([similarity_matrix[k % similarity_matrix.shape[0], j % similarity_matrix.shape[1]] 
                                         for k in range(len(current_group))])
                        
                        if avg_sim >= threshold:
                            current_group.append(candidates[j])
                            used.add(j)
                
                # 그룹을 하나의 단위로 합치기
                if len(current_group) > 1:
                    groups.append(''.join(current_group))
                else:
                    groups.append(current_group[0])
        
        return groups
        
    except Exception as e:
        logger.error(f"최적 그룹화 실패: {e}")
        return candidates

if __name__ == "__main__":
    # 테스트
    logging.basicConfig(level=logging.DEBUG)
    
    print("🧪 토크나이저 매개변수 테스트")
    
    test_src = "興者는 喩衆民之不從襄公政令者는 得周禮以敎之면 則服이라"
    test_tgt = "興한 것은 襄公의 政令을 따르지 않는 백성들은 <군주가> 周禮를 따라 교화시키면 복종한다는 것을 비유한 것이다."
    
    print(f"\n원문: {test_src}")
    print(f"번역: {test_tgt}")
    
    # 다양한 매개변수로 테스트
    for min_tok, max_tok in [(1, 5), (2, 8), (1, 15)]:
        print(f"\n--- min_tokens={min_tok}, max_tokens={max_tok} ---")
        
        src_units = split_src_meaning_units(test_src, min_tok, max_tok)
        print(f"원문 분할: {src_units}")
        
        tgt_units = split_tgt_meaning_units(
            test_src, test_tgt, 
            embed_func=None, use_semantic=False,
            min_tokens=min_tok, max_tokens=max_tok
        )
        print(f"번역 분할: {tgt_units}")