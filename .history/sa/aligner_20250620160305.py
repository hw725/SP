"""토큰 정렬 모듈"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from embedder import get_embeddings, batch_similarity  # 절대 임포트

logger = logging.getLogger(__name__)

def align_tokens_with_embeddings(
    src_units: List[str],
    tgt_units: List[str],
    src_text: str = "",
    tgt_text: str = "",
    threshold: float = 0.3
) -> List[Dict]:
    """
    임베딩 기반 토큰 정렬
    
    Args:
        src_units: 원문 토큰 리스트
        tgt_units: 번역문 토큰 리스트
        src_text: 전체 원문 (참고용)
        tgt_text: 전체 번역문 (참고용)
        threshold: 정렬 임계값
    
    Returns:
        List[Dict]: 정렬 결과
    """
    try:
        if not src_units or not tgt_units:
            logger.warning("⚠️ 빈 토큰 리스트")
            return []
        
        logger.info(f"🔗 토큰 정렬 시작: {len(src_units)} → {len(tgt_units)}")
        
        # 유사도 매트릭스 계산
        similarity_matrix = batch_similarity(src_units, tgt_units)
        
        if similarity_matrix.size == 0:
            logger.error("❌ 유사도 매트릭스 계산 실패")
            return []
        
        # 최적 정렬 찾기 (헝가리안 알고리즘 대신 그리디)
        alignments = greedy_alignment(
            src_units, tgt_units, 
            similarity_matrix, threshold
        )
        
        logger.info(f"✅ 정렬 완료: {len(alignments)}개 쌍")
        return alignments
        
    except Exception as e:
        logger.error(f"❌ 토큰 정렬 실패: {e}")
        return []

def greedy_alignment(
    src_units: List[str],
    tgt_units: List[str],
    similarity_matrix: np.ndarray,
    threshold: float = 0.3
) -> List[Dict]:
    """
    그리디 알고리즘 기반 정렬
    
    Args:
        src_units: 원문 토큰
        tgt_units: 번역문 토큰
        similarity_matrix: 유사도 매트릭스
        threshold: 임계값
    
    Returns:
        List[Dict]: 정렬 결과
    """
    alignments = []
    used_src = set()
    used_tgt = set()
    
    try:
        # 유사도가 높은 순서로 정렬
        positions = []
        for i in range(len(src_units)):
            for j in range(len(tgt_units)):
                if similarity_matrix[i, j] >= threshold:
                    positions.append((i, j, similarity_matrix[i, j]))
        
        # 유사도 내림차순 정렬
        positions.sort(key=lambda x: x[2], reverse=True)
        
        # 그리디 선택
        for src_idx, tgt_idx, score in positions:
            if src_idx not in used_src and tgt_idx not in used_tgt:
                alignments.append({
                    'src_idx': int(src_idx),
                    'tgt_idx': int(tgt_idx),
                    'src_text': src_units[src_idx],
                    'tgt_text': tgt_units[tgt_idx],
                    'confidence': float(score)
                })
                used_src.add(src_idx)
                used_tgt.add(tgt_idx)
        
        # 정렬되지 않은 토큰 처리 (낮은 신뢰도로 추가)
        for i, src_unit in enumerate(src_units):
            if i not in used_src:
                # 가장 유사한 미사용 타겟 찾기
                best_j = -1
                best_score = 0
                for j in range(len(tgt_units)):
                    if j not in used_tgt and similarity_matrix[i, j] > best_score:
                        best_j = j
                        best_score = similarity_matrix[i, j]
                
                if best_j >= 0:
                    alignments.append({
                        'src_idx': int(i),
                        'tgt_idx': int(best_j),
                        'src_text': src_unit,
                        'tgt_text': tgt_units[best_j],
                        'confidence': float(best_score)
                    })
                    used_tgt.add(best_j)
        
        # 정렬 결과를 src_idx 순서로 정렬
        alignments.sort(key=lambda x: x['src_idx'])
        
        return alignments
        
    except Exception as e:
        logger.error(f"❌ 그리디 정렬 실패: {e}")
        return []

def simple_alignment(
    src_units: List[str],
    tgt_units: List[str]
) -> List[Dict]:
    """
    단순 순서 기반 정렬 (백업용)
    
    Args:
        src_units: 원문 토큰
        tgt_units: 번역문 토큰
    
    Returns:
        List[Dict]: 정렬 결과
    """
    alignments = []
    
    try:
        min_len = min(len(src_units), len(tgt_units))
        
        for i in range(min_len):
            alignments.append({
                'src_idx': i,
                'tgt_idx': i,
                'src_text': src_units[i],
                'tgt_text': tgt_units[i],
                'confidence': 0.5  # 기본 신뢰도
            })
        
        return alignments
        
    except Exception as e:
        logger.error(f"❌ 단순 정렬 실패: {e}")
        return []

if __name__ == "__main__":
    # 테스트
    logging.basicConfig(level=logging.INFO)
    
    src_test = ["興也라"]
    tgt_test = ["興이", "다."]
    
    print("🧪 정렬 테스트")
    alignments = align_tokens_with_embeddings(src_test, tgt_test)
    
    for align in alignments:
        print(f"✅ {align['src_text']} → {align['tgt_text']} (신뢰도: {align['confidence']:.3f})")