"""개선된 토큰 정렬 모듈"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from embedder import get_embeddings, batch_similarity

logger = logging.getLogger(__name__)

class ImprovedAligner:
    """개선된 정렬기 클래스"""
    
    def __init__(self):
        self.alignment_types = ['1:1', '1:N', 'N:1', 'N:M']
    
    def align_tokens_with_embeddings(
        self,
        src_units: List[str],
        tgt_units: List[str],
        src_text: str = "",
        tgt_text: str = "",
        threshold: float = 0.3,
        max_alignments: int = 50
    ) -> List[Dict]:
        """개선된 임베딩 기반 토큰 정렬"""
        
        try:
            if not src_units or not tgt_units:
                logger.warning("⚠️ 빈 토큰 리스트")
                return []
            
            logger.info(f"🔗 토큰 정렬 시작: {len(src_units)} → {len(tgt_units)}")
            
            # 유사도 매트릭스 계산
            similarity_matrix = batch_similarity(src_units, tgt_units)
            
            if similarity_matrix.size == 0:
                logger.error("❌ 유사도 매트릭스 계산 실패")
                return self._fallback_alignment(src_units, tgt_units)
            
            # 다양한 정렬 패턴 적용
            alignments = self._multi_pattern_alignment(
                src_units, tgt_units, similarity_matrix, threshold, max_alignments
            )
            
            # 정렬 품질 개선
            alignments = self._enhance_alignment_quality(
                alignments, src_units, tgt_units, similarity_matrix
            )
            
            logger.info(f"✅ 정렬 완료: {len(alignments)}개 쌍")
            return alignments
            
        except Exception as e:
            logger.error(f"❌ 토큰 정렬 실패: {e}")
            return self._fallback_alignment(src_units, tgt_units)
    
    def _multi_pattern_alignment(
        self,
        src_units: List[str],
        tgt_units: List[str],
        similarity_matrix: np.ndarray,
        threshold: float,
        max_alignments: int
    ) -> List[Dict]:
        """다중 패턴 정렬"""
        
        alignments = []
        src_used = set()
        tgt_used = set()
        
        # 1단계: 고신뢰도 1:1 정렬
        one_to_one = self._find_one_to_one_alignments(
            src_units, tgt_units, similarity_matrix, threshold + 0.2
        )
        
        for align in one_to_one:
            if align['src_idx'] not in src_used and align['tgt_idx'] not in tgt_used:
                alignments.append(align)
                src_used.add(align['src_idx'])
                tgt_used.add(align['tgt_idx'])
        
        # 2단계: 1:N 정렬 (하나의 원문 → 여러 번역)
        one_to_many = self._find_one_to_many_alignments(
            src_units, tgt_units, similarity_matrix, threshold, src_used, tgt_used
        )
        
        for align in one_to_many:
            if align['src_idx'] not in src_used:
                alignments.append(align)
                src_used.add(align['src_idx'])
                for tgt_idx in align['tgt_indices']:
                    tgt_used.add(tgt_idx)
        
        # 3단계: N:1 정렬 (여러 원문 → 하나의 번역)
        many_to_one = self._find_many_to_one_alignments(
            src_units, tgt_units, similarity_matrix, threshold, src_used, tgt_used
        )
        
        for align in many_to_one:
            if align['tgt_idx'] not in tgt_used:
                alignments.append(align)
                tgt_used.add(align['tgt_idx'])
                for src_idx in align['src_indices']:
                    src_used.add(src_idx)
        
        # 4단계: 잔여 단위들 처리
        remaining_alignments = self._align_remaining_units(
            src_units, tgt_units, similarity_matrix, 
            src_used, tgt_used, threshold - 0.1
        )
        
        alignments.extend(remaining_alignments)
        
        return alignments[:max_alignments]  # 최대 개수 제한
    
    def _find_one_to_one_alignments(
        self,
        src_units: List[str],
        tgt_units: List[str],
        similarity_matrix: np.ndarray,
        threshold: float
    ) -> List[Dict]:
        """1:1 정렬 찾기"""
        
        alignments = []
        
        # 유사도가 높은 순서로 정렬
        candidates = []
        for i in range(len(src_units)):
            for j in range(len(tgt_units)):
                if similarity_matrix[i, j] >= threshold:
                    candidates.append((i, j, similarity_matrix[i, j]))
        
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        used_src = set()
        used_tgt = set()
        
        for src_idx, tgt_idx, score in candidates:
            if src_idx not in used_src and tgt_idx not in used_tgt:
                alignments.append({
                    'type': '1:1',
                    'src_idx': src_idx,
                    'tgt_idx': tgt_idx,
                    'src_text': src_units[src_idx],
                    'tgt_text': tgt_units[tgt_idx],
                    'confidence': float(score),
                    'alignment_type': '1:1'
                })
                used_src.add(src_idx)
                used_tgt.add(tgt_idx)
        
        return alignments
    
    def _find_one_to_many_alignments(
        self,
        src_units: List[str],
        tgt_units: List[str],
        similarity_matrix: np.ndarray,
        threshold: float,
        src_used: set,
        tgt_used: set
    ) -> List[Dict]:
        """1:N 정렬 찾기"""
        
        alignments = []
        
        for src_idx in range(len(src_units)):
            if src_idx in src_used:
                continue
            
            # 이 원문과 유사한 번역 단위들 찾기
            candidates = []
            for tgt_idx in range(len(tgt_units)):
                if tgt_idx not in tgt_used and similarity_matrix[src_idx, tgt_idx] >= threshold:
                    candidates.append((tgt_idx, similarity_matrix[src_idx, tgt_idx]))
            
            if len(candidates) >= 2:  # 2개 이상일 때만 1:N
                # 인접한 번역 단위들 우선 선택
                candidates.sort(key=lambda x: (x[1], -abs(x[0] - src_idx)), reverse=True)
                
                selected_tgt = []
                total_score = 0
                
                for tgt_idx, score in candidates[:3]:  # 최대 3개까지
                    if tgt_idx not in tgt_used:
                        selected_tgt.append(tgt_idx)
                        total_score += score
                        if len(selected_tgt) >= 2:  # 적어도 2개
                            break
                
                if len(selected_tgt) >= 2:
                    tgt_texts = [tgt_units[idx] for idx in selected_tgt]
                    
                    alignments.append({
                        'type': '1:N',
                        'src_idx': src_idx,
                        'tgt_indices': selected_tgt,
                        'src_text': src_units[src_idx],
                        'tgt_text': ' | '.join(tgt_texts),
                        'confidence': float(total_score / len(selected_tgt)),
                        'alignment_type': f'1:{len(selected_tgt)}'
                    })
        
        return alignments
    
    def _find_many_to_one_alignments(
        self,
        src_units: List[str],
        tgt_units: List[str],
        similarity_matrix: np.ndarray,
        threshold: float,
        src_used: set,
        tgt_used: set
    ) -> List[Dict]:
        """N:1 정렬 찾기"""
        
        alignments = []
        
        for tgt_idx in range(len(tgt_units)):
            if tgt_idx in tgt_used:
                continue
            
            # 이 번역과 유사한 원문 단위들 찾기
            candidates = []
            for src_idx in range(len(src_units)):
                if src_idx not in src_used and similarity_matrix[src_idx, tgt_idx] >= threshold:
                    candidates.append((src_idx, similarity_matrix[src_idx, tgt_idx]))
            
            if len(candidates) >= 2:  # 2개 이상일 때만 N:1
                candidates.sort(key=lambda x: x[1], reverse=True)
                
                selected_src = []
                total_score = 0
                
                for src_idx, score in candidates[:3]:  # 최대 3개까지
                    if src_idx not in src_used:
                        selected_src.append(src_idx)
                        total_score += score
                        if len(selected_src) >= 2:
                            break
                
                if len(selected_src) >= 2:
                    src_texts = [src_units[idx] for idx in selected_src]
                    
                    alignments.append({
                        'type': 'N:1',
                        'src_indices': selected_src,
                        'tgt_idx': tgt_idx,
                        'src_text': ' | '.join(src_texts),
                        'tgt_text': tgt_units[tgt_idx],
                        'confidence': float(total_score / len(selected_src)),
                        'alignment_type': f'{len(selected_src)}:1'
                    })
        
        return alignments
    
    def _align_remaining_units(
        self,
        src_units: List[str],
        tgt_units: List[str],
        similarity_matrix: np.ndarray,
        src_used: set,
        tgt_used: set,
        threshold: float
    ) -> List[Dict]:
        """잔여 단위들 정렬"""
        
        alignments = []
        
        # 잔여 원문 단위들
        remaining_src = [i for i in range(len(src_units)) if i not in src_used]
        remaining_tgt = [j for j in range(len(tgt_units)) if j not in tgt_used]
        
        # 단순 순서 기반 정렬 시도
        for i, src_idx in enumerate(remaining_src):
            if i < len(remaining_tgt):
                tgt_idx = remaining_tgt[i]
                confidence = similarity_matrix[src_idx, tgt_idx] if similarity_matrix[src_idx, tgt_idx] > 0 else 0.1
                
                alignments.append({
                    'type': '1:1',
                    'src_idx': src_idx,
                    'tgt_idx': tgt_idx,
                    'src_text': src_units[src_idx],
                    'tgt_text': tgt_units[tgt_idx],
                    'confidence': float(confidence),
                    'alignment_type': '1:1-remaining'
                })
        
        return alignments
    
    def _enhance_alignment_quality(
        self,
        alignments: List[Dict],
        src_units: List[str],
        tgt_units: List[str],
        similarity_matrix: np.ndarray
    ) -> List[Dict]:
        """정렬 품질 향상"""
        
        enhanced = []
        
        for align in alignments:
            # 신뢰도 재계산
            enhanced_confidence = self._calculate_enhanced_confidence(
                align, src_units, tgt_units, similarity_matrix
            )
            
            align['confidence'] = enhanced_confidence
            align['quality_score'] = self._calculate_quality_score(align)
            
            enhanced.append(align)
        
        # 품질 점수 기준으로 정렬
        enhanced.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
        
        return enhanced
    
    def _calculate_enhanced_confidence(
        self,
        align: Dict,
        src_units: List[str],
        tgt_units: List[str],
        similarity_matrix: np.ndarray
    ) -> float:
        """강화된 신뢰도 계산"""
        
        base_confidence = align.get('confidence', 0.0)
        
        # 길이 비율 점수
        if align['type'] == '1:1':
            src_len = len(align['src_text'])
            tgt_len = len(align['tgt_text'])
            length_ratio = min(src_len, tgt_len) / max(src_len, tgt_len) if max(src_len, tgt_len) > 0 else 0
            length_score = length_ratio * 0.2
        else:
            length_score = 0.1  # 다중 정렬은 기본 점수
        
        # 어휘 중복 점수
        overlap_score = self._calculate_lexical_overlap(align) * 0.1
        
        # 위치 점수 (순서 일치도)
        position_score = self._calculate_position_score(align, len(src_units), len(tgt_units)) * 0.1
        
        return min(1.0, base_confidence + length_score + overlap_score + position_score)
    
    def _calculate_lexical_overlap(self, align: Dict) -> float:
        """어휘 중복도 계산"""
        
        src_text = align['src_text']
        tgt_text = align['tgt_text']
        
        # 한자 중복 확인
        src_hanja = set(re.findall(r'[\u4e00-\u9fff]', src_text))
        tgt_hanja = set(re.findall(r'[\u4e00-\u9fff]', tgt_text))
        
        if src_hanja and tgt_hanja:
            overlap = len(src_hanja & tgt_hanja) / len(src_hanja | tgt_hanja)
            return overlap
        
        return 0.0
    
    def _calculate_position_score(self, align: Dict, src_total: int, tgt_total: int) -> float:
        """위치 점수 계산"""
        
        if align['type'] == '1:1':
            src_pos = align['src_idx'] / src_total if src_total > 0 else 0
            tgt_pos = align['tgt_idx'] / tgt_total if tgt_total > 0 else 0
            position_diff = abs(src_pos - tgt_pos)
            return max(0, 1 - position_diff)
        
        return 0.5  # 다중 정렬 기본 점수
    
    def _calculate_quality_score(self, align: Dict) -> float:
        """전체 품질 점수 계산"""
        
        confidence = align.get('confidence', 0.0)
        alignment_type = align.get('alignment_type', '')
        
        # 정렬 타입별 가중치
        type_weights = {
            '1:1': 1.0,
            '1:2': 0.8,
            '1:3': 0.6,
            '2:1': 0.8,
            '3:1': 0.6
        }
        
        base_weight = type_weights.get(alignment_type.split('-')[0], 0.5)
        
        return confidence * base_weight
    
    def _fallback_alignment(self, src_units: List[str], tgt_units: List[str]) -> List[Dict]:
        """백업 정렬 (단순 순서 기반)"""
        
        alignments = []
        min_len = min(len(src_units), len(tgt_units))
        
        for i in range(min_len):
            alignments.append({
                'type': '1:1',
                'src_idx': i,
                'tgt_idx': i,
                'src_text': src_units[i],
                'tgt_text': tgt_units[i],
                'confidence': 0.3,
                'alignment_type': '1:1-fallback',
                'quality_score': 0.3
            })
        
        return alignments

# 전역 정렬기 인스턴스
_aligner = ImprovedAligner()

def align_tokens_with_embeddings(
    src_units: List[str],
    tgt_units: List[str],
    src_text: str = "",
    tgt_text: str = "",
    threshold: float = 0.3
) -> List[Dict]:
    """토큰 정렬 (전역 함수)"""
    return _aligner.align_tokens_with_embeddings(src_units, tgt_units, src_text, tgt_text, threshold)

if __name__ == "__main__":
    # 테스트
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 개선된 정렬기 테스트")
    
    src_test = ["興也라"]
    tgt_test = ["興이", "다."]
    
    alignments = align_tokens_with_embeddings(src_test, tgt_test)
    
    for align in alignments:
        print(f"✅ {align['src_text']} → {align['tgt_text']}")
        print(f"   타입: {align['alignment_type']}, 신뢰도: {align['confidence']:.3f}")