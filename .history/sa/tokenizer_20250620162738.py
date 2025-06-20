"""개선된 텍스트 토크나이징 모듈"""

import jieba
import MeCab
import logging
import re
from typing import List, Optional, Callable, Dict, Tuple
import numpy as np

# 로깅 설정
logger = logging.getLogger(__name__)

# MeCab 초기화
try:
    mecab = MeCab.Tagger()
    logger.info("✅ MeCab 초기화 성공")
except Exception as e:
    logger.warning(f"⚠️ MeCab 초기화 실패: {e}")
    mecab = None

class ImprovedTokenizer:
    """개선된 토크나이저 클래스"""
    
    def __init__(self):
        self.mecab = mecab
        
        # 한문 구문 분할 패턴
        self.classical_patterns = [
            r'然後에?',      # 시간 접속
            r'然後',
            r'이요(?!\w)',   # 병렬 접속 (뒤에 문자가 없을 때)
            r'이며',
            r'이고',
            r'라가(?!\w)',   # 전환
            r'라서',
            r'(?<!.)면(?!\w)',     # 조건 (앞에 문자가 없고 뒤에 문자가 없을 때)
            r'이면',
            r'하면',
            r'則(?=\s|\w)',   # 한문 접속사 (뒤에 공백이나 문자)
            r'而(?=\s|\w)',
            r'且(?=\s|\w)',
        ]
        
        # 한국어 구문 경계 패턴
        self.korean_boundaries = [
            r'(?<=다)\s*(?=[가-힣])',  # 서술어 뒤
            r'(?<=[.!?])\s*',         # 구두점 뒤
            r'(?<=고)\s+(?=[가-힣])',  # 연결어미 뒤
            r'(?<=며)\s+(?=[가-힣])',
            r'(?<=지만)\s+(?=[가-힣])',
            r'(?<=하여)\s+(?=[가-힣])',
        ]

    def split_src_meaning_units(
        self, 
        text: str, 
        min_tokens: int = 1, 
        max_tokens: int = 15,
        use_advanced: bool = True
    ) -> List[str]:
        """개선된 원문 의미 단위 분할"""
        
        if not text or not text.strip():
            return []
        
        try:
            logger.debug(f"🔤 원문 분할 시작: {text[:50]}...")
            
            # 1단계: 구문 패턴 기반 기본 분할
            units = self._split_by_classical_patterns(text)
            
            # 2단계: 고급 분할 (한자어 + 조사 단위)
            if use_advanced:
                units = self._advanced_src_split(units)
            
            # 3단계: 길이 제한 적용
            units = self._apply_length_constraints(units, min_tokens, max_tokens, is_src=True)
            
            # 4단계: 빈 단위 제거 및 정리
            units = [u.strip() for u in units if u.strip()]
            
            logger.debug(f"✅ 원문 분할 완료: {len(units)}개 단위")
            return units
            
        except Exception as e:
            logger.error(f"❌ 원문 분할 실패: {e}")
            return [text]  # 실패 시 원본 반환

    def split_tgt_meaning_units(
        self,
        src_text: str,
        tgt_text: str,
        embed_func: Optional[Callable] = None,
        use_semantic: bool = True,
        min_tokens: int = 1,
        max_tokens: int = 15,
        similarity_threshold: float = 0.3
    ) -> List[str]:
        """개선된 번역문 의미 단위 분할"""
        
        if not tgt_text or not tgt_text.strip():
            return []
        
        try:
            logger.debug(f"🔤 번역문 분할 시작: {tgt_text[:50]}...")
            
            if use_semantic and embed_func is not None:
                # 의미 기반 분할
                units = self._semantic_tgt_split(
                    src_text, tgt_text, embed_func, 
                    similarity_threshold, min_tokens, max_tokens
                )
            else:
                # 개선된 단순 분할
                units = self._improved_simple_tgt_split(tgt_text, min_tokens, max_tokens)
            
            logger.debug(f"✅ 번역문 분할 완료: {len(units)}개 단위")
            return units
            
        except Exception as e:
            logger.error(f"❌ 번역문 분할 실패: {e}")
            return [tgt_text]  # 실패 시 원본 반환

    def _split_by_classical_patterns(self, text: str) -> List[str]:
        """한문 구문 패턴 기반 분할"""
        
        units = [text]
        
        for pattern in self.classical_patterns:
            new_units = []
            
            for unit in units:
                # 패턴으로 분할하되 구분자 보존
                parts = re.split(f'({pattern})', unit)
                
                current = ""
                for part in parts:
                    if re.match(pattern, part):
                        # 구분자는 앞 단위에 붙임
                        if current:
                            new_units.append(current + part)
                            current = ""
                        else:
                            # 단독 구분자는 따로 처리
                            new_units.append(part)
                    else:
                        current += part
                
                if current:
                    new_units.append(current)
            
            units = [u.strip() for u in new_units if u.strip()]
        
        return units

    def _advanced_src_split(self, units: List[str]) -> List[str]:
        """고급 원문 분할 - 한자어 + 조사 단위"""
        
        advanced_units = []
        
        for unit in units:
            if len(unit) > 20:  # 긴 단위만 추가 분할
                # 한자어 + 조사/어미 패턴으로 분할
                pattern = r'([\u4e00-\u9fff]+[가-힣]*(?:이라|이요|에서|라서|하여|면서)?)'
                matches = re.findall(pattern, unit)
                
                if len(matches) > 1:
                    # 패턴 매칭된 부분들 추출
                    remaining = unit
                    for match in matches:
                        if match in remaining:
                            pos = remaining.find(match)
                            if pos > 0:
                                # 앞부분이 있으면 추가
                                advanced_units.append(remaining[:pos].strip())
                            advanced_units.append(match)
                            remaining = remaining[pos + len(match):]
                    
                    if remaining.strip():
                        advanced_units.append(remaining.strip())
                else:
                    advanced_units.append(unit)
            else:
                advanced_units.append(unit)
        
        return [u for u in advanced_units if u.strip()]

    def _improved_simple_tgt_split(self, text: str, min_tokens: int, max_tokens: int) -> List[str]:
        """개선된 단순 번역문 분할"""
        
        if not self.mecab:
            return self._basic_tgt_split(text, min_tokens, max_tokens)
        
        try:
            # MeCab 형태소 분석
            morphemes = self._analyze_morphemes(text)
            
            # 의미 단위로 그룹화 (공백 보존)
            units = self._group_morphemes_meaningfully(morphemes, min_tokens, max_tokens)
            
            return units
            
        except Exception as e:
            logger.error(f"❌ MeCab 분할 실패: {e}")
            return self._basic_tgt_split(text, min_tokens, max_tokens)

    def _analyze_morphemes(self, text: str) -> List[Dict]:
        """형태소 분석 결과를 구조화"""
        
        result = self.mecab.parse(text)
        morphemes = []
        position = 0
        
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
                        'features': features,
                        'start': position,
                        'end': position + len(surface)
                    })
                    position += len(surface)
        
        return morphemes

    def _group_morphemes_meaningfully(
        self, 
        morphemes: List[Dict], 
        min_tokens: int, 
        max_tokens: int
    ) -> List[str]:
        """형태소를 의미 있는 단위로 그룹화 (공백 보존)"""
        
        if not morphemes:
            return []
        
        units = []
        current_group = []
        current_text = ""
        
        # 의미 경계 품사들
        boundary_pos = ['SF', 'SP', 'SS', 'EC', 'EF', 'ETM', 'ETN']
        # 독립적 의미 품사들
        content_pos = ['NNG', 'NNP', 'VV', 'VA', 'MAG', 'MM']
        
        i = 0
        while i < len(morphemes):
            morph = morphemes[i]
            surface = morph['surface']
            pos = morph['pos']
            
            current_group.append(morph)
            current_text += surface
            
            # 경계 조건 확인
            is_boundary = (
                pos in boundary_pos or  # 품사 경계
                len(current_group) >= max_tokens or  # 최대 길이
                (len(current_group) >= min_tokens and 
                 pos in content_pos and 
                 i + 1 < len(morphemes) and 
                 morphemes[i + 1]['pos'] in content_pos)  # 내용어 연속
            )
            
            # 한자어 + 조사 패턴 특별 처리
            if (self._is_hanja(surface) and 
                i + 1 < len(morphemes) and 
                self._is_particle(morphemes[i + 1]['surface'])):
                # 다음 토큰(조사)까지 포함
                i += 1
                next_morph = morphemes[i]
                current_group.append(next_morph)
                current_text += next_morph['surface']
                is_boundary = True
            
            if is_boundary and len(current_group) >= min_tokens:
                units.append(current_text.strip())
                current_group = []
                current_text = ""
            
            i += 1
        
        # 마지막 그룹 처리
        if current_group and current_text.strip():
            if units and len(current_group) < min_tokens:
                # 너무 짧으면 이전 단위와 합치기
                units[-1] = units[-1] + current_text
            else:
                units.append(current_text.strip())
        
        return [u for u in units if u.strip()]

    def _semantic_tgt_split(
        self,
        src_text: str, 
        tgt_text: str, 
        embed_func: Callable,
        similarity_threshold: float,
        min_tokens: int,
        max_tokens: int
    ) -> List[str]:
        """의미 기반 번역문 분할 - 원문 단위를 고려"""
        
        try:
            # 1. 원문 단위 분할
            src_units = self.split_src_meaning_units(src_text, min_tokens, max_tokens)
            
            # 2. 번역문 기본 분할
            tgt_candidates = self._improved_simple_tgt_split(tgt_text, 1, max_tokens // 2)
            
            # 3. 원문 단위 수에 따른 적응적 재조합
            if len(src_units) > 1 and len(tgt_candidates) > len(src_units):
                tgt_units = self._adaptive_regrouping(
                    src_units, tgt_candidates, embed_func, similarity_threshold
                )
            else:
                tgt_units = tgt_candidates
            
            return tgt_units
            
        except Exception as e:
            logger.error(f"❌ 의미 기반 분할 실패: {e}")
            return self._improved_simple_tgt_split(tgt_text, min_tokens, max_tokens)

    def _adaptive_regrouping(
        self,
        src_units: List[str], 
        tgt_candidates: List[str], 
        embed_func: Callable,
        similarity_threshold: float
    ) -> List[str]:
        """적응적 재조합 - 원문 구조를 고려한 번역문 재구성"""
        
        try:
            # 목표 분할 수 결정
            src_count = len(src_units)
            cand_count = len(tgt_candidates)
            
            if cand_count <= src_count:
                return tgt_candidates
            
            # 목표: 원문 단위 수의 80-120% 범위
            target_count = max(2, min(src_count + 2, cand_count // 2))
            
            # 의미 유사도 기반 그룹화
            embeddings = embed_func(src_units + tgt_candidates)
            src_embeddings = embeddings[:src_count]
            tgt_embeddings = embeddings[src_count:]
            
            # 유사도 매트릭스 계산
            similarity_matrix = self._calculate_similarity_matrix(src_embeddings, tgt_embeddings)
            
            # 최적 그룹화
            grouped_units = self._find_optimal_grouping(
                tgt_candidates, similarity_matrix, target_count, similarity_threshold
            )
            
            return grouped_units
            
        except Exception as e:
            logger.error(f"❌ 적응적 재조합 실패: {e}")
            return tgt_candidates

    def _basic_tgt_split(self, text: str, min_tokens: int, max_tokens: int) -> List[str]:
        """기본 번역문 분할 (MeCab 없을 때)"""
        
        # 구두점과 접속사 기준 분할
        patterns = [
            r'([.!?。！？]+)',      # 구두점
            r'(그런데|하지만|따라서|그러므로|즉|또한)',  # 접속사
            r'([,，;：:]+)',        # 쉼표류
        ]
        
        units = [text]
        
        for pattern in patterns:
            new_units = []
            
            for unit in units:
                parts = re.split(pattern, unit)
                current = ""
                
                for part in parts:
                    if re.match(pattern, part):
                        if current:
                            new_units.append(current + part)
                            current = ""
                    else:
                        current += part
                
                if current:
                    new_units.append(current)
            
            units = [u.strip() for u in new_units if u.strip()]
        
        return self._apply_length_constraints(units, min_tokens, max_tokens, is_src=False)

    def _apply_length_constraints(
        self, 
        units: List[str], 
        min_tokens: int, 
        max_tokens: int, 
        is_src: bool = True
    ) -> List[str]:
        """길이 제한 적용"""
        
        if min_tokens <= 1 and max_tokens >= 50:
            return units
        
        # 최대 길이 제한
        constrained_units = []
        
        for unit in units:
            if len(unit) > max_tokens * 4:  # 글자 수 기준 (대략)
                # 긴 단위 분할
                split_points = self._find_split_points(unit)
                if split_points:
                    start = 0
                    for point in split_points:
                        constrained_units.append(unit[start:point].strip())
                        start = point
                    if start < len(unit):
                        constrained_units.append(unit[start:].strip())
                else:
                    # 분할점 없으면 중간에서 분할
                    mid = len(unit) // 2
                    constrained_units.append(unit[:mid].strip())
                    constrained_units.append(unit[mid:].strip())
            else:
                constrained_units.append(unit)
        
        # 최소 길이 제한 (병합)
        if min_tokens > 1:
            merged_units = []
            temp = ""
            
            for unit in constrained_units:
                if len(temp + unit) < min_tokens * 3:  # 대략적 기준
                    temp += unit
                else:
                    if temp:
                        merged_units.append(temp.strip())
                    temp = unit
            
            if temp:
                merged_units.append(temp.strip())
            
            constrained_units = merged_units
        
        return [u for u in constrained_units if u.strip()]

    def _find_split_points(self, text: str) -> List[int]:
        """분할점 찾기"""
        
        split_points = []
        
        # 구두점 위치
        for match in re.finditer(r'[.!?。！？,，]', text):
            split_points.append(match.end())
        
        # 접속 표현 위치
        connectors = ['그런데', '하지만', '따라서', '그리고', '또한']
        for connector in connectors:
            for match in re.finditer(connector, text):
                split_points.append(match.start())
        
        # 중복 제거 및 정렬
        split_points = sorted(set(split_points))
        
        # 너무 가까운 분할점 제거
        filtered_points = []
        last_point = 0
        
        for point in split_points:
            if point - last_point > 10:  # 최소 간격
                filtered_points.append(point)
                last_point = point
        
        return filtered_points

    def _calculate_similarity_matrix(self, emb1: List, emb2: List) -> np.ndarray:
        """유사도 매트릭스 계산"""
        
        try:
            embeddings1 = np.array(emb1)
            embeddings2 = np.array(emb2)
            
            # 정규화
            emb1_norm = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
            emb2_norm = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
            
            # 코사인 유사도
            similarity = np.dot(emb1_norm, emb2_norm.T)
            
            # 0~1 범위로 정규화
            similarity = (similarity + 1) / 2
            
            return similarity
            
        except Exception as e:
            logger.error(f"❌ 유사도 계산 실패: {e}")
            return np.zeros((len(emb1), len(emb2)))

    def _find_optimal_grouping(
        self,
        candidates: List[str], 
        similarity_matrix: np.ndarray,
        target_count: int,
        threshold: float
    ) -> List[str]:
        """최적 그룹화 - 동적 계획법 기반"""
        
        try:
            # 그리디 그룹화
            groups = []
            used = set()
            
            # 유사도가 높은 인접 후보들을 우선 그룹화
            for i in range(len(candidates)):
                if i in used:
                    continue
                
                current_group = [candidates[i]]
                used.add(i)
                
                # 인접한 후보 중 유사도가 높은 것들 추가
                for j in range(i + 1, min(i + 3, len(candidates))):  # 최대 3개까지만 확인
                    if j not in used:
                        max_sim = 0
                        if i < similarity_matrix.shape[1]:
                            max_sim = max(similarity_matrix[:, j % similarity_matrix.shape[1]])
                        
                        if max_sim >= threshold:
                            current_group.append(candidates[j])
                            used.add(j)
                            break
                
                # 그룹 결합
                groups.append(''.join(current_group))
            
            # 목표 개수에 맞춰 조정
            while len(groups) > target_count and len(groups) > 1:
                # 가장 짧은 두 그룹 병합
                min_len = min(len(g) for g in groups)
                for i in range(len(groups) - 1):
                    if len(groups[i]) == min_len:
                        groups[i] = groups[i] + groups[i + 1]
                        groups.pop(i + 1)
                        break
            
            return groups
            
        except Exception as e:
            logger.error(f"❌ 최적 그룹화 실패: {e}")
            return candidates

    def _is_hanja(self, token: str) -> bool:
        """한자 포함 여부"""
        return bool(re.search(r'[\u4e00-\u9fff]', token))
    
    def _is_particle(self, token: str) -> bool:
        """조사/어미 여부"""
        particles = ['은', '는', '이', '가', '을', '를', '에', '에서', '으로', '로', 
                    '와', '과', '의', '도', '만', '부터', '까지', '라', '이라']
        return token in particles or (len(token) <= 2 and re.search(r'[가-힣]', token))

# 전역 토크나이저 인스턴스
_tokenizer = ImprovedTokenizer()

def split_src_meaning_units(
    text: str, 
    min_tokens: int = 1, 
    max_tokens: int = 15,
    use_advanced: bool = True
) -> List[str]:
    """원문 의미 단위 분할 (전역 함수)"""
    return _tokenizer.split_src_meaning_units(text, min_tokens, max_tokens, use_advanced)

def split_tgt_meaning_units(
    src_text: str,
    tgt_text: str,
    embed_func: Optional[Callable] = None,
    use_semantic: bool = True,
    min_tokens: int = 1,
    max_tokens: int = 15,
    similarity_threshold: float = 0.3
) -> List[str]:
    """번역문 의미 단위 분할 (전역 함수)"""
    return _tokenizer.split_tgt_meaning_units(
        src_text, tgt_text, embed_func, use_semantic, 
        min_tokens, max_tokens, similarity_threshold
    )

if __name__ == "__main__":
    # 테스트
    logging.basicConfig(level=logging.DEBUG)
    
    print("🧪 개선된 토크나이저 테스트")
    print("=" * 60)
    
    test_cases = [
        (
            "興也라",
            "興이다."
        ),
        (
            "蒹은 薕(렴)이요 葭는 蘆也라",
            "蒹은 물억새이고 葭는 갈대이다."
        ),
        (
            "白露凝戾爲霜然後에 歲事成이요 國家待禮然後興이라",
            "白露가 얼어 서리가 된 뒤에야 歲事가 이루어지고 國家는 禮가 행해진 뒤에야 흥성한다."
        )
    ]
    
    for i, (src, tgt) in enumerate(test_cases, 1):
        print(f"\n📝 테스트 케이스 {i}:")
        print(f"원문: {src}")
        print(f"번역: {tgt}")
        
        # 원문 분할
        src_units = split_src_meaning_units(src, min_tokens=1, max_tokens=15)
        print(f"✅ 원문 분할: {src_units}")
        
        # 번역문 분할 (단순)
        tgt_units = split_tgt_meaning_units(
            src, tgt, embed_func=None, use_semantic=False, 
            min_tokens=1, max_tokens=10
        )
        print(f"✅ 번역 분할: {tgt_units}")