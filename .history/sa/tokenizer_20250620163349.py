"""텍스트 토크나이징 모듈 - regex 모듈 사용"""

import jieba
import MeCab
import logging
import re
import regex  # 🆕 유니코드 속성 정규식 지원
from typing import List, Optional, Callable
import numpy as np

logger = logging.getLogger(__name__)

# MeCab 초기화
try:
    mecab = MeCab.Tagger()
    logger.info("✅ MeCab 초기화 성공")
except Exception as e:
    logger.warning(f"⚠️ MeCab 초기화 실패: {e}")
    mecab = None

def split_src_meaning_units(
    text: str, 
    min_tokens: int = 1, 
    max_tokens: int = 10,
    use_advanced: bool = True
) -> List[str]:
    """원문을 의미 단위로 분할"""
    
    if not text or not text.strip():
        return []
    
    try:
        # 🎯 한문 구문 경계 패턴
        patterns = [
            r'然後에',
            r'이요(?=\s|\p{Han}|\p{Hangul}|$)',  # 뒤에 공백, 한자, 한글, 끝
            r'이라가',
            r'이면',
            r'하면',
            r'則(?=\s|\p{Han}|\p{Hangul})',      # 한문 접속사
            r'而(?=\s|\p{Han}|\p{Hangul})',
            r'且(?=\s|\p{Han}|\p{Hangul})'
        ]
        
        units = [text]
        
        for pattern in patterns:
            new_units = []
            for unit in units:
                if regex.search(pattern, unit):
                    parts = regex.split(f'({pattern})', unit)
                    current = ""
                    for part in parts:
                        if regex.match(pattern, part):
                            if current:
                                new_units.append(current + part)
                                current = ""
                        else:
                            current += part
                    if current:
                        new_units.append(current)
                else:
                    new_units.append(unit)
            units = [u.strip() for u in new_units if u.strip()]
        
        # 🆕 고급 분할: 한자어 + 조사 단위
        if use_advanced:
            units = _advanced_han_split(units)
        
        return units
        
    except Exception as e:
        logger.error(f"❌ 원문 분할 실패: {e}")
        return [text]

def _advanced_han_split(units: List[str]) -> List[str]:
    """한자어 + 조사 단위로 고급 분할"""
    
    advanced_units = []
    
    for unit in units:
        if len(unit) > 15:  # 긴 단위만 추가 분할
            # 🆕 한자어 + 한글 조사/어미 패턴
            pattern = r'(\p{Han}+\p{Hangul}*(?:이라|이요|에서|라서|하여|면서|에|는|은|이|가)?)'
            matches = regex.findall(pattern, unit)
            
            if len(matches) > 1:
                remaining = unit
                for match in matches:
                    if match in remaining:
                        pos = remaining.find(match)
                        if pos > 0:
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

def split_tgt_meaning_units(
    src_text: str,
    tgt_text: str,
    embed_func: Optional[Callable] = None,
    use_semantic: bool = True,
    min_tokens: int = 1,
    max_tokens: int = 10,
    similarity_threshold: float = 0.3
) -> List[str]:
    """번역문을 의미 단위로 분할"""
    
    if not tgt_text or not tgt_text.strip():
        return []
    
    try:
        if mecab:
            return _mecab_split_with_han_awareness(tgt_text, max_tokens)
        else:
            return _basic_split_with_regex(tgt_text, max_tokens)
        
    except Exception as e:
        logger.error(f"❌ 번역문 분할 실패: {e}")
        return [tgt_text]

def _mecab_split_with_han_awareness(text: str, max_tokens: int) -> List[str]:
    """MeCab 분할 + 한자 인식"""
    
    result = mecab.parse(text)
    morphemes = []
    
    for line in result.split('\n'):
        if line and line != 'EOS':
            parts = line.split('\t')
            if len(parts) >= 2:
                surface = parts[0]
                pos = parts[1].split(',')[0]
                morphemes.append((surface, pos))
    
    # 의미 단위로 그룹화
    units = []
    current_unit = ""
    
    for i, (surface, pos) in enumerate(morphemes):
        current_unit += surface
        
        # 🆕 한자 + 조사 특별 처리
        is_han_particle_boundary = False
        if _is_han(surface) and i + 1 < len(morphemes):
            next_surface, next_pos = morphemes[i + 1]
            if _is_particle(next_surface):
                # 다음이 조사면 함께 묶기
                current_unit += next_surface
                morphemes[i + 1] = ('', '')  # 다음 토큰 스킵 표시
                is_han_particle_boundary = True
        
        # 경계 조건
        is_boundary = (
            pos in ['JKS', 'JKO', 'JKC', 'JX', 'SF', 'SP'] or  # 조사, 구두점
            (pos.startswith('V') and surface in ['다', '고', '며']) or  # 어미
            is_han_particle_boundary or
            len(current_unit) >= max_tokens * 2
        )
        
        if is_boundary and current_unit.strip():
            units.append(current_unit.strip())
            current_unit = ""
    
    if current_unit.strip():
        units.append(current_unit.strip())
    
    return [u for u in units if u]

def _basic_split_with_regex(text: str, max_tokens: int) -> List[str]:
    """regex 기반 기본 분할"""
    
    # 🆕 한자+한글, 구두점 패턴으로 분할
    patterns = [
        r'([.!?。！？]+)',                    # 구두점
        r'(\p{Han}+\p{Hangul}*)',            # 한자+한글 조합
        r'(\p{Hangul}+(?:다|고|며|지만))',    # 한글 어미
        r'([,，;：:]+)'                       # 쉼표류
    ]
    
    units = [text]
    
    for pattern in patterns:
        new_units = []
        for unit in units:
            if regex.search(pattern, unit):
                parts = regex.split(pattern, unit)
                current = ""
                for part in parts:
                    if regex.match(pattern, part):
                        current += part
                        if len(current) >= max_tokens or part in ['.', '!', '?', '。']:
                            new_units.append(current.strip())
                            current = ""
                    else:
                        current += part
                if current.strip():
                    new_units.append(current.strip())
            else:
                new_units.append(unit)
        units = [u.strip() for u in new_units if u.strip()]
    
    return units

def _is_han(token: str) -> bool:
    """한자 포함 여부 - regex 사용"""
    return bool(regex.search(r'\p{Han}', token))

def _is_hangul(token: str) -> bool:
    """한글 포함 여부 - regex 사용"""
    return bool(regex.search(r'\p{Hangul}', token))

def _is_particle(token: str) -> bool:
    """조사/어미 여부"""
    particles = ['은', '는', '이', '가', '을', '를', '에', '에서', '으로', '로', 
                '와', '과', '의', '도', '만', '부터', '까지', '라', '이라']
    return token in particles or (len(token) <= 2 and _is_hangul(token))

if __name__ == "__main__":
    # 테스트
    logging.basicConfig(level=logging.INFO)
    
    test_cases = [
        "興也라",
        "蒹은 薕(렴)이요 葭는 蘆也라", 
        "白露凝戾爲霜然後에 歲事成이요"
    ]
    
    for text in test_cases:
        print(f"\n원문: {text}")
        units = split_src_meaning_units(text)
        print(f"분할: {units}")
        
        # 한자/한글 검출 테스트
        print(f"한자 포함: {_is_han(text)}")
        print(f"한글 포함: {_is_hangul(text)}")