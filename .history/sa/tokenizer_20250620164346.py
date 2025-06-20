"""텍스트 토크나이징 모듈 - 공백 보존 개선"""

import jieba
import logging
import re
import regex
from typing import List, Optional, Callable
import numpy as np

logger = logging.getLogger(__name__)

# MeCab 초기화
mecab = None
try:
    import MeCab
    mecab = MeCab.Tagger()
    logger.info("✅ MeCab-ko 초기화 성공")
except ImportError:
    logger.warning("⚠️ MeCab 없음, 기본 토크나이저 사용")
    mecab = None
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
        # 한문 구문 경계 패턴
        patterns = [
            r'然後에',
            r'이요(?=\s|\p{Han}|\p{Hangul}|$)',
            r'이라가',
            r'이면',
            r'하면',
            r'則(?=\s|\p{Han}|\p{Hangul})',
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
        if len(unit) > 15:
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
    """번역문을 의미 단위로 분할 - 공백 보존 개선"""
    
    if not tgt_text or not tgt_text.strip():
        return []
    
    try:
        if mecab:
            return _mecab_split_preserve_spaces(tgt_text, max_tokens)
        else:
            return _basic_split_preserve_spaces(tgt_text, max_tokens)
        
    except Exception as e:
        logger.error(f"❌ 번역문 분할 실패: {e}")
        return [tgt_text]

def _mecab_split_preserve_spaces(text: str, max_tokens: int) -> List[str]:
    """🔧 MeCab 분할 + 공백 보존"""
    
    try:
        # 원본 텍스트에서 공백 위치 기록
        space_positions = []
        for i, char in enumerate(text):
            if char.isspace():
                space_positions.append(i)
        
        # MeCab 형태소 분석
        result = mecab.parse(text)
        morphemes = []
        position = 0
        
        for line in result.split('\n'):
            if line and line != 'EOS':
                parts = line.split('\t')
                if len(parts) >= 2:
                    surface = parts[0]
                    pos = parts[1].split(',')[0]
                    morphemes.append({
                        'surface': surface,
                        'pos': pos,
                        'start': position,
                        'end': position + len(surface)
                    })
                    position += len(surface)
        
        # 🔧 공백을 고려한 의미 단위 그룹화
        units = []
        current_unit = ""
        current_start = 0
        
        for i, morph in enumerate(morphemes):
            surface = morph['surface']
            pos = morph['pos']
            start_pos = morph['start']
            
            # 이전 형태소와 현재 형태소 사이의 공백 추가
            if current_unit and start_pos > current_start:
                spaces_between = text[current_start:start_pos]
                current_unit += spaces_between
            
            current_unit += surface
            current_start = morph['end']
            
            # 한자 + 조사 특별 처리
            is_han_particle_boundary = False
            if _is_han(surface) and i + 1 < len(morphemes):
                next_morph = morphemes[i + 1]
                if _is_particle_pos(next_morph['pos']):
                    # 다음 조사까지 포함
                    next_surface = next_morph['surface']
                    next_start = next_morph['start']
                    
                    # 공백 포함
                    if next_start > current_start:
                        spaces_between = text[current_start:next_start]
                        current_unit += spaces_between
                    
                    current_unit += next_surface
                    current_start = next_morph['end']
                    morphemes[i + 1]['surface'] = ''  # 스킵 표시
                    is_han_particle_boundary = True
            
            # 경계 조건
            is_boundary = (
                pos in ['JKS', 'JKO', 'JKC', 'JX', 'SF', 'SP'] or
                pos.startswith('E') or
                is_han_particle_boundary or
                len(current_unit.replace(' ', '')) >= max_tokens * 2 or  # 공백 제외한 길이
                _is_natural_boundary(surface, pos)
            )
            
            if is_boundary and current_unit.strip():
                units.append(current_unit.strip())
                current_unit = ""
                current_start = morph['end']
        
        if current_unit.strip():
            units.append(current_unit.strip())
        
        return [u for u in units if u.strip()]
        
    except Exception as e:
        logger.error(f"❌ MeCab 공백 보존 분할 실패: {e}")
        return _basic_split_preserve_spaces(text, max_tokens)

def _basic_split_preserve_spaces(text: str, max_tokens: int) -> List[str]:
    """🔧 기본 분할 + 공백 보존"""
    
    try:
        # 공백으로 일차 분할
        words = text.split()
        if not words:
            return [text]
        
        # 의미 단위로 재그룹화
        units = []
        current_unit = []
        current_length = 0
        
        for word in words:
            current_unit.append(word)
            current_length += len(word)
            
            # 경계 조건
            is_boundary = (
                regex.search(r'[.!?。！？]$', word) or  # 구두점으로 끝남
                _ends_with_korean_ending(word) or      # 한국어 어미로 끝남
                current_length >= max_tokens * 3 or   # 길이 제한
                len(current_unit) >= 5                 # 단어 수 제한
            )
            
            if is_boundary and current_unit:
                units.append(' '.join(current_unit))
                current_unit = []
                current_length = 0
        
        if current_unit:
            units.append(' '.join(current_unit))
        
        return [u.strip() for u in units if u.strip()]
        
    except Exception as e:
        logger.error(f"❌ 기본 공백 보존 분할 실패: {e}")
        return [text]

def _is_natural_boundary(surface: str, pos: str) -> bool:
    """자연스러운 경계인지 판단"""
    
    # 구두점
    if pos in ['SF', 'SP'] or surface in ['.', '!', '?', '。', '！', '？', ',', '，']:
        return True
    
    # 한국어 어미
    if _ends_with_korean_ending(surface):
        return True
    
    # 접속사
    connectives = ['그런데', '하지만', '따라서', '그러므로', '그리고', '또한']
    if surface in connectives:
        return True
    
    return False

def _ends_with_korean_ending(word: str) -> bool:
    """한국어 어미로 끝나는지 확인"""
    
    endings = ['다', '고', '며', '면서', '지만', '하여', '하고', '한다', '였다', '습니다']
    return any(word.endswith(ending) for ending in endings)

def _is_han(token: str) -> bool:
    """한자 포함 여부"""
    return bool(regex.search(r'\p{Han}', token))

def _is_hangul(token: str) -> bool:
    """한글 포함 여부"""
    return bool(regex.search(r'\p{Hangul}', token))

def _is_particle_pos(pos: str) -> bool:
    """품사가 조사인지 확인"""
    particle_pos = ['JKS', 'JKO', 'JKC', 'JX']
    return pos in particle_pos

def _is_particle(token: str) -> bool:
    """조사/어미 여부"""
    particles = ['은', '는', '이', '가', '을', '를', '에', '에서', '으로', '로', 
                '와', '과', '의', '도', '만', '부터', '까지', '라', '이라']
    return token in particles or (len(token) <= 2 and _is_hangul(token))

if __name__ == "__main__":
    # 공백 보존 테스트
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 공백 보존 토크나이저 테스트")
    print("=" * 50)
    
    test_cases = [
        "蒹은 물억새이고 葭는 갈대이다.",
        "白露가 얼어 서리가 된 뒤에야 歲事가 이루어지고",
        "여러 풀 가운데에 푸르게 무성했다가 白露가 얼어 서리가 되면"
    ]
    
    for text in test_cases:
        print(f"\n원본: {text}")
        units = split_tgt_meaning_units("", text, use_semantic=False)
        print(f"분할: {units}")
        
        # 공백 보존 확인
        for unit in units:
            spaces = unit.count(' ')
            print(f"  '{unit}' (공백 {spaces}개)")