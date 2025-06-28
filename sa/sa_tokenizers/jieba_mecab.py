import logging
import numpy as np
import regex
import re
from typing import List, Callable, Tuple
import jieba
import MeCab
import os

# 로그 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 기본 설정값
DEFAULT_MIN_TOKENS = 1
DEFAULT_MAX_TOKENS = 50
DEFAULT_SIMILARITY_THRESHOLD = 0.7

# 전역 변수 선언
mecab = None
mecab_initialized = False

def initialize_mecab() -> MeCab.Tagger:
    """MeCab 실제 초기화 로직"""
    global mecab
    try:
        logger.debug("⚙️ MeCab 초기화 시도 중...")
        dicdir = "C:/Users/junto/miniconda3/envs/myenv/Lib/site-packages/mecab_ko_dic/dicdir"
        required_files = [
            f"{dicdir}/mecabrc",
            f"{dicdir}/sys.dic",
            f"{dicdir}/matrix.bin",
            f"{dicdir}/user.dic",
        ]
        missing = [p for p in required_files if not os.path.exists(p)]
        if missing:
            logger.warning(f"⚠️ MeCab 필수 파일 누락: {len(missing)}개")
            for p in missing:
                logger.warning(f"  - {p}")
            return None
        # 경로 정규화
        mecabrc_path = required_files[0].replace('\\', '/')
        dicdir_path = dicdir.replace('\\', '/')
        user_dic_path = required_files[3].replace('\\', '/')
        # 사용자 사전 생성
        if os.path.getsize(user_dic_path) == 0:
            with open(user_dic_path, 'w', encoding='utf-8') as f:
                f.write("# 사용자 사전\n")
        # 초기화
        mecab = MeCab.Tagger(f'-r "{mecabrc_path}" -d "{dicdir_path}" -u "{user_dic_path}"')
        # 테스트: 개별 토큰 존재 여부로 확인
        test = mecab.parse("초기화 테스트")
        # MeCab이 형태소 단위로 분리하므로 각 surface를 추출
        tokens = [line.split('\t')[0] for line in test.split('\n') if line and line != 'EOS']
        if {'초기', '화', '테스트'}.issubset(tokens):
            logger.info("✅ MeCab 정상 초기화")
            logger.debug(f"테스트 토큰: {tokens[:5]}")
            return mecab
        else:
            logger.error(f"❌ MeCab 테스트 실패, 토큰 누락: {tokens}")
            return None
    except Exception:
        logger.exception("🔥 MeCab 초기화 중 심각한 오류")
        try:
            logger.debug(f"디렉토리 내용: {os.listdir(dicdir)}")
        except Exception:
            pass
        return None

def ensure_mecab_initialized() -> MeCab.Tagger:
    """안전한 MeCab 초기화 보장"""
    global mecab_initialized, mecab
    if not mecab_initialized:
        mecab = initialize_mecab()
        mecab_initialized = True
    return mecab

# 모듈 로드 시 자동 초기화
ensure_mecab_initialized()

# 미리 컴파일된 정규식
hanja_re = regex.compile(r'\p{Han}+')
hangul_re = regex.compile(r'^\p{Hangul}+$')

def split_src_meaning_units(
    text: str,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    use_advanced: bool = True
) -> List[str]:
    """한문 텍스트를 의미 단위로 분할 - jieba 분석 참고"""
    words = text.replace('\n', ' ').replace('：', '： ').split()
    if not words:
        return []
    jieba_tokens = list(jieba.cut(text))
    units = []
    i = 0
    while i < len(words):
        word = words[i]
        if hanja_re.search(word):
            units.append(word)
            i += 1
            continue
        if hangul_re.match(word):
            group = [word]
            j = i + 1
            while j < len(words) and hangul_re.match(words[j]):
                if _should_group_words_by_jieba(group + [words[j]], jieba_tokens):
                    group.append(words[j])
                    j += 1
                else:
                    break
            units.append(' '.join(group))
            i = j
            continue
        units.append(word)
        i += 1
    return units

def _should_group_words_by_jieba(word_group: List[str], jieba_tokens: List[str]) -> bool:
    """jieba 분석 결과를 참고해서 어절들을 묶을지 결정"""
    combined = ''.join(word_group)
    for token in jieba_tokens:
        if token.replace(' ', '') == combined.replace(' ', ''):
            return True
    if len(combined) > 10:
        return False
    return len(word_group) <= 3

def split_inside_chunk(chunk: str) -> List[str]:
    """번역문 청크를 의미 단위로 분할 - MeCab 분석 참고"""
    m = ensure_mecab_initialized()
    if not chunk or not chunk.strip():
        return []
    words = chunk.split()
    morpheme_info = []
    if m:
        result = m.parse(chunk)
        for line in result.split('\n'):
            if line and line != 'EOS':
                surface, pos = line.split('\t')[0], line.split('\t')[1].split(',')[0]
                morpheme_info.append((surface, pos))
    delimiters = ['을', '를', '이', '가', '은', '는', '에', '에서', '로', '으로',
                  '와', '과', '고', '며', '하고', '때', '의', '도', '만', '때에', '：']
    units = []
    current_group = []
    for word in words:
        current_group.append(word)
        should_break = any(word.endswith(delimiter) for delimiter in delimiters)
        if morpheme_info and not should_break:
            should_break = _should_break_by_mecab(word, morpheme_info)
        if should_break:
            units.append(' '.join(current_group))
            current_group = []
    if current_group:
        units.append(' '.join(current_group))
    return [unit.strip() for unit in units if unit.strip()]

def _should_break_by_mecab(word: str, morpheme_info: List[tuple]) -> bool:
    """MeCab 품사를 참고한 경계 결정"""
    for surface, pos in morpheme_info:
        if surface in word:
            if pos in ['JKS', 'JKO', 'JKC', 'JX', 'EF', 'EC', 'ETN', 'SF', 'SP']:
                return True
            if pos in ['VV', 'VA', 'VX']:
                return True
    return False

def find_target_span_end_simple(src_unit: str, remaining_tgt: str) -> int:
    """간단한 타겟 스팬 탐색"""
    hanja_chars = regex.findall(r'\p{Han}+', src_unit)
    if not hanja_chars:
        return 0
    last = hanja_chars[-1]
    idx = remaining_tgt.rfind(last)
    if idx == -1:
        return len(remaining_tgt)
    end = idx + len(last)
    next_space = remaining_tgt.find(' ', end)
    return next_space + 1 if next_space != -1 else len(remaining_tgt)

def find_target_span_end_semantic(
    src_unit: str,
    remaining_tgt: str,
    embed_func: Callable,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
) -> int:
    """최적화된 타겟 스팬 탐색"""
    if not src_unit or not remaining_tgt:
        return 0
    try:
        src_emb = embed_func([src_unit])[0]
        tgt_tokens = remaining_tgt.split()
        if not tgt_tokens:
            return 0
        upper = min(len(tgt_tokens), max_tokens)
        cumulative_lengths = [0]
        current_length = 0
        for tok in tgt_tokens:
            current_length += len(tok) + 1
            cumulative_lengths.append(current_length)
        candidates, candidate_indices = [], []
        step_size = 1 if upper <= 10 else 2
        for end_i in range(min_tokens-1, upper, step_size):
            candidates.append(" ".join(tgt_tokens[:end_i+1]))
            candidate_indices.append(end_i)
        cand_embs = embed_func(candidates)
        best_score, best_end_idx = -1.0, cumulative_lengths[-1]
        for i, emb in enumerate(cand_embs):
            score = np.dot(src_emb, emb) / (np.linalg.norm(src_emb)*np.linalg.norm(emb) + 1e-8)
            end_i = candidate_indices[i]
            length_ratio = (end_i+1)/len(tgt_tokens)
            length_penalty = min(1.0, length_ratio*2)
            adjusted = score * length_penalty
            if adjusted > best_score and score >= similarity_threshold:
                best_score, best_end_idx = adjusted, cumulative_lengths[end_i+1]
        return best_end_idx
    except Exception as e:
        logger.warning(f"의미 매칭 오류, 단순 매칭으로 대체: {e}")
        return find_target_span_end_simple(src_unit, remaining_tgt)

def split_tgt_by_src_units(src_units: List[str], tgt_text: str) -> List[str]:
    """단순 원문-번역 연결 방식"""
    results, cursor, total = [], 0, len(tgt_text)
    for src_u in src_units:
        remaining = tgt_text[cursor:]
        end_len = find_target_span_end_simple(src_u, remaining)
        chunk = tgt_text[cursor:cursor+end_len]
        results.extend(split_inside_chunk(chunk))
        cursor += end_len
    if cursor < total:
        results.extend(split_inside_chunk(tgt_text[cursor:]))
    return results

def split_tgt_by_src_units_semantic(
    src_units: List[str], tgt_text: str, embed_func: Callable, min_tokens: int = DEFAULT_MIN_TOKENS
) -> List[str]:
    """의미 기반 DP 분할"""
    tgt_tokens, N, T = tgt_text.split(), len(src_units), len(tgt_text.split())
    if N==0 or T==0: return []
    dp = np.full((N+1,T+1), -np.inf); back = np.zeros((N+1,T+1),dtype=int); dp[0,0]=0.0
    src_embs = embed_func(src_units)
    span_map, all_spans = {}, []
    for i in range(1,N+1):
        for j in range(i*min_tokens, T-(N-i)*min_tokens+1):
            for k in range((i-1)*min_tokens, j-min_tokens+1):
                span = " ".join(tgt_tokens[k:j]).strip()
                if span and (k,j) not in span_map:
                    span_map[(k,j)] = span; all_spans.append(span)
    all_spans = list(set(all_spans))
    def batch_embed(spans,bs=100):
        out=[]
        for s in range(0,len(spans),bs): out.extend(embed_func(spans[s:s+bs]))
        return out
    span_embs = batch_embed(all_spans)
    emb_dict = {sp:em for sp,em in zip(all_spans,span_embs)}
    for i in range(1,N+1):
        for j in range(i*min_tokens, T-(N-i)*min_tokens+1):
            for k in range((i-1)*min_tokens, j-min_tokens+1):
                span = span_map[(k,j)]; sim = float(np.dot(src_embs[i-1],emb_dict[span])/(np.linalg.norm(src_embs[i-1])*np.linalg.norm(emb_dict[span])+1e-8))
                score = dp[i-1,k]+sim
                if score>dp[i,j]: dp[i,j]=score; back[i,j]=k
    cuts, curr = [T], T
    for i in range(N,0,-1): prev=back[i,curr]; cuts.append(prev); curr=prev
    cuts=cuts[::-1]
    assert cuts[0]==0 and cuts[-1]==T and len(cuts)==N+1
    return [" ".join(tgt_tokens[cuts[i]:cuts[i+1]]).strip() for i in range(N)]

def split_tgt_meaning_units(
    src_text: str,
    tgt_text: str,
    use_semantic: bool = True,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    embed_func: Callable = None
) -> List[str]:
    """통합 분할 인터페이스"""
    if embed_func is None:
        from sa_embedders import compute_embeddings_with_cache
        embed_func = compute_embeddings_with_cache
    src_units = split_src_meaning_units(src_text,min_tokens,max_tokens)
    return (split_tgt_by_src_units_semantic(src_units,tgt_text,embed_func,min_tokens)
            if use_semantic else split_tgt_by_src_units(src_units,tgt_text))

def tokenize_text(text: str) -> List[str]:
    """형태소 분석 및 토큰화"""
    m = ensure_mecab_initialized()
    if m:
        return [line.split('\t')[0] for line in m.parse(text).split('\n') if line and line!='EOS']
    return text.split()

def pos_tag_text(text: str) -> List[Tuple[str, str]]:
    """품사 태깅"""
    m = ensure_mecab_initialized()
    if m:
        return [(line.split('\t')[0], line.split('\t')[1].split(',')[0])
                for line in m.parse(text).split('\n') if line and line!='EOS']
    return [(w,'UNKNOWN') for w in text.split()]

def sentence_split(text: str) -> List[str]:
    """문장 단위 분리"""
    return [s.strip() for s in re.split(r'[.!?。！？]+', text) if s.strip()]
