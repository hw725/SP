"""mecab, jieba 기반 발화자/발화동사 탐지 유틸리티 (pa용)"""
import re
try:
    import MeCab
    mecab = MeCab.Tagger()
except Exception:
    mecab = None
try:
    import jieba
except Exception:
    jieba = None

def is_korean(text):
    return bool(re.search(r'[가-힣]', text))

def is_chinese(text):
    return bool(re.search(r'[一-鿌]', text))

# 발화동사/부사 리스트(확장 가능)
SPEECH_VERBS = ['말하', '대답하', '묻', '답하', '덧붙이', '이어', '계속하', '전하', '알리', '외치', '부르', '명하']
SPEECH_ADVERBS = ['또', '다시', '이어', '계속하여', '계속해', '계속', '그리고', '그러고', '이어서']

# mecab 기반: 주어(JKS) + 발화동사(VV) 또는 부사+발화동사

def has_speaker_and_speech_verb_ko(text):
    if not mecab:
        return False
    parsed = mecab.parse(text)
    # JKS(주격조사) + 발화동사
    for line in parsed.split('\n'):
        if '\t' not in line:
            continue
        surface, feats = line.split('\t', 1)
        if 'JKS' in feats:
            # 이후에 발화동사(VV) 등장 여부
            if any(verb in parsed for verb in SPEECH_VERBS):
                return True
    # 부사+발화동사 패턴
    for adv in SPEECH_ADVERBS:
        for verb in SPEECH_VERBS:
            if adv in text and verb in text:
                return True
    return False

def has_speech_verb_ko(text):
    # 주어 없이 부사+발화동사 또는 발화동사 단독
    for verb in SPEECH_VERBS:
        if verb in text:
            return True
    return False

def has_speech_verb_zh(text):
    if not jieba:
        return False
    tokens = list(jieba.cut(text))
    # 중국어 발화동사 패턴(확장 가능)
    for verb in ['曰', '云', '言', '對', '答', '謂', '問', '報', '命', '呼']:
        if verb in tokens or verb in text:
            return True
    return False
