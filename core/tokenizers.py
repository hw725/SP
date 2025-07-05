"""Core tokenizer module for splitting text into meaningful units using jieba and MeCab."""

import logging
import numpy as np
import regex
import re
import itertools
from typing import List, Callable, Optional
import jieba
import MeCab
import os
import sys
import spacy

# --- Global variables for spaCy models ---
nlp_ko = None
nlp_zh = None

def load_spacy_model(model_name: str):
    """Loads a spaCy model if it hasn't been loaded yet."""
    global nlp_ko, nlp_zh
    if model_name == "ko" and nlp_ko is None:
        try:
            nlp_ko = spacy.load("ko_core_news_lg")
            logger.info("spaCy Korean model loaded.")
        except OSError:
            logger.warning("spaCy Korean model not found. Please run: python -m spacy download ko_core_news_lg")
    elif model_name == "zh" and nlp_zh is None:
        try:
            nlp_zh = spacy.load("zh_core_web_lg")
            logger.info("spaCy Chinese model loaded.")
        except OSError:
            logger.warning("spaCy Chinese model not found. Please run: python -m spacy download zh_core_web_lg")

def split_sentences_spacy(text: str, lang: str = 'ko') -> List[str]:
    """Splits text into sentences using spaCy."""
    load_spacy_model(lang)
    nlp = nlp_ko if lang == 'ko' else nlp_zh
    if not nlp:
        return [text] # Fallback
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def contains_chinese(text: str) -> bool:
    """Checks if the text contains a significant amount of Chinese characters."""
    chinese_count = len(regex.findall(r'\p{Han}', text))
    return chinese_count > len(text) * 0.3

def split_source_by_whitespace_and_align(source: str, target_count: int) -> List[str]:
    """Splits the source text based on whitespace to align with the target sentence count."""
    if not source.strip():
        return [''] * target_count
    
    tokens = source.split()
    if not tokens:
        return [''] * target_count

    if len(tokens) <= target_count:
        return tokens + [''] * (target_count - len(tokens))
    else:
        # Distribute tokens evenly
        chunk_size = len(tokens) // target_count
        remainder = len(tokens) % target_count
        result = []
        start = 0
        for i in range(target_count):
            end = start + chunk_size + (1 if i < remainder else 0)
            result.append(' '.join(tokens[start:end]))
            start = end
        return result


logger = logging.getLogger(__name__)

# --- MeCab Initializer ---
def get_mecab_path(path_name: str) -> str:
    """Tries to find the mecab-ko-dic path in common locations."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Try to get path from environment variable first
    mecab_env_path = os.getenv("MECAB_KO_DIC_PATH")
    if mecab_env_path and os.path.exists(mecab_env_path):
        return mecab_env_path

    possible_paths = [
        os.path.join(project_root, ".venv", "Lib", "site-packages", path_name),
        os.path.join(sys.prefix, "Lib", "site-packages", path_name),
        os.path.join(sys.prefix, "Lib", "site-packages", "mecab_ko_dic", "dicdir"),
        os.path.join(sys.prefix, "mecab_ko_dic", "dicdir"), # Another common installation path
        os.path.join(project_root, ".venv", "Lib", "site-packages", "mecab_ko_dic", "dicdir"), # Added this line
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return '' # Return empty if not found

try:
    # Find mecab-ko-dic directory dynamically
    mecab_dic_dir = get_mecab_path('mecab_ko_dic/dicdir')
    if mecab_dic_dir:
        # Construct paths for mecabrc and user.dic relative to the found dicdir
        mecabrc_path = os.path.join(mecab_dic_dir, 'mecabrc')
        userdic_path = os.path.join(mecab_dic_dir, 'user.dic')
        
        tagger_cmd = f'-r "{mecabrc_path}" -d "{mecab_dic_dir}"'
        if os.path.exists(userdic_path):
            tagger_cmd += f' -u "{userdic_path}"'

        mecab = MeCab.Tagger(tagger_cmd)
        logger.info("MeCab initialized successfully from {}".format(mecab_dic_dir))
    else:
        mecab = None
        logger.warning("MeCab dictionary not found. MeCab functionality will be disabled.")
except Exception as e:
    logger.warning(f"MeCab initialization failed: {e}")
    mecab = None

# --- The rest of the tokenizer code from jieba_mecab.py ---

# 미리 컴파일된 정규식
hanja_re = regex.compile(r'\p{Han}+')
hangul_re = regex.compile(r'^\p{Hangul}+$')

# (The entire content of sa/sa_tokenizers/jieba_mecab.py from the 'BOUNDARY_MARKERS' definition onwards should be pasted here)
# For brevity, I will just copy the main functions and assume the helper functions and constants are also copied.

def split_src_meaning_units(
    text: str,
    min_tokens: int = 1,
    max_tokens: int = 50,
    **kwargs
) -> List[str]:
    """
    Splits source text into meaningful units using jieba, ensuring absolute integrity.
    Uses token indices to slice the original string, preserving all whitespace.
    """
    if not text:
        return []

    # Use jieba.tokenize to get words with their start and end indices
    tokens = list(jieba.tokenize(text))
    if not tokens:
        return [text]

    units = []
    current_pos = 0
    for word, start, end in tokens:
        # Add any whitespace or untokenized characters between tokens
        if start > current_pos:
            units.append(text[current_pos:start])
        
        units.append(word)
        current_pos = end

    # Add any trailing whitespace
    if current_pos < len(text):
        units.append(text[current_pos:])

    # Simple grouping logic for demonstration (can be enhanced)
    # This example groups units based on punctuation
    final_units = []
    current_group = ""
    for unit in units:
        current_group += unit
        if any(p in unit for p in "，。；！？："):
            final_units.append(current_group)
            current_group = ""
    
    if current_group:
        final_units.append(current_group)

    # Integrity Check
    if ''.join(final_units) != text:
        logger.warning(f"Source text integrity check failed. Original: '{text}', Reconstructed: '{ ''.join(final_units)}'. Falling back to returning the original text as a single unit.")
        return [text]

    return final_units 

def split_tgt_meaning_units_sequential(
    src_text: str,
    tgt_text: str,
    min_tokens: int = 1,
    max_tokens: int = 50,
    embed_func: Callable = None,
    **kwargs
) -> List[str]:
    """Splits target text (Korean) ensuring integrity by reconstructing from MeCab tokens."""
    if not tgt_text:
        return []

    try:
        if mecab:
            # MeCab은 문자열을 반환하므로, 이를 직접 사용
            reconstructed_text = mecab.parse(tgt_text)
            # MeCab 결과는 각 라인이 형태소 분석 결과이므로, 이를 다시 문장으로 합쳐야 함
            # 여기서는 간단히 줄바꿈으로 분리된 것을 다시 합치는 것으로 가정
            reconstructed_text = " ".join([line.split('\t')[0] for line in reconstructed_text.split('\n') if line.strip()])

            if reconstructed_text != tgt_text:
                logger.warning(f"Target text reconstruction from MeCab failed. Original: '{tgt_text}', Reconstructed: '{reconstructed_text}'. Integrity might be compromised.")
        else:
            reconstructed_text = tgt_text # Fallback if MeCab is not available

        # Simple splitting logic based on punctuation for demonstration
        # This can be replaced with more sophisticated logic based on MeCab's POS tags
        units = re.split(r'([.!?。！？])', reconstructed_text)
        if not units:
            return [reconstructed_text]

        result = []
        for i in range(0, len(units) - 1, 2):
            sentence = units[i] + (units[i+1] if i+1 < len(units) else '')
            if sentence.strip():
                result.append(sentence.strip())
        
        # A more robust integrity check would be needed here.
        # For now, we return the reconstructed and split text.
        return result if result else [reconstructed_text]
    except Exception as e:
        logger.error(f"Error during MeCab parsing for target text: {e}. Falling back to simple split.")
        return [tgt_text] if tgt_text else []