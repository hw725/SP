"""구조 분석 관련 함수들"""
import logging
import numpy as np
from typing import Dict, List, Callable, Optional

logger = logging.getLogger(__name__)

def analyze_unit_structure(unit: str, source_analyzer=None, target_analyzer=None, text_type: str = "source") -> Dict:
    """단위 텍스트의 구조 분석"""
    analysis = {
        "original": unit,
        "chinese_parts": [],
        "korean_parts": [],
        "structure_info": None
    }
    
    chinese_chars = []
    korean_chars = []
    
    for char in unit:
        if '\u4e00' <= char <= '\u9fff':
            chinese_chars.append(char)
        elif '\uac00' <= char <= '\ud7af':
            korean_chars.append(char)
    
    analysis["chinese_parts"] = chinese_chars
    analysis["korean_parts"] = korean_chars
    
    analyzer = target_analyzer if text_type == "target" else source_analyzer
    if analyzer and hasattr(analyzer, 'analyze_structure'):
        try:
            analysis["structure_info"] = analyzer.analyze_structure(unit)
        except Exception as e:
            logger.warning(f"구조 분석 실패 ({text_type}, unit: {unit[:30]}...): {e}")
    
    return analysis

def analyze_merged_eojeols(
    eojeols: List[str],
    source_analyzer=None,
    target_analyzer=None,
    text_type: str = "target"
) -> Dict:
    """병합된 어절들의 구조 분석"""
    merged_text = " ".join(eojeols)
    analysis = {
        "original": merged_text,
        "eojeols": eojeols,
        "chinese_parts": [],
        "korean_parts": [],
        "structure_info_list": [] # 각 어절별 분석 결과
        # "merged_structure_info": None # 병합된 전체 텍스트에 대한 분석 결과
    }
    
    all_chinese_parts = []
    all_korean_parts = []
    
    for eojeol in eojeols:
        eojeol_analysis = analyze_unit_structure(eojeol, source_analyzer, target_analyzer, text_type)
        all_chinese_parts.extend(eojeol_analysis.get("chinese_parts", []))
        all_korean_parts.extend(eojeol_analysis.get("korean_parts", []))
        if eojeol_analysis.get("structure_info"):
            analysis["structure_info_list"].append(eojeol_analysis["structure_info"])

    analysis["chinese_parts"] = list(set(all_chinese_parts)) # 중복 제거
    analysis["korean_parts"] = list(set(all_korean_parts)) # 중복 제거
    
    # 병합된 전체 텍스트에 대한 분석 (선택적)
    current_analyzer = target_analyzer if text_type == "target" else source_analyzer
    if current_analyzer and hasattr(current_analyzer, 'analyze_structure'):
        try:
            merged_structure_info = current_analyzer.analyze_structure(merged_text)
            analysis["merged_structure_info"] = merged_structure_info
        except Exception as e:
            logger.warning(f"병합된 텍스트 구조 분석 실패 ({text_type}, merged_text: {merged_text[:30]}...): {e}")
            
    return analysis


def embed_with_structure_analysis(text: str, analysis: dict, embed_func: Callable) -> Optional[np.ndarray]:
    """구조 분석 정보를 활용한 임베딩 계산"""
    if not text.strip():
        return None

    try:
        base_embedding_array_2d = embed_func([text]) # embed_func는 List[str]을 받고 np.ndarray (N x Dim) 반환
    except Exception as e:
        logger.error(f"embed_func 호출 중 예외 발생 (text: '{text[:100]}...'): {e}")
        return None

    if base_embedding_array_2d is None or base_embedding_array_2d.size == 0:
        logger.warning(f"embed_func가 None 또는 빈 배열을 반환 (text: '{text[:100]}...')")
        return None
    
    base_embedding_1d = base_embedding_array_2d[0] # 2D (1 x Dim) -> 1D (Dim,)

    final_embedding = base_embedding_1d

    chinese_parts = analysis.get("chinese_parts", [])
    korean_parts = analysis.get("korean_parts", [])

    if chinese_parts or korean_parts:
        structure_hints = []
        if chinese_parts: structure_hints.append("한자성분: " + " ".join(list(set(chinese_parts))[:5]))
        if korean_parts: structure_hints.append("한글성분: " + " ".join(list(set(korean_parts))[:5]))
        
        if structure_hints:
            context_text = text + " [힌트: " + "; ".join(structure_hints) + "]"
            try:
                context_embedding_array_2d = embed_func([context_text])
            except Exception as e:
                logger.error(f"embed_func 호출 중 예외 발생 (context_text: '{context_text[:100]}...'): {e}")
                context_embedding_array_2d = None

            if context_embedding_array_2d is not None and context_embedding_array_2d.size > 0:
                context_embedding_1d = context_embedding_array_2d[0]
                if base_embedding_1d.shape == context_embedding_1d.shape: # 차원 일치 확인
                    final_embedding = 0.8 * base_embedding_1d + 0.2 * context_embedding_1d
                else:
                    logger.warning(f"기본 임베딩과 컨텍스트 임베딩 차원 불일치. 기본 임베딩 사용. Text: {text[:30]}")
    return final_embedding

def calculate_structure_similarity(src_analysis: dict, tgt_analysis: dict) -> float:
    """구조 분석 정보를 바탕으로 원문과 번역문 간의 구조적 유사도 계산"""
    if not src_analysis or not tgt_analysis:
        return 0.0
    
    similarity_score = 0.0
    total_weights = 0.0
    
    src_chinese = set(src_analysis.get("chinese_parts", []))
    tgt_chinese = set(tgt_analysis.get("chinese_parts", [])) # tgt_analysis는 merged_eojeols의 분석 결과일 수 있음
    if src_chinese or tgt_chinese:
        chinese_sim = len(src_chinese & tgt_chinese) / max(len(src_chinese | tgt_chinese), 1)
        similarity_score += chinese_sim * 0.4
        total_weights += 0.4
    
    src_korean = set(src_analysis.get("korean_parts", []))
    tgt_korean = set(tgt_analysis.get("korean_parts", []))
    if src_korean or tgt_korean:
        korean_sim = len(src_korean & tgt_korean) / max(len(src_korean | tgt_korean), 1)
        similarity_score += korean_sim * 0.3
        total_weights += 0.3
    
    # src_structure는 단일 유닛 분석, tgt_structure는 병합된 어절 전체 또는 각 어절 리스트일 수 있음
    # 현재는 src_analysis.get("structure_info")와 tgt_analysis.get("merged_structure_info") 또는
    # tgt_analysis.get("structure_info_list")를 비교해야 함.
    # 단순화를 위해 현재 로직(기본 점수) 유지 또는 이 부분 유사도 계산 제거.
    # src_structure_info = src_analysis.get("structure_info")
    # tgt_merged_structure_info = tgt_analysis.get("merged_structure_info") # 병합된 어절 전체에 대한 분석 결과
    # if src_structure_info and tgt_merged_structure_info:
    #     # TODO: structure_info 간의 실제 유사도 계산 로직 필요
    #     structure_sim = 0.5  # 임시 기본값
    #     similarity_score += structure_sim * 0.3
    #     total_weights += 0.3
    
    return similarity_score / max(total_weights, 1.0) if total_weights > 0 else 0.0