# """Prototype02 로직으로 단순화된 파이프라인"""
import logging
import pandas as pd
from typing import Dict, Any, List, Optional
from .components import BaseTokenizer, BaseEmbedder
from .text_alignment import (
    split_src_meaning_units,
    split_tgt_meaning_units,
    TextMasker
)

logger = logging.getLogger(__name__)

def process_single_row(
    row_data: Dict[str, Any],
    source_tokenizer: BaseTokenizer,  # 호환성 위해 유지하되 실제론 안 씀
    target_tokenizer: BaseTokenizer,  # 호환성 위해 유지하되 실제론 안 씀
    embedder: BaseEmbedder,
) -> Dict[str, Any]:
    """Prototype02 로직으로 단순화된 단일 행 처리"""
    
    src_text = str(row_data.get('원문', ''))
    tgt_text = str(row_data.get('번역문', ''))
    
    processing_info = {}

    if not src_text or not tgt_text:
        processing_info['error'] = 'Empty source or target text'
        return {
            'aligned_source': src_text,
            'aligned_target': tgt_text,
            'processing_info': processing_info
        }
    
    try:
        # Prototype02 방식 그대로
        text_masker = TextMasker()
        masked_src, src_masks = text_masker.mask(src_text, text_type="source")
        masked_tgt, tgt_masks = text_masker.mask(tgt_text, text_type="target")
        
        logger.debug(f"마스킹 완료 - 원문: {len(src_masks)}개, 번역문: {len(tgt_masks)}개")

        # 원문 분할 (Prototype02 로직)
        src_units = split_src_meaning_units(masked_src)
        if not src_units:
            src_units = [masked_src]
        
        logger.debug(f"원문 분할 완료: {len(src_units)}개 단위")

        # 번역문 정렬 (Prototype02 DP 알고리즘)
        tgt_units = split_tgt_meaning_units(
            src_units,
            masked_tgt,
            embedder.embed
        )
        
        logger.debug(f"번역문 분할 완료: {len(tgt_units)}개 단위")

        # 길이 맞춤
        min_len = min(len(src_units), len(tgt_units))
        if min_len > 0:
            src_units = src_units[:min_len]
            tgt_units = tgt_units[:min_len]
        else:
            src_units = [src_text]
            tgt_units = [tgt_text]

        # 결과 조합
        aligned_source_str = ' | '.join(src_units)
        aligned_target_str = ' | '.join(tgt_units)
        
        # 마스킹 복원 (Prototype02 방식)
        final_source = text_masker.restore(aligned_source_str, src_masks)
        final_target = text_masker.restore(aligned_target_str, tgt_masks)
        
        # 복원 검증
        if '[MASK' in final_source or '[MASK' in final_target:
            logger.warning("마스킹 복원이 완전하지 않음")
            # 강제 복원 시도
            for i, original in enumerate(src_masks):
                final_source = final_source.replace(f'[MASK{i}]', original)
            for i, original in enumerate(tgt_masks):
                final_target = final_target.replace(f'[MASK{i}]', original)
        
        processing_info['status'] = 'success'
        processing_info['algorithm'] = 'prototype02_integrated'
        processing_info['units_count'] = len(src_units)

        return {
            'aligned_source': final_source,
            'aligned_target': final_target,
            'processing_info': processing_info
        }

    except Exception as e:
        logger.error(f"행 처리 중 오류: {e}", exc_info=True)
        processing_info['error'] = str(e)
        return {
            'aligned_source': src_text,
            'aligned_target': tgt_text,
            'processing_info': processing_info
        }

def process_dataframe(df: pd.DataFrame, components: Dict[str, Any], 
                     progress_callback: Optional[callable] = None) -> pd.DataFrame:
    """데이터프레임 처리"""
    
    if df.empty:
        logger.warning("빈 데이터프레임입니다")
        return df

    source_tokenizer = components.get('source_tokenizer')
    target_tokenizer = components.get('target_tokenizer')
    embedder = components.get('embedder') 

    if not all([source_tokenizer, target_tokenizer, embedder]): # 수정된 코드
        logger.error("필수 컴포넌트가 누락되었습니다: source_tokenizer, target_tokenizer, embedder") # 수정된 코드
        df['aligned_source'] = df['원문']
        df['aligned_target'] = df['번역문']
        df['processing_info'] = str({'error': 'Missing essential components'})
        return df

    results_list_of_dicts = []
    total_rows = len(df)

    for idx, row_series in df.iterrows():
        row_data_dict = row_series.to_dict()
        try:
            processed_dict = process_single_row(
                row_data_dict, 
                source_tokenizer, 
                target_tokenizer, 
                embedder, 
            )
            merged_row = row_data_dict.copy()
            merged_row.update(processed_dict)
            results_list_of_dicts.append(merged_row)
            
            if progress_callback:
                progress_callback(idx + 1, total_rows)
                
        except Exception as e:
            logger.error(f"Row {idx} 처리 실패 (in process_dataframe): {e}", exc_info=True)
            error_row = row_data_dict.copy()
            error_row['aligned_source'] = error_row.get('원문', '')
            error_row['aligned_target'] = error_row.get('번역문', '')
            error_row['processing_info'] = str({'error': str(e)})
            results_list_of_dicts.append(error_row)

    if results_list_of_dicts:
        return pd.DataFrame(results_list_of_dicts)
    else:
        return df
