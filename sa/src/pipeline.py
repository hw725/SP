# src/pipeline.py
import logging
import pandas as pd
import numpy as np  # ✅ 추가
from typing import Dict, Any, List, Optional
import re  # ✅ 정규 표현식 모듈 추가

from .components import BaseTokenizer, BaseEmbedder, BaseAligner
from .utils import (
    mask_brackets, restore_masks, 
    split_src_meaning_units, split_tgt_meaning_units  # ✅ 복원된 함수 호출 확인
)

logger = logging.getLogger(__name__)

def _create_embed_func(embedder: BaseEmbedder):
    """임베딩 함수 생성 (멀티프로세싱 호환)"""
    def embed_func(texts: List[str]):
        if not texts: # 빈 리스트 처리
            return np.array([])  # ✅ numpy 사용 가능
        return embedder.embed(texts)
    return embed_func

def process_single_row(
    row_data: Dict[str, Any],
    source_tokenizer: BaseTokenizer,
    target_tokenizer: BaseTokenizer,
    embedder: BaseEmbedder,
    aligner: BaseAligner,
) -> Dict[str, Any]:
    """단일 행 처리 - 통합된 단일 함수"""
    
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
        # 1. 마스킹
        masked_src, src_masks = mask_brackets(src_text, text_type="source")
        masked_tgt, tgt_masks = mask_brackets(tgt_text, text_type="target")
        processing_info['masking'] = 'success'

        # 2. embed_func 생성
        embed_func = _create_embed_func(embedder)

        # 3. 원문 의미 단위 분할 (복원된 한문 분할 로직 사용)
        src_units = split_src_meaning_units(masked_src)  # ✅ 복원된 함수 호출
        if not src_units:
            processing_info['error'] = 'Source text could not be split into units'
            processing_info['src_units_count'] = 0
            return {
                'aligned_source': src_text,
                'aligned_target': tgt_text,
                'processing_info': processing_info
            }
        processing_info['src_units_count'] = len(src_units)
        
        # 4. 번역문 분할
        tgt_units = split_tgt_meaning_units(  # ✅ 복원된 함수 호출
            masked_tgt, 
            src_units=src_units, 
            embed_func=embed_func, 
            source_analyzer=source_tokenizer if hasattr(source_tokenizer, 'analyze_structure') else None,
            target_analyzer=target_tokenizer if hasattr(target_tokenizer, 'analyze_structure') else None
        )
        if not tgt_units:
            processing_info['error'] = 'Target text could not be split into units aligned with source'
            processing_info['tgt_units_count'] = 0
            return {
                'aligned_source': src_text,
                'aligned_target': tgt_text,
                'processing_info': processing_info
            }
        processing_info['tgt_units_count'] = len(tgt_units)

        logger.debug(f"Source units ({len(src_units)}): {src_units}")
        logger.debug(f"Target units ({len(tgt_units)}): {tgt_units}")

        # 5. 정렬
        aligned_pairs = aligner.align(src_units, tgt_units, embed_func)
        processing_info['aligned_pairs_count_raw'] = len(aligned_pairs)

        final_aligned_pairs = aligned_pairs

        # 6. 최종 결과 - 구조 보존하면서 조합
        aligned_source_parts = [src for src, tgt in final_aligned_pairs]
        aligned_target_parts = [tgt for src, tgt in final_aligned_pairs]

        # ✅ 빈 문자열도 유지하여 구조 보존, 구분자 사용
        aligned_source_str = ' | '.join(aligned_source_parts)
        aligned_target_str = ' | '.join(aligned_target_parts)
        
        # 연속된 빈 구분자 정리
        aligned_source_str = re.sub(r'\s*\|\s*\|\s*', ' | ', aligned_source_str)
        aligned_target_str = re.sub(r'\s*\|\s*\|\s*', ' | ', aligned_target_str)
        
        # 앞뒤 구분자 제거
        aligned_source_str = aligned_source_str.strip(' |')
        aligned_target_str = aligned_target_str.strip(' |')
        
        final_source = restore_masks(aligned_source_str, src_masks)
        final_target = restore_masks(aligned_target_str, tgt_masks)
        
        processing_info['status'] = 'success'
        processing_info['final_aligned_pairs_count'] = len(final_aligned_pairs)

        return {
            'aligned_source': final_source,
            'aligned_target': final_target,
            'processing_info': processing_info
        }

    except Exception as e:
        logger.error(f"Error processing row: {row_data.get('id', 'N/A')} - {e}", exc_info=True)
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
    aligner = components.get('aligner')

    if not all([source_tokenizer, target_tokenizer, embedder, aligner]):
        logger.error("필수 컴포넌트가 누락되었습니다: source_tokenizer, target_tokenizer, embedder, aligner")
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
                aligner
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
