"""Prototype02 io_manager.py의 처리 로직을 완전 통합한 파이프라인"""
import logging
from typing import Dict, List, Any, Optional
from .components import BaseTokenizer, BaseEmbedder
from .text_alignment import (
    TextMasker,
    SourceTextSplitter, 
    TargetTextAligner,
    TextAligner
)

logger = logging.getLogger(__name__)

def process_single_row(
    row_data: Dict[str, Any],
    source_tokenizer: Optional[BaseTokenizer],
    target_tokenizer: Optional[BaseTokenizer], 
    embedder: BaseEmbedder,
) -> Dict[str, Any]:
    """Prototype02 _process_single_row 로직 완전 통합"""
    
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
        # *** Prototype02 처리 순서 그대로 ***
        
        # 1. 괄호 마스킹 (Prototype02 그대로)
        text_masker = TextMasker()
        masked_src, src_masks = text_masker.mask(src_text, text_type="source")
        masked_tgt, tgt_masks = text_masker.mask(tgt_text, text_type="target")

        # 2. 원문 의미 단위 분할 (Prototype02 그대로)
        src_splitter = SourceTextSplitter()
        src_units = src_splitter.split(masked_src)

        # 3. 번역문 의미 단위 분할 (Prototype02 그대로)
        tgt_aligner = TargetTextAligner(min_tokens=1)
        tgt_units = tgt_aligner.align(src_units, masked_tgt, embedder.embed)

        # 분할 결과 검증
        if not src_units or not tgt_units:
            logger.warning("의미 단위 분할 결과가 비어 있습니다.")
            return {
                'aligned_source': src_text,
                'aligned_target': tgt_text,
                'processing_info': {'error': 'Empty units after splitting'}
            }

        # 4. 마스크 복원 (Prototype02 그대로)
        restored_src_units = [text_masker.restore(unit, src_masks) for unit in src_units]
        restored_tgt_units = [text_masker.restore(unit, tgt_masks) for unit in tgt_units]

        # 5. 정렬 수행 (Prototype02 그대로)
        aligner = TextAligner()
        aligned_pairs = aligner.align(restored_src_units, restored_tgt_units, embedder.embed)

        if not aligned_pairs:
            return {
                'aligned_source': src_text,
                'aligned_target': tgt_text,
                'processing_info': {'error': 'Empty alignment result'}
            }

        # 6. 결과 조합 (Prototype02 출력 형식에 맞춤)
        aligned_src_units, aligned_tgt_units = zip(*aligned_pairs)
        
        # 빈 결과 필터링
        filtered_pairs = []
        for src_gu, tgt_gu in zip(aligned_src_units, aligned_tgt_units):
            if src_gu.strip() and tgt_gu.strip():
                filtered_pairs.append((src_gu, tgt_gu))
        
        if not filtered_pairs:
            filtered_pairs = [(src_text, tgt_text)]
        
        # 결과 조합 - ' | '로 연결
        final_src_parts, final_tgt_parts = zip(*filtered_pairs)
        final_source = ' | '.join(final_src_parts)
        final_target = ' | '.join(final_tgt_parts)
        
        processing_info['status'] = 'success'
        processing_info['algorithm'] = 'prototype02_complete'
        processing_info['units_count'] = len(filtered_pairs)
        processing_info['src_masks'] = len(src_masks)
        processing_info['tgt_masks'] = len(tgt_masks)

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
