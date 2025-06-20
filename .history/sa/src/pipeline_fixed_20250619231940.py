"""Prototype02 로직으로 복구된 파이프라인"""
import logging
from typing import Dict, List, Any
from .components import BaseTokenizer, BaseEmbedder
from .text_alignment_fixed import (
    split_src_meaning_units,
    split_tgt_meaning_units,
    TextMasker
)

logger = logging.getLogger(__name__)

def process_single_row(
    row_data: Dict[str, Any],
    source_tokenizer: BaseTokenizer,
    target_tokenizer: BaseTokenizer, 
    embedder: BaseEmbedder,
) -> Dict[str, Any]:
    """Prototype02 로직으로 복구된 단일 행 처리"""
    
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

        src_units = split_src_meaning_units(masked_src)
        if not src_units:
            src_units = [masked_src]

        tgt_units = split_tgt_meaning_units(
            src_units,
            masked_tgt,
            embedder.embed
        )

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
        
        processing_info['status'] = 'success'
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