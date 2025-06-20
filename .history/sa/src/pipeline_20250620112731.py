"""체인 토크나이저를 사용한 파이프라인"""
import logging
from typing import Dict, Any, Optional
from .components import BaseTokenizer, BaseEmbedder
from .text_alignment import TargetTextAligner

logger = logging.getLogger(__name__)

def process_single_row(
    row_data: Dict[str, Any],
    source_tokenizer: BaseTokenizer,  # ChainedTokenizer
    target_tokenizer: BaseTokenizer,  # ChainedTokenizer  
    embedder: BaseEmbedder,
) -> Dict[str, Any]:
    """체인 토크나이저를 사용한 행 처리"""
    
    src_text = str(row_data.get('원문', ''))
    tgt_text = str(row_data.get('번역문', ''))

    if not src_text or not tgt_text:
        return {
            'aligned_source': src_text,
            'aligned_target': tgt_text,
            'processing_info': {'error': 'Empty source or target text'}
        }
    
    try:
        # 1. 원문 체인 처리 (Prototype02 + 언어별 토크나이저)
        src_chain = getattr(source_tokenizer, 'chain', None)
        if src_chain:
            masked_src, src_masks, src_units = src_chain.process_source_text(src_text)
        else:
            # 폴백: 기본 토크나이징
            src_units = source_tokenizer.tokenize(src_text, column_name='원문')
            src_masks = []
        
        # 2. 번역문 체인 처리 (Prototype02 마스킹만)
        tgt_chain = getattr(target_tokenizer, 'chain', None)
        if tgt_chain:
            masked_tgt, tgt_masks = tgt_chain.process_target_text(tgt_text)
        else:
            # 폴백: 기본 처리
            masked_tgt = tgt_text
            tgt_masks = []
        
        if not src_units:
            src_units = [src_text]

        # 3. DP 정렬 (Prototype02 알고리즘)
        tgt_aligner = TargetTextAligner(min_tokens=1)
        tgt_units = tgt_aligner.align(src_units, masked_tgt, embedder.embed)

        # 길이 맞춤
        min_len = min(len(src_units), len(tgt_units))
        if min_len > 0:
            src_units = src_units[:min_len]
            tgt_units = tgt_units[:min_len]
        else:
            src_units = [src_text]
            tgt_units = [tgt_text]

        # 4. 마스크 복원
        if tgt_chain:
            restored_src_units = [tgt_chain.restore_masks(unit, src_masks) for unit in src_units]
            restored_tgt_units = [tgt_chain.restore_masks(unit, tgt_masks) for unit in tgt_units]
        else:
            restored_src_units = src_units
            restored_tgt_units = tgt_units

        # 5. 결과 조합
        filtered_pairs = []
        for src_unit, tgt_unit in zip(restored_src_units, restored_tgt_units):
            if src_unit.strip() and tgt_unit.strip():
                filtered_pairs.append((src_unit, tgt_unit))
        
        if not filtered_pairs:
            filtered_pairs = [(src_text, tgt_text)]
        
        final_src_parts, final_tgt_parts = zip(*filtered_pairs)
        final_source = ' | '.join(final_src_parts)
        final_target = ' | '.join(final_tgt_parts)
        
        processing_info = {
            'status': 'success',
            'algorithm': 'prototype02_chained_tokenizer',
            'source_tokenizer': getattr(source_tokenizer, 'language_type', 'prototype02'),
            'target_tokenizer': getattr(target_tokenizer, 'language_type', 'prototype02'),
            'units_count': len(filtered_pairs),
            'src_masks': len(src_masks),
            'tgt_masks': len(tgt_masks)
        }

        return {
            'aligned_source': final_source,
            'aligned_target': final_target,
            'processing_info': processing_info
        }

    except Exception as e:
        logger.error(f"행 처리 중 오류: {e}", exc_info=True)
        return {
            'aligned_source': src_text,
            'aligned_target': tgt_text,
            'processing_info': {'error': str(e)}
        }