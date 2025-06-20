"""통합된 TextAlignmentProcessor 사용 - 완전히 새로 작성"""
import logging
from typing import Dict, Any, Optional
from .components import BaseTokenizer, BaseEmbedder
from .text_alignment import TextAlignmentProcessor

logger = logging.getLogger(__name__)

def process_single_row(
    row_data: Dict[str, Any],
    source_tokenizer: Optional[BaseTokenizer],
    target_tokenizer: Optional[BaseTokenizer], 
    embedder: BaseEmbedder,
) -> Dict[str, Any]:
    """TextAlignmentProcessor를 사용한 단일 행 처리"""
    
    src_text = str(row_data.get('원문', ''))
    tgt_text = str(row_data.get('번역문', ''))

    if not src_text or not tgt_text:
        return {
            'aligned_source': src_text,
            'aligned_target': tgt_text,
            'processing_info': {'error': 'Empty source or target text'}
        }
    
    try:
        # TextAlignmentProcessor 사용 (Prototype02 완전 통합)
        processor = TextAlignmentProcessor(min_tokens=1)
        final_source, final_target, processing_info = processor.process(
            src_text, tgt_text, embedder.embed
        )

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