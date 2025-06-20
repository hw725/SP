"""분석기 정보를 활용한 파이프라인"""
import logging
from typing import Dict, Any, Optional
from .components import BaseTokenizer, BaseEmbedder

logger = logging.getLogger(__name__)

def process_single_row(
    row_data: Dict[str, Any],
    source_tokenizer: BaseTokenizer,  # AnalyzerAwareTokenizer
    target_tokenizer: BaseTokenizer,  # AnalyzerAwareTokenizer  
    embedder: BaseEmbedder,
) -> Dict[str, Any]:
    """분석기 정보를 활용한 행 처리"""
    
    src_text = str(row_data.get('원문', ''))
    tgt_text = str(row_data.get('번역문', ''))

    if not src_text or not tgt_text:
        return {
            'aligned_source': src_text,
            'aligned_target': tgt_text,
            'processing_info': {'error': 'Empty source or target text'}
        }
    
    try:
        # 분석기 정보를 활용한 처리
        if hasattr(source_tokenizer, 'process_row'):
            # 분석기 지원 토크나이저 사용
            final_source, final_target, processing_info = source_tokenizer.process_row(
                src_text, tgt_text, embedder.embed
            )
        else:
            # 폴백: 기존 방식
            from .text_alignment import TextAlignmentProcessor
            processor = TextAlignmentProcessor()
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