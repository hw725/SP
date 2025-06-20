"""파이프라인 처리 로직"""
import logging
from typing import Dict, Any, Callable

logger = logging.getLogger(__name__)

def process_single_row(row: Dict[str, Any], source_tokenizer, target_tokenizer, embedder) -> Dict[str, Any]:
    """단일 행 처리"""
    
    try:
        # 입력 텍스트 추출
        src_text = str(row.get('원문', '')).strip()
        tgt_text = str(row.get('번역문', '')).strip()
        
        if not src_text or not tgt_text:
            return {
                'aligned_source': src_text,
                'aligned_target': tgt_text,
                'processing_info': {'error': 'Empty input text'}
            }
        
        # 임베딩 함수 준비
        embed_func = embedder.embed
        
        # 원문 토크나이저로 처리
        aligned_src, aligned_tgt, processing_info = source_tokenizer.process_row(
            src_text, tgt_text, embed_func
        )
        
        return {
            'aligned_source': aligned_src,
            'aligned_target': aligned_tgt,
            'processing_info': processing_info
        }
        
    except Exception as e:
        logger.error(f"행 처리 실패: {e}")
        return {
            'aligned_source': str(row.get('원문', '')),
            'aligned_target': str(row.get('번역문', '')),
            'processing_info': {'error': str(e)}
        }