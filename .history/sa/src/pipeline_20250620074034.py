"""통합된 TextAlignmentProcessor 사용 - 완전 정리된 버전"""
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
    """TextAlignmentProcessor를 사용한 단순화된 처리"""
    
    src_text = str(row_data.get('원문', ''))
    tgt_text = str(row_data.get('번역문', ''))

    if not src_text or not tgt_text:
        return {
            'aligned_source': src_text,
            'aligned_target': tgt_text,
            'processing_info': {'error': 'Empty source or target text'}
        }
    
    try:
        # *** 통합된 TextAlignmentProcessor 사용 ***
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
