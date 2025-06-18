"""파일 입출력 관련 함수들"""
import pandas as pd
import logging
from typing import List, Dict, Any
import json
import numpy as np   # 추가

logger = logging.getLogger(__name__)

def load_data(file_path: str) -> List[Dict[str, str]]:
    """Excel 또는 CSV 파일에서 데이터 로드"""
    try:
        if file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path, engine='openpyxl')
        elif file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            logger.error(f"지원하지 않는 파일 형식입니다: {file_path}. .xlsx 또는 .csv를 사용하세요.")
            return []
        
        df = df.astype(str)
        df.replace('nan', '', inplace=True)
        return df.to_dict('records')
    except FileNotFoundError:
        logger.error(f"파일을 찾을 수 없습니다: {file_path}")
        raise
    except Exception as e:
        logger.error(f"데이터 로드 중 오류 발생 ({file_path}): {e}", exc_info=True)
        raise

def save_data(data: List[Dict[str, Any]], file_path: str):
    """
    데이터를 여러 행으로 확장하여 Excel 또는 CSV 파일로 저장합니다.
    각 행은 '문장식별자', '구식별자', '원문구', '번역구' 컬럼을 가집니다.
    """
    if not data:
        logger.warning("저장할 데이터가 없습니다.")
        return

    expanded_data_for_saving = []
    
    for original_row_dict in data:
        processing_info = original_row_dict.get('processing_info', {})
        if processing_info.get('status') != 'success':
            continue
        
        # 원본 ID 및 순서 보존
        sentence_id = original_row_dict.get('문장식별자') or original_row_dict.get('id', None)
        row_order = original_row_dict.get('__row_order', None)
        full_src = original_row_dict.get('원문', '')
        full_tgt = original_row_dict.get('번역문', '')

        source_phrases = [p.strip() for p in original_row_dict['aligned_source'].split('|')]
        target_phrases = [p.strip() for p in original_row_dict['aligned_target'].split('|')]
        num_phrases = min(len(source_phrases), len(target_phrases))

        for i in range(num_phrases):
            src_seg = source_phrases[i] or full_src
            tgt_seg = target_phrases[i] or full_tgt

            expanded_row = {
                '__row_order': row_order,            # 정렬용
                '문장식별자': sentence_id,           # 원본 ID
                '구식별자': i + 1,                   # phrase id
                '원문_전체': full_src,               # 원본 전체 텍스트
                '번역문_전체': full_tgt,             # 번역문 전체 텍스트
                '원문구': src_seg,                   # 분할된 원문구
                '번역구': tgt_seg                    # 분할된 번역구
            }
            expanded_data_for_saving.append(expanded_row)

    if not expanded_data_for_saving:
        logger.warning("확장하여 저장할 데이터가 없습니다.")
        return

    # DataFrame 생성 후 원본 순서·구식별자 기준으로 정렬
    df = pd.DataFrame(expanded_data_for_saving)
    if '__row_order' in df.columns:
        df.sort_values(by=['__row_order', '구식별자'], inplace=True)
        df.drop(columns=['__row_order'], inplace=True)

    # NaN 대체
    df = df.fillna('')

    # 파일 저장
    if file_path.endswith(".xlsx"):
        df.to_excel(file_path, index=False, engine='openpyxl')
    elif file_path.endswith(".csv"):
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
    else:
        logger.error(f"지원하지 않는 파일 형식입니다: {file_path}")
        return

    logger.info(f"데이터가 {file_path}에 성공적으로 저장되었습니다.")