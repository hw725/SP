"""파일 입출력 관련 함수들"""
import pandas as pd
import logging
from typing import List, Dict, Any
import json

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
        if isinstance(processing_info, str):
            try:
                processing_info = json.loads(processing_info)
            except json.JSONDecodeError:
                logger.warning(f"processing_info 문자열 파싱 실패: {original_row_dict.get('문장식별자', '알 수 없는 ID')}")
                processing_info = {}

        if processing_info.get('status') != 'success':
            logger.warning(f"처리 실패 또는 정보 부족으로 행 확장 건너뜀: ID {original_row_dict.get('문장식별자', '알 수 없는 ID')}")
            continue

        sentence_id = original_row_dict.get('문장식별자')
        if sentence_id is None:
            sentence_id = original_row_dict.get('id', 'ID_없음')
            if sentence_id == 'ID_없음':
                logger.warning(f"문장식별자를 찾을 수 없는 행: {original_row_dict}")

        aligned_source_str = original_row_dict.get('aligned_source')
        aligned_target_str = original_row_dict.get('aligned_target')

        if not isinstance(aligned_source_str, str) or not isinstance(aligned_target_str, str):
            logger.warning(f"aligned_source 또는 aligned_target이 문자열이 아니거나 누락되어 행 확장 건너뜀: ID {sentence_id}")
            continue

        source_phrases = [phrase.strip() for phrase in aligned_source_str.split('|')]
        target_phrases = [phrase.strip() for phrase in aligned_target_str.split('|')]

        num_phrases = min(len(source_phrases), len(target_phrases))
        if len(source_phrases) != len(target_phrases):
            logger.warning(f"ID {sentence_id}: 원문구({len(source_phrases)})와 번역구({len(target_phrases)})의 개수가 불일치합니다. 최소 개수에 맞춰 처리합니다.")

        for i in range(num_phrases):
            if not source_phrases[i] and not target_phrases[i]:
                continue
            
            expanded_row = {
                '문장식별자': sentence_id,
                '구식별자': i + 1,
                '원문구': source_phrases[i],
                '번역구': target_phrases[i]
            }
            expanded_data_for_saving.append(expanded_row)

    if not expanded_data_for_saving:
        logger.warning("확장하여 저장할 데이터가 없습니다.")
        return

    try:
        df = pd.DataFrame(expanded_data_for_saving)
        
        if file_path.endswith(".xlsx"):
            df.to_excel(file_path, index=False, engine='openpyxl')
        elif file_path.endswith(".csv"):
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
        else:
            logger.error(f"지원하지 않는 파일 형식입니다: {file_path}. .xlsx 또는 .csv를 사용하세요.")
            return
        logger.info(f"데이터가 여러 행으로 확장되어 {file_path}에 성공적으로 저장되었습니다.")
    except Exception as e:
        logger.error(f"데이터 저장 중 오류 발생 ({file_path}): {e}", exc_info=True)
        raise