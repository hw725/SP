"""파일 입출력 관련 함수들"""
import pandas as pd
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def load_data(file_path: str) -> List[Dict[str, str]]:
    """Excel 파일에서 데이터 로드"""
    try:
        if file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {file_path}")
        
        df = df.fillna('')
        return df.to_dict('records')
    except FileNotFoundError:
        logger.error(f"파일을 찾을 수 없습니다: {file_path}")
        raise
    except Exception as e:
        logger.error(f"데이터 로드 실패 ({file_path}): {e}", exc_info=True)
        raise

def save_data(data: List[Dict[str, Any]], file_path: str):
    """데이터를 Excel 파일로 저장"""
    if not data:
        logger.warning("저장할 데이터가 없습니다.")
        return
    try:
        df = pd.DataFrame(data)
        
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(lambda x: str(x) if isinstance(x, (list, dict)) else x)

        if file_path.endswith('.xlsx'):
            df.to_excel(file_path, index=False, engine='openpyxl')
        elif file_path.endswith('.csv'):
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
        else:
            logger.info(f"파일 확장자가 지정되지 않아 .xlsx로 저장합니다: {file_path}.xlsx")
            df.to_excel(file_path + '.xlsx', index=False, engine='openpyxl')
            
        logger.info(f"데이터가 {file_path} (또는 .xlsx)에 저장되었습니다.")
    except Exception as e:
        logger.error(f"데이터 저장 실패 ({file_path}): {e}", exc_info=True)
        raise