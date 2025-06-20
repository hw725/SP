"""파일 입출력 처리"""
import logging
import pandas as pd
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """데이터 파일 로드"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
    
    try:
        if file_path.suffix.lower() == '.xlsx':
            df = pd.read_excel(file_path)
        elif file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path, encoding='utf-8')
        else:
            # 확장자 불명시 -> 자동 감지 시도
            try:
                df = pd.read_excel(file_path)
            except:
                df = pd.read_csv(file_path, encoding='utf-8')
        
        logger.info(f"데이터 로드 성공: {file_path} ({df.shape[0]}행, {df.shape[1]}열)")
        return df
        
    except Exception as e:
        logger.error(f"데이터 로드 실패: {file_path} - {e}")
        raise

def save_results(df: pd.DataFrame, file_path: str) -> None:
    """결과 데이터 저장"""
    file_path = Path(file_path)
    
    # 출력 디렉토리 생성
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if file_path.suffix.lower() == '.xlsx':
            df.to_excel(file_path, index=False, engine='openpyxl')
        elif file_path.suffix.lower() == '.csv':
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
        else:
            # 확장자 불명시 -> 기본 Excel 저장
            xlsx_path = file_path.with_suffix('.xlsx')
            df.to_excel(xlsx_path, index=False, engine='openpyxl')
            logger.info(f"확장자 불명시로 Excel 형태로 저장: {xlsx_path}")
            return
        
        logger.info(f"결과 저장 성공: {file_path} ({df.shape[0]}행, {df.shape[1]}열)")
        
    except Exception as e:
        logger.error(f"결과 저장 실패: {file_path} - {e}")
        raise

def validate_input_data(df: pd.DataFrame) -> pd.DataFrame:
    """입력 데이터 검증 및 정제"""
    
    # 필수 컬럼 확인
    required_columns = ['원문', '번역문']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")
    
    # 빈 행 제거
    initial_rows = len(df)
    df = df.dropna(subset=['원문', '번역문'])
    df = df[(df['원문'].str.strip() != '') & (df['번역문'].str.strip() != '')]
    
    removed_rows = initial_rows - len(df)
    if removed_rows > 0:
        logger.info(f"빈 행 {removed_rows}개 제거됨")
    
    # 문장식별자가 없으면 자동 생성
    if '문장식별자' not in df.columns:
        df['문장식별자'] = range(1, len(df) + 1)
        logger.info("문장식별자 자동 생성됨")
    
    return df.reset_index(drop=True)

def get_file_info(file_path: str) -> dict:
    """파일 정보 반환"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        return {'exists': False}
    
    return {
        'exists': True,
        'size': file_path.stat().st_size,
        'extension': file_path.suffix,
        'name': file_path.name,
        'parent': str(file_path.parent)
    }