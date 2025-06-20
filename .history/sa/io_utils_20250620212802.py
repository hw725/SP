"""파일 입출력 유틸리티 - 고정 컬럼명"""

import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def load_excel_file(file_path: str) -> Optional[pd.DataFrame]:
    """Excel 파일 로드 - 고정 컬럼명"""
    logger.info(f"📂 파일 로딩 중: {file_path}")
    
    try:
        df = pd.read_excel(file_path)
        logger.info(f"✅ 파일 로드 성공: {len(df)}개 행")
        
        # 고정 컬럼명으로 변환
        expected_columns = ['문장식별자', '원문', '번역문']
        
        # 컬럼 존재 확인
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"❌ 필수 컬럼 누락: {missing_columns}")
            logger.info(f"📋 현재 컬럼: {list(df.columns)}")
            return None
        
        # 내부 처리용으로 컬럼명 변경
        df_processed = df.rename(columns={
            '문장식별자': 'id',
            '원문': 'src', 
            '번역문': 'tgt'
        })
        
        logger.info(f"✅ 컬럼 매핑 완료: {len(df_processed)}개 행")
        return df_processed
        
    except Exception as e:
        logger.error(f"❌ 파일 로드 실패: {e}")
        return None

def save_alignment_results(df: pd.DataFrame, file_path: str) -> bool:
    """정렬 결과 저장"""
    logger.info(f"💾 결과 저장 중: {file_path}")
    
    try:
        # 결과 저장 시 원래 컬럼명으로 복원 (선택사항)
        df_output = df.copy()
        
        # 원하는 컬럼 순서로 정렬
        output_columns = ['id', 'src', 'tgt', 'src_units', 'tgt_units', 'alignments', 
                         'src_count', 'tgt_count', 'alignment_count', 'status']
        
        # 존재하는 컬럼만 선택
        available_columns = [col for col in output_columns if col in df_output.columns]
        df_output = df_output[available_columns]
        
        df_output.to_excel(file_path, index=False)
        
        logger.info(f"✅ 결과 저장 완료: {file_path}")
        logger.info(f"📊 저장된 데이터: {len(df_output)}개 행, {len(df_output.columns)}개 컬럼")
        return True
        
    except Exception as e:
        logger.error(f"❌ 결과 저장 실패: {e}")
        return False

def save_phrase_format_results(df: pd.DataFrame, file_path: str) -> bool:
    """구 단위별 분할 결과 저장"""
    logger.info(f"💾 구 단위 결과 저장 중: {file_path}")
    
    try:
        phrase_data = []
        
        for _, row in df.iterrows():
            sentence_id = row.get('id', 0)
            src_units = row.get('src_units', [])
            tgt_units = row.get('tgt_units', [])
            
            # 리스트가 아닌 경우 처리
            if isinstance(src_units, str):
                try:
                    import ast
                    src_units = ast.literal_eval(src_units)
                except:
                    src_units = []
            
            if isinstance(tgt_units, str):
                try:
                    import ast
                    tgt_units = ast.literal_eval(tgt_units)
                except:
                    tgt_units = []
            
            # 구별 데이터 생성
            max_units = max(len(src_units), len(tgt_units))
            for i in range(max_units):
                src_unit = src_units[i] if i < len(src_units) else ""
                tgt_unit = tgt_units[i] if i < len(tgt_units) else ""
                
                phrase_data.append({
                    '문장식별자': sentence_id,
                    '구식별자': i + 1,
                    '원문구': src_unit,
                    '번역구': tgt_unit
                })
        
        # 구 단위 DataFrame 저장
        phrase_df = pd.DataFrame(phrase_data)
        phrase_df.to_excel(file_path, index=False)
        
        logger.info(f"✅ 구 단위 결과 저장 완료: {len(phrase_data)}개 구")
        return True
        
    except Exception as e:
        logger.error(f"❌ 구 단위 결과 저장 실패: {e}")
        return False

# 기존 함수명과의 호환성
load_excel = load_excel_file
save_excel = save_alignment_results