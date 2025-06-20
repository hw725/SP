"""파일 입출력 유틸리티"""

import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

def load_excel_file(file_path: str) -> Optional[pd.DataFrame]:
    """
    엑셀 파일 로드
    
    Args:
        file_path: 파일 경로
    
    Returns:
        pd.DataFrame 또는 None
    """
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"❌ 파일이 존재하지 않습니다: {file_path}")
            return None
        
        if not file_path.suffix.lower() in ['.xlsx', '.xls']:
            logger.error(f"❌ 지원하지 않는 파일 형식: {file_path.suffix}")
            return None
        
        logger.info(f"📂 파일 로딩 중: {file_path}")
        
        # 엑셀 파일 읽기
        df = pd.read_excel(file_path, engine='openpyxl')
        
        # 기본 검증
        if df.empty:
            logger.warning(f"⚠️ 빈 파일: {file_path}")
            return None
        
        # 필수 컬럼 확인
        required_columns = ['src', 'tgt']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"❌ 필수 컬럼 누락: {missing_columns}")
            logger.info(f"📋 현재 컬럼: {list(df.columns)}")
            return None
        
        # 데이터 정리
        df['src'] = df['src'].astype(str).str.strip()
        df['tgt'] = df['tgt'].astype(str).str.strip()
        
        # 빈 행 제거
        df = df[df['src'].notna() & df['tgt'].notna()]
        df = df[df['src'] != ''] 
        df = df[df['tgt'] != '']
        
        logger.info(f"✅ 파일 로드 성공: {len(df)}개 행")
        return df
        
    except Exception as e:
        logger.error(f"❌ 파일 로드 실패: {e}")
        return None

def save_alignment_results(results: List[Dict], output_file: str) -> bool:
    """
    정렬 결과를 엑셀로 저장
    
    Args:
        results: 정렬 결과 리스트
        output_file: 출력 파일 경로
    
    Returns:
        bool: 성공 여부
    """
    try:
        if not results:
            logger.error("❌ 저장할 결과가 없습니다")
            return False
        
        logger.info(f"💾 결과 저장 중: {output_file}")
        
        # DataFrame 생성
        df = pd.DataFrame(results)
        
        # 컬럼 순서 정리
        column_order = [
            'id', 'src', 'tgt', 
            'src_units', 'tgt_units', 'alignments',
            'src_count', 'tgt_count', 'alignment_count'
        ]
        
        # 존재하는 컬럼만 선택
        existing_columns = [col for col in column_order if col in df.columns]
        remaining_columns = [col for col in df.columns if col not in existing_columns]
        
        final_columns = existing_columns + remaining_columns
        df = df[final_columns]
        
        # 엑셀 저장
        output_path = Path(output_file)
        df.to_excel(output_path, index=False, engine='openpyxl')
        
        logger.info(f"✅ 결과 저장 완료: {output_path}")
        logger.info(f"📊 저장된 데이터: {len(df)}개 행, {len(df.columns)}개 컬럼")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 결과 저장 실패: {e}")
        return False

def save_detailed_results(results: List[Dict], output_file: str) -> bool:
    """
    상세 결과를 여러 시트로 저장
    
    Args:
        results: 결과 리스트
        output_file: 출력 파일 경로
    
    Returns:
        bool: 성공 여부
    """
    try:
        if not results:
            logger.error("❌ 저장할 결과가 없습니다")
            return False
        
        logger.info(f"📋 상세 결과 저장 중: {output_file}")
        
        output_path = Path(output_file)
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 1. 요약 시트
            summary_data = []
            for result in results:
                summary_data.append({
                    'id': result.get('id', ''),
                    'src': result.get('src', ''),
                    'tgt': result.get('tgt', ''),
                    'src_count': result.get('src_count', 0),
                    'tgt_count': result.get('tgt_count', 0),
                    'alignment_count': result.get('alignment_count', 0),
                    'error': result.get('error', '')
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_sheet(writer, sheet_name='요약', index=False)
            
            # 2. 분할 결과 시트
            tokenization_data = []
            for result in results:
                try:
                    src_units = eval(result.get('src_units', '[]'))
                    tgt_units = eval(result.get('tgt_units', '[]'))
                    
                    tokenization_data.append({
                        'id': result.get('id', ''),
                        'src_units': ' | '.join(src_units) if src_units else '',
                        'tgt_units': ' | '.join(tgt_units) if tgt_units else ''
                    })
                except:
                    tokenization_data.append({
                        'id': result.get('id', ''),
                        'src_units': result.get('src_units', ''),
                        'tgt_units': result.get('tgt_units', '')
                    })
            
            tokenization_df = pd.DataFrame(tokenization_data)
            tokenization_df.to_sheet(writer, sheet_name='분할결과', index=False)
            
            # 3. 정렬 결과 시트
            alignment_data = []
            for result in results:
                try:
                    alignments = eval(result.get('alignments', '[]'))
                    for align in alignments:
                        alignment_data.append({
                            'id': result.get('id', ''),
                            'src_idx': align.get('src_idx', ''),
                            'tgt_idx': align.get('tgt_idx', ''),
                            'src_text': align.get('src_text', ''),
                            'tgt_text': align.get('tgt_text', ''),
                            'confidence': align.get('confidence', 0)
                        })
                except:
                    pass
            
            if alignment_data:
                alignment_df = pd.DataFrame(alignment_data)
                alignment_df.to_sheet(writer, sheet_name='정렬결과', index=False)
        
        logger.info(f"✅ 상세 결과 저장 완료: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ 상세 결과 저장 실패: {e}")
        return False

def validate_input_file(file_path: str) -> bool:
    """
    입력 파일 유효성 검증
    
    Args:
        file_path: 파일 경로
    
    Returns:
        bool: 유효성 여부
    """
    try:
        df = load_excel_file(file_path)
        return df is not None
        
    except Exception as e:
        logger.error(f"❌ 파일 검증 실패: {e}")
        return False

def get_file_info(file_path: str) -> Optional[Dict]:
    """
    파일 정보 가져오기
    
    Args:
        file_path: 파일 경로
    
    Returns:
        Dict: 파일 정보 또는 None
    """
    try:
        df = load_excel_file(file_path)
        if df is None:
            return None
        
        info = {
            'file_path': str(file_path),
            'total_rows': len(df),
            'columns': list(df.columns),
            'src_avg_length': df['src'].str.len().mean() if 'src' in df else 0,
            'tgt_avg_length': df['tgt'].str.len().mean() if 'tgt' in df else 0,
            'empty_src': df[df['src'].str.strip() == ''].shape[0] if 'src' in df else 0,
            'empty_tgt': df[df['tgt'].str.strip() == ''].shape[0] if 'tgt' in df else 0
        }
        
        return info
        
    except Exception as e:
        logger.error(f"❌ 파일 정보 가져오기 실패: {e}")
        return None

if __name__ == "__main__":
    # 테스트
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 IO 유틸리티 테스트")
    
    # 테스트 데이터 생성
    test_data = [
        {'id': 1, 'src': '興也라', 'tgt': '興이다.'},
        {'id': 2, 'src': '蒹은 薕이요', 'tgt': '蒹은 물억새이고'}
    ]
    
    test_df = pd.DataFrame(test_data)
    test_file = "test_io.xlsx"
    test_df.to_excel(test_file, index=False)
    
    # 로드 테스트
    loaded_df = load_excel_file(test_file)
    if loaded_df is not None:
        print(f"✅ 로드 테스트 성공: {len(loaded_df)}행")
    else:
        print("❌ 로드 테스트 실패")
    
    # 저장 테스트
    test_results = [
        {
            'id': 1, 'src': '興也라', 'tgt': '興이다.',
            'src_units': "['興也라']", 'tgt_units': "['興이', '다.']",
            'alignments': "[]", 'src_count': 1, 'tgt_count': 2, 'alignment_count': 0
        }
    ]
    
    success = save_alignment_results(test_results, "test_results.xlsx")
    if success:
        print("✅ 저장 테스트 성공")
    else:
        print("❌ 저장 테스트 실패")
    
    # 정리
    import os
    try:
        os.remove(test_file)
        os.remove("test_results.xlsx")
    except:
        pass