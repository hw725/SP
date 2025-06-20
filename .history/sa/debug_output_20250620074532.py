"""출력 파일 구조 디버깅"""
import pandas as pd
import tempfile
import sys
import os

sys.path.append('src')

def debug_output_structure():
    """출력 파일 구조 확인"""
    
    # 테스트 데이터 생성
    test_data = {
        '원문': ["中國人民解放軍은 强력한 軍隊이다."],
        '번역문': ["The Chinese People's Liberation Army is a powerful military force."]
    }
    df = pd.DataFrame(test_data)
    
    # 임시 파일 생성
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as input_file:
        input_path = input_file.name
        df.to_excel(input_path, index=False)
    
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as output_file:
        output_path = output_file.name
    
    try:
        # Config 생성
        from src.config import Config
        config = Config(
            input_path=input_path,
            output_path=output_path,
            source_tokenizer_type="prototype02",
            target_tokenizer_type="prototype02",
            embedder_type="bge-m3",
            use_parallel=False,
            verbose=True
        )
        
        # 파이프라인 실행
        from src.orchestrator import run_processing
        run_processing(config)
        
        # 결과 확인
        if os.path.exists(output_path):
            result_df = pd.read_excel(output_path)
            print(f"출력 파일 컬럼들: {list(result_df.columns)}")
            print(f"출력 파일 크기: {result_df.shape}")
            
            # 첫 번째 행 내용 확인
            print("\n첫 번째 행 내용:")
            for col in result_df.columns:
                print(f"  {col}: {result_df.iloc[0][col]}")
        else:
            print("출력 파일이 생성되지 않았습니다.")
            
    finally:
        # 임시 파일 정리
        try:
            os.unlink(input_path)
            os.unlink(output_path)
        except:
            pass

if __name__ == "__main__":
    debug_output_structure()