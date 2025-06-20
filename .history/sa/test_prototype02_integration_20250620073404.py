"""Prototype02 통합 완성도 실행 테스트"""
import sys
import os
import tempfile
import pandas as pd

sys.path.append('src')

def create_test_data():
    """테스트용 데이터 생성"""
    test_data = {
        '원문': [
            "中國人民解放軍은 强力한 軍隊이다.",
            "這個 (내용)은 重要한 部分입니다.",
            "日本語로 번역하면 어떻게 될까요?",
        ],
        '번역문': [
            "The Chinese People's Liberation Army is a powerful military force.",
            "This (content) is an important part.",
            "How would it be translated into Japanese?",
        ]
    }
    return pd.DataFrame(test_data)

def test_full_pipeline():
    """전체 파이프라인 테스트"""
    print("=== Prototype02 통합 파이프라인 테스트 ===\n")
    
    try:
        # 테스트 데이터 생성
        df = create_test_data()
        
        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as input_file:
            input_path = input_file.name
            df.to_excel(input_path, index=False)
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as output_file:
            output_path = output_file.name
        
        print(f"입력 파일: {input_path}")
        print(f"출력 파일: {output_path}")
        
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
        print("\n파이프라인 실행 중...")
        run_processing(config)
        
        # 결과 확인
        if os.path.exists(output_path):
            result_df = pd.read_excel(output_path)
            print(f"\n✓ 파이프라인 실행 성공!")
            print(f"  - 입력 행 수: {len(df)}")
            print(f"  - 출력 행 수: {len(result_df)}")
            
            # 결과 컬럼 확인
            required_columns = ['aligned_source', 'aligned_target']
            missing_columns = [col for col in required_columns if col not in result_df.columns]
            
            if missing_columns:
                print(f"  ✗ 누락된 컬럼: {missing_columns}")
                return False
            else:
                print(f"  ✓ 필수 컬럼 모두 존재")
                
                # 샘플 결과 출력
                print(f"\n--- 처리 결과 샘플 ---")
                for i, row in result_df.head(2).iterrows():
                    print(f"행 {i+1}:")
                    print(f"  원문: {row.get('원문', 'N/A')}")
                    print(f"  정렬된 원문: {row.get('aligned_source', 'N/A')}")
                    print(f"  번역문: {row.get('번역문', 'N/A')}")
                    print(f"  정렬된 번역문: {row.get('aligned_target', 'N/A')}")
                    print()
                
                return True
        else:
            print(f"✗ 출력 파일이 생성되지 않았습니다: {output_path}")
            return False
            
    except Exception as e:
        print(f"✗ 파이프라인 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 임시 파일 정리
        try:
            if 'input_path' in locals():
                os.unlink(input_path)
            if 'output_path' in locals():
                os.unlink(output_path)
        except:
            pass

def test_individual_components():
    """개별 컴포넌트 테스트"""
    print("=== 개별 컴포넌트 테스트 ===\n")
    
    try:
        from src.text_alignment import TextAlignmentProcessor
        from src.embedders import BGEM3Embedder
        
        # 임베더 초기화
        embedder = BGEM3Embedder()
        processor = TextAlignmentProcessor()
        
        # 테스트 케이스
        src_text = "中國人民解放軍은 (강력한) 軍隊이다."
        tgt_text = "The Chinese People's Liberation Army is a [powerful] military force."
        
        print(f"원문: {src_text}")
        print(f"번역문: {tgt_text}")
        
        # 처리 실행
        aligned_src, aligned_tgt, info = processor.process(src_text, tgt_text, embedder.embed)
        
        print(f"\n처리 결과:")
        print(f"  정렬된 원문: {aligned_src}")
        print(f"  정렬된 번역문: {aligned_tgt}")
        print(f"  처리 정보: {info}")
        
        if info.get('status') == 'success':
            print("✓ 개별 컴포넌트 테스트 성공!")
            return True
        else:
            print(f"✗ 처리 실패: {info.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"✗ 개별 컴포넌트 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Prototype02 통합 완성도 테스트를 시작합니다...\n")
    
    # 개별 컴포넌트 테스트
    component_test_passed = test_individual_components()
    print()
    
    # 전체 파이프라인 테스트
    pipeline_test_passed = test_full_pipeline()
    print()
    
    # 최종 결과
    if component_test_passed and pipeline_test_passed:
        print("🎉 모든 테스트 통과! Prototype02 통합이 완벽하게 완료되었습니다.")
    else:
        print("❌ 일부 테스트 실패. 통합에 문제가 있습니다.")
        if not component_test_passed:
            print("  - 개별 컴포넌트 테스트 실패")
        if not pipeline_test_passed:
            print("  - 파이프라인 테스트 실패")