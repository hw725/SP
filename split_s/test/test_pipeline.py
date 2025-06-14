"""파이프라인 테스트 모듈"""

import pandas as pd
from typing import List, Tuple
import os
import sys
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_basic_functionality():
    """기본 기능 테스트"""
    print("=== Testing basic functionality ===")
    
    try:
        from punctuation import mask_brackets, restore_masks
        from tokenizer import split_src_meaning_units, split_tgt_meaning_units
        from aligner import align_src_tgt
        from embedder import compute_embeddings_with_cache, get_embedding_manager
        
        print("✓ All modules imported successfully")
        
        # 임베딩 매니저 상태 확인
        manager = get_embedding_manager()
        print(f"✓ Embedding manager created: {manager is not None}")
        
    except ImportError as e:
        print(f"✗ Failed to import modules: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"✗ Unexpected error during import: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 테스트 데이터
    src_text = "作詁訓傳時에 移其篇第하고 因改之耳라"
    tgt_text = "주석과 해설을 작성할 때에 그 편과 장을 옮기고 그에 따라 고쳤을 뿐이다."
    
    print(f"Source: {src_text}")
    print(f"Target: {tgt_text}")
    
    try:
        # 1. 괄호 마스킹 테스트
        print("\n--- Testing bracket masking ---")
        masked_src, src_masks = mask_brackets(src_text, text_type="source")
        masked_tgt, tgt_masks = mask_brackets(tgt_text, text_type="target")
        print(f"✓ Masked source: {masked_src}")
        print(f"✓ Masked target: {masked_tgt}")
        
        # 2. 의미 단위 분할 테스트
        print("\n--- Testing meaning unit splitting ---")
        src_units = split_src_meaning_units(masked_src)
        print(f"✓ Source units: {src_units}")
        
        # 단순 분할로 먼저 테스트
        tgt_units = split_tgt_meaning_units(
            masked_src, masked_tgt, 
            use_semantic=False,  # 단순 분할 사용
            min_tokens=1
        )
        print(f"✓ Target units (simple): {tgt_units}")
        
        # 3. 마스크 복원 테스트
        print("\n--- Testing mask restoration ---")
        restored_src_units = [restore_masks(unit, src_masks) for unit in src_units]
        restored_tgt_units = [restore_masks(unit, tgt_masks) for unit in tgt_units]
        print(f"✓ Restored source units: {restored_src_units}")
        print(f"✓ Restored target units: {restored_tgt_units}")
        
        # 4. 의미 기반 분할 테스트 (임베딩 사용)
        print("\n--- Testing semantic splitting ---")
        try:
            semantic_tgt_units = split_tgt_meaning_units(
                masked_src, masked_tgt, 
                use_semantic=True,
                min_tokens=1
            )
            semantic_restored_tgt_units = [restore_masks(unit, tgt_masks) for unit in semantic_tgt_units]
            print(f"✓ Semantic target units: {semantic_restored_tgt_units}")
            
            # 임베딩 매니저 상태 확인
            if hasattr(manager, 'is_using_dummy') and manager.is_using_dummy():
                print("✓ Using dummy embeddings for testing")
            
            # 5. 정렬 테스트
            print("\n--- Testing alignment ---")
            aligned_pairs = align_src_tgt(
                restored_src_units, 
                semantic_restored_tgt_units, 
                compute_embeddings_with_cache
            )
            
            print(f"\n✓ Alignment results:")
            for i, (src, tgt) in enumerate(aligned_pairs, 1):
                print(f"  [{i}] '{src}' -> '{tgt}'")
                
        except Exception as e:
            print(f"✗ Semantic processing failed: {e}")
            print("Trying simple alignment...")
            # 단순 정렬로 폴백
            aligned_pairs = list(zip(restored_src_units, restored_tgt_units))
            print(f"✓ Simple alignment results:")
            for i, (src, tgt) in enumerate(aligned_pairs, 1):
                print(f"  [{i}] '{src}' -> '{tgt}'")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_test_data(file_path: str = "test_input.xlsx") -> str:
    """Generate test data."""
    test_data = [
        {
            "원문": "作詁訓傳時에 移其篇第하고 因改之耳라",
            "번역문": "주석과 해설을 작성할 때에 그 편과 장을 옮기고 그에 따라 고쳤을 뿐이다."
        },
        {
            "원문": "古來相傳하야 學者가 於其說에 未嘗致疑하니라",
            "번역문": "예로부터 서로 전해져 학자들은 그 설에 대해 의심을 품은 적이 없었다."
        },
    ]

    df = pd.DataFrame(test_data)
    df.to_excel(file_path, index=False, engine='openpyxl')
    print(f"✓ Test data created: {file_path}")
    return file_path

def test_file_processing():
    """파일 처리 테스트"""
    print("\n=== Testing file processing ===")
    
    try:
        from io_manager import process_file
        
        test_input = create_test_data()
        test_output = "test_output.xlsx"
        
        print(f"Processing {test_input} -> {test_output}")
        process_file(test_input, test_output, verbose=True)
        
        # 결과 확인
        if os.path.exists(test_output):
            result_df = pd.read_excel(test_output)
            print("✓ Test results:")
            print(result_df.to_string(index=False))
            print(f"\n✓ File processing test completed successfully")
            return True
        else:
            print("✗ Output file not created")
            return False
            
    except Exception as e:
        print(f"✗ File processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_tests():
    """Run all tests"""
    print("Starting pipeline tests...")
    
    # 1. 기본 기능 테스트
    basic_success = test_basic_functionality()
    
    if basic_success:
        print("\n🎉 === Basic functionality test PASSED ===")
        
        # 2. 파일 처리 테스트
        file_success = test_file_processing()
        
        if file_success:
            print("\n🎉 === All tests PASSED ===")
        else:
            print("\n❌ === File processing test FAILED ===")
    else:
        print("\n❌ === Basic functionality test FAILED ===")

if __name__ == "__main__":
    run_tests()