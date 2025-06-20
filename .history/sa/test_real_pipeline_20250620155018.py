"""실제 데이터로 전체 파이프라인 테스트"""

import pandas as pd
import logging
import time
from pathlib import Path

def test_with_real_data():
    """실제 선별된 데이터로 파이프라인 테스트"""
    
    print("🔬 실제 데이터 파이프라인 테스트 시작")
    print("=" * 80)
    
    # 1. 실제 테스트 데이터 생성
    try:
        from create_real_test_data import create_real_test_data
        test_file = create_real_test_data()
        print(f"✅ 실제 테스트 데이터 준비 완료")
    except Exception as e:
        print(f"❌ 테스트 데이터 준비 실패: {e}")
        return False
    
    # 2. 개별 문장 토크나이징 테스트
    print(f"\n🔤 개별 문장 토크나이징 테스트")
    print("-" * 60)
    
    try:
        from tokenizer import split_src_meaning_units, split_tgt_meaning_units
        
        df = pd.read_excel(test_file)
        
        for i, row in df.head(5).iterrows():  # 처음 5개만 테스트
            src = row['src']
            tgt = row['tgt']
            
            print(f"\n📝 문장 {i+1} [ID {row['id']}]:")
            print(f"원문: {src}")
            print(f"번역: {tgt}")
            
            # 원문 분할
            try:
                src_units = split_src_meaning_units(src)
                print(f"✅ 원문 분할: {src_units}")
            except Exception as e:
                print(f"❌ 원문 분할 실패: {e}")
                continue
            
            # 번역문 분할 (더미 임베딩 사용)
            try:
                def dummy_embed_func(texts):
                    import numpy as np
                    return [np.random.randn(100) for _ in texts]
                
                tgt_units = split_tgt_meaning_units(
                    src, tgt, 
                    embed_func=dummy_embed_func,
                    use_semantic=False  # 단순 모드
                )
                print(f"✅ 번역 분할: {tgt_units}")
            except Exception as e:
                print(f"❌ 번역 분할 실패: {e}")
    
    except Exception as e:
        print(f"❌ 토크나이징 테스트 실패: {e}")
        return False
    
    # 3. 전체 파이프라인 실행
    print(f"\n🚀 전체 파이프라인 실행")
    print("-" * 60)
    
    try:
        from processor import process_file
        
        output_file = "real_test_results.xlsx"
        
        print(f"📊 파이프라인 실행 중... (실제 임베딩 사용)")
        start_time = time.time()
        
        success = process_file(
            input_file=str(test_file),
            output_file=output_file,
            use_semantic=True,  # 의미 기반 모드
            min_tokens=1,
            max_tokens=15      # 긴 문장 고려해서 늘림
        )
        
        end_time = time.time()
        
        if success:
            print(f"✅ 파이프라인 실행 성공")
            print(f"⏱️ 실행 시간: {end_time - start_time:.2f}초")
            
            # 결과 분석
            analyze_results(output_file, test_file)
            return True
        else:
            print(f"❌ 파이프라인 실행 실패")
            return False
            
    except Exception as e:
        print(f"❌ 파이프라인 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_results(result_file: str, original_file: str):
    """결과 분석"""
    
    print(f"\n📊 결과 분석")
    print("-" * 60)
    
    try:
        df_result = pd.read_excel(result_file)
        df_original = pd.read_excel(original_file)
        
        print(f"✅ 처리 완료된 문장 수: {len(df_result)}")
        
        # 길이별 성능 분석
        print(f"\n📏 길이별 처리 결과:")
        for i, row in df_result.iterrows():
            src_len = len(row['src'])
            
            # 분할 결과 확인
            try:
                src_units = eval(row.get('src_units', '[]')) if 'src_units' in row else []
                tgt_units = eval(row.get('tgt_units', '[]')) if 'tgt_units' in row else []
                
                print(f"문장 {i+1} (길이 {src_len}자):")
                print(f"  원문 분할: {len(src_units)}개 단위")
                print(f"  번역 분할: {len(tgt_units)}개 단위")
                
                if 'alignments' in df_result.columns:
                    alignments = eval(row.get('alignments', '[]'))
                    print(f"  정렬 결과: {len(alignments)}개")
                
            except Exception as e:
                print(f"문장 {i+1}: 결과 파싱 실패 - {e}")
        
        # 전반적인 통계
        print(f"\n📈 전반적인 통계:")
        successful = len([r for r in df_result.iterrows() if r[1].get('src_units', '[]') != '[]'])
        print(f"  성공적으로 처리된 문장: {successful}/{len(df_result)} ({successful/len(df_result)*100:.1f}%)")
        
        # 처리 시간 대비 성능
        avg_src_len = sum(len(row['src']) for _, row in df_original.iterrows()) / len(df_original)
        print(f"  평균 원문 길이: {avg_src_len:.1f}자")
        
    except Exception as e:
        print(f"❌ 결과 분석 실패: {e}")

def main():
    """실제 데이터 테스트 메인"""
    
    success = test_with_real_data()
    
    print(f"\n{'='*80}")
    print(f"🏁 실제 데이터 테스트 결과")
    print(f"{'='*80}")
    
    if success:
        print("🎉 실제 데이터 테스트 성공!")
        print("📁 결과 파일: real_test_results.xlsx")
        print("🔍 상세 분석 결과를 확인하세요.")
    else:
        print("⚠️ 실제 데이터 테스트 실패")
        print("🔧 문제점을 해결한 후 다시 시도하세요.")
    
    return success

if __name__ == "__main__":
    main()