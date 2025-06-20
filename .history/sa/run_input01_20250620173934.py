"""input01.xlsx 실제 데이터 처리 실행 - 고정 컬럼명"""

from processor import process_file
import logging
import time
import os

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s:%(name)s:%(message)s'
)

def main():
    """input01.xlsx 처리 메인 함수"""
    
    print("🚀 input01.xlsx 실제 데이터 처리 시작")
    print("=" * 80)
    
    # 파일 존재 확인
    input_file = "input01.xlsx"
    if not os.path.exists(input_file):
        print(f"❌ 파일을 찾을 수 없습니다: {input_file}")
        return
    
    # 파일 정보 확인
    try:
        import pandas as pd
        df_info = pd.read_excel(input_file)
        print(f"📊 입력 파일 정보:")
        print(f"   파일명: {input_file}")
        print(f"   행 수: {len(df_info)}개")
        print(f"   컬럼: {list(df_info.columns)}")
        
        # 고정 컬럼명 확인
        expected_columns = ['문장식별자', '원문', '번역문']
        if all(col in df_info.columns for col in expected_columns):
            print("✅ 컬럼명 확인 완료")
        else:
            print("⚠️ 예상 컬럼명과 다릅니다")
        
        print(f"   첫 번째 행 미리보기:")
        if len(df_info) > 0:
            print(f"     문장식별자: {df_info.iloc[0]['문장식별자']}")
            print(f"     원문: {str(df_info.iloc[0]['원문'])[:50]}...")
            print(f"     번역문: {str(df_info.iloc[0]['번역문'])[:50]}...")
        
    except Exception as e:
        print(f"⚠️ 파일 정보 확인 실패: {e}")
    
    print("\n" + "-" * 60)
    
    # 처리 시작
    start_time = time.time()
    
    try:
        print("🔄 처리 중... (131개 문장 - 시간이 걸립니다)")
        
        results = process_file(
            input_file,
            use_semantic=True,        # 의미 기반 매칭 사용
            min_tokens=1,            # 최소 토큰 수
            max_tokens=10,           # 최대 토큰 수  
            save_results=True,       # 결과 저장
            output_file="input01_results.xlsx"
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if results is not None:
            print(f"\n🎉 처리 성공!")
            print(f"⏱️  처리 시간: {processing_time:.2f}초")
            print(f"📊 처리 결과:")
            print(f"   처리된 문장 수: {len(results)}개")
            print(f"   결과 파일: input01_results.xlsx")
            
            # 성공/실패 통계
            if 'status' in results.columns:
                success_count = len(results[results['status'] == 'success'])
                print(f"   성공률: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
            
            # 평균 분할 수
            if 'src_count' in results.columns and 'tgt_count' in results.columns:
                avg_src = results['src_count'].mean()
                avg_tgt = results['tgt_count'].mean()
                print(f"   평균 원문 분할: {avg_src:.1f}개/문장")
                print(f"   평균 번역 분할: {avg_tgt:.1f}개/문장")
            
            # 처리 속도
            sentences_per_sec = len(results) / processing_time
            print(f"   처리 속도: {sentences_per_sec:.2f}문장/초")
            
            print(f"\n📁 결과 파일: input01_results.xlsx")
            print("🔍 Excel에서 결과를 확인하세요!")
            
        else:
            print(f"\n❌ 처리 실패")
            
    except Exception as e:
        print(f"\n💥 처리 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("🏁 input01.xlsx 처리 완료")

if __name__ == "__main__":
    main()