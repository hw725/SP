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
        
        # 컬럼명 확인 및 출력
        print(f"📋 데이터 컬럼: {list(df.columns)}")
        print(f"📊 데이터 행 수: {len(df)}")
        
        for i, row in df.head(5).iterrows():  # 처음 5개만 테스트
            # 컬럼명 맞춤
            src = row['원문']  # 'src' → '원문'
            tgt = row['번역문']  # 'tgt' → '번역문'
            
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
        
        print(f"\n✅ 토크나이징 테스트 완료")
    
    except Exception as e:
        print(f"❌ 토크나이징 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. 컬럼명 변환을 위한 전처리
    print(f"\n🔄 데이터 전처리 (컬럼명 변환)")
    print("-" * 60)
    
    try:
        # 기존 시스템 호환을 위해 컬럼명 변환
        df_processed = df.copy()
        df_processed = df_processed.rename(columns={
            '원문': 'src',
            '번역문': 'tgt'
        })
        
        # 임시 파일로 저장
        processed_file = "real_test_data_processed.xlsx"
        df_processed.to_excel(processed_file, index=False)
        
        print(f"✅ 컬럼명 변환 완료: {processed_file}")
        print(f"📋 변환된 컬럼: {list(df_processed.columns)}")
        
    except Exception as e:
        print(f"❌ 데이터 전처리 실패: {e}")
        return False
    
    # 4. 전체 파이프라인 실행
    print(f"\n🚀 전체 파이프라인 실행")
    print("-" * 60)
    
    try:
        from processor import process_file
        
        output_file = "real_test_results.xlsx"
        
        print(f"📊 파이프라인 실행 중... (실제 임베딩 사용)")
        start_time = time.time()
        
        success = process_file(
            input_file=processed_file,  # 전처리된 파일 사용
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
            analyze_results(output_file, processed_file)
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
                
                # 분할 결과 미리보기
                if src_units:
                    print(f"  원문 단위: {src_units}")
                if tgt_units:
                    tgt_preview = tgt_units[:3] + ['...'] if len(tgt_units) > 3 else tgt_units
                    print(f"  번역 단위: {tgt_preview}")
                
                if 'alignments' in df_result.columns:
                    alignments = eval(row.get('alignments', '[]'))
                    print(f"  정렬 결과: {len(alignments)}개")
                
                print()  # 빈 줄
                
            except Exception as e:
                print(f"문장 {i+1}: 결과 파싱 실패 - {e}")
        
        # 전반적인 통계
        print(f"\n📈 전반적인 통계:")
        successful = len([r for r in df_result.iterrows() if r[1].get('src_units', '[]') != '[]'])
        print(f"  성공적으로 처리된 문장: {successful}/{len(df_result)} ({successful/len(df_result)*100:.1f}%)")
        
        # 처리 시간 대비 성능
        avg_src_len = sum(len(row['src']) for _, row in df_original.iterrows()) / len(df_original)
        print(f"  평균 원문 길이: {avg_src_len:.1f}자")
        
        # 분할 효율성 분석
        total_src_units = 0
        total_tgt_units = 0
        
        for _, row in df_result.iterrows():
            try:
                src_units = eval(row.get('src_units', '[]'))
                tgt_units = eval(row.get('tgt_units', '[]'))
                total_src_units += len(src_units)
                total_tgt_units += len(tgt_units)
            except:
                pass
        
        if len(df_result) > 0:
            print(f"  평균 원문 분할 수: {total_src_units/len(df_result):.1f}개/문장")
            print(f"  평균 번역 분할 수: {total_tgt_units/len(df_result):.1f}개/문장")
            if total_src_units > 0:
                print(f"  분할 비율 (번역/원문): {total_tgt_units/total_src_units:.2f}")
        
    except Exception as e:
        print(f"❌ 결과 분석 실패: {e}")
        import traceback
        traceback.print_exc()

def test_individual_tokenization():
    """개별 토크나이저 단위 테스트"""
    print(f"\n🧪 개별 토크나이저 단위 테스트")
    print("-" * 60)
    
    test_cases = [
        ("興也라", "興이다."),
        ("蒹은 薕(렴)이요 葭는 蘆也라", "蒹은 물억새이고 葭는 갈대이다."),
        ("白露凝戾爲霜然後에 歲事成이요", "白露가 얼어 서리가 된 뒤에야 歲事가 이루어지고")
    ]
    
    try:
        from tokenizer import split_src_meaning_units, split_tgt_meaning_units
        
        for i, (src, tgt) in enumerate(test_cases, 1):
            print(f"\n테스트 케이스 {i}:")
            print(f"원문: {src}")
            print(f"번역: {tgt}")
            
            # 원문 분할
            try:
                src_units = split_src_meaning_units(src)
                print(f"✅ 원문 분할: {src_units}")
            except Exception as e:
                print(f"❌ 원문 분할 실패: {e}")
            
            # 번역문 분할 (간단 모드)
            try:
                def simple_embed_func(texts):
                    import numpy as np
                    return [np.random.randn(10) for _ in texts]
                
                tgt_units = split_tgt_meaning_units(
                    src, tgt, 
                    embed_func=simple_embed_func,
                    use_semantic=False
                )
                print(f"✅ 번역 분할: {tgt_units}")
            except Exception as e:
                print(f"❌ 번역 분할 실패: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 개별 토크나이저 테스트 실패: {e}")
        return False

def main():
    """실제 데이터 테스트 메인"""
    
    # 개별 토크나이저 테스트 먼저 수행
    tokenizer_success = test_individual_tokenization()
    
    if not tokenizer_success:
        print(f"\n⚠️ 개별 토크나이저 테스트 실패로 인해 전체 테스트를 중단합니다.")
        return False
    
    # 전체 파이프라인 테스트
    success = test_with_real_data()
    
    print(f"\n{'='*80}")
    print(f"🏁 실제 데이터 테스트 결과")
    print(f"{'='*80}")
    
    if success:
        print("🎉 실제 데이터 테스트 성공!")
        print("📁 결과 파일: real_test_results.xlsx")
        print("🔍 상세 분석 결과를 확인하세요.")
        
        # 파일 정리 옵션
        print(f"\n🗂️ 생성된 파일들:")
        print(f"  • real_test_data.xlsx - 원본 테스트 데이터")
        print(f"  • real_test_data_processed.xlsx - 전처리된 데이터")
        print(f"  • real_test_results.xlsx - 최종 결과")
        
    else:
        print("⚠️ 실제 데이터 테스트 실패")
        print("🔧 문제점을 해결한 후 다시 시도하세요.")
    
    return success

if __name__ == "__main__":
    main()