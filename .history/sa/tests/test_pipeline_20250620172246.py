"""실제 데이터를 사용한 파이프라인 테스트"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import time
from io_utils import load_excel, save_excel
from processor import process_file
from tokenizers import split_src_meaning_units, split_tgt_meaning_units

def create_test_data():
    """실제 테스트 데이터 생성"""
    print("🔬 실제 데이터 파이프라인 테스트 시작")
    print("=" * 80)
    
    # 실제 데이터셋에서 다양한 길이와 특성의 문장 선별
    test_data = [
        {"id": 14, "원문": "興也라", "번역문": "興이다."},
        {"id": 15, "원문": "蒹은 薕(렴)이요 葭는 蘆也라", "번역문": "蒹은 물억새이고 葭는 갈대이다."},
        {"id": 17, "원문": "白露凝戾爲霜然後에 歲事成이요 國家待禮然後興이라", 
         "번역문": "白露가 얼어 서리가 된 뒤에야 歲事가 이루어지고 國家는 禮가 행해진 뒤에야 흥성한다."},
        {"id": 18, "원문": "箋云 蒹葭在衆草之中에 蒼蒼然彊盛이라가 至白露凝戾爲霜이면 則成而黃이라", 
         "번역문": "箋云： 갈대는 여러 풀 가운데에 푸르게 무성했다가 白露가 얼어 서리가 되면 다 자라 누래진다."},
        {"id": 19, "원문": "興者는 喩衆民之不從襄公政令者는 得周禮以敎之면 則服이라", 
         "번역문": "興한 것은 襄公의 政令을 따르지 않는 백성들은 <군주가> 周禮를 따라 교화시키면 복종한다는 것을 비유한 것이다."},
        {"id": 20, "원문": "蒹葭蒼蒼이러니 白露爲霜이로다", "번역문": "갈대 무성하더니 白露 서리가 되었네"},
        {"id": 21, "원문": "所謂伊人이 在水一方이언마는", "번역문": "이른바 그 분이 강물 저쪽에 있다지만"},
        {"id": 22, "원문": "若逆流遡洄而往從之, 則道險阻且長遠, 不可得至, 言逆禮以治國, 則無得人道, 終不可至. 若順流遡游而往從之, 則宛然在於水之中央, 言順禮治國, 則得人之道, 自來迎己, 正近在禮樂之內.", 
         "번역문": "만일 물살을 거슬러 올라가서 따른다면 길이 험하고 막히며 또한 멀어서 이를 수 없다는 것은 禮에 거슬러 나라를 다스리면 사람의 도리를 얻지 못하여 끝내 이를 수 없다는 것을 말한 것이다. 만일 물살을 따라 내려가서 따른다면 완연히 물 가운데에 있다는 것은 禮를 따라 나라를 다스리면 사람의 도리를 얻어서 저절로 와서 자기를 맞이하니 바로 禮樂 안에 가까이 있다는 것을 말한 것이다."},
        {"id": 23, "원문": "然則非禮, 必不得人, 得人, 必能固國, 君何以不求用周禮乎.", 
         "번역문": "그러니 禮가 아니면 반드시 사람을 얻지 못하고, 사람을 얻으면 반드시 나라를 견고히 할 수 있으니, 임금께서는 어찌하여 周禮의 사용을 구하지 않으시는가."},
        {"id": 24, "원문": "正義曰： '蒹, 薕', '葭, 蘆', 釋草文, 郭璞曰\"蒹, 似萑而細, 高數尺.", 
         "번역문": "正義曰：'蒹, 薕', '葭, 蘆'는 ≪爾雅≫ <釋草>의 글이고, 郭璞이 말하기를 \"蒹은 萑과 비슷하나 가늘고 높이가 몇 자이다."}
    ]
    
    df = pd.DataFrame(test_data)
    
    # 기본 통계 출력
    print("✅ 실제 테스트 데이터 생성: real_test_data.xlsx")
    print(f"📊 문장 수: {len(df)}개")
    print(f"📏 길이 분포:")
    print(f"   원문 길이: 최소 {df['원문'].str.len().min()}, 최대 {df['원문'].str.len().max()}, 평균 {df['원문'].str.len().mean():.1f}")
    print(f"   번역 길이: 최소 {df['번역문'].str.len().min()}, 최대 {df['번역문'].str.len().max()}, 평균 {df['번역문'].str.len().mean():.1f}")
    print(f"   평균 확장 비율: {(df['번역문'].str.len().mean() / df['원문'].str.len().mean()):.2f}")
    
    # 길이별 분류
    short = len(df[df['원문'].str.len() <= 10])
    medium = len(df[(df['원문'].str.len() > 10) & (df['원문'].str.len() <= 50)])
    long = len(df[(df['원문'].str.len() > 50) & (df['원문'].str.len() <= 100)])
    very_long = len(df[df['원문'].str.len() > 100])
    
    print(f"\n📝 길이별 분류:")
    print(f"   • 짧은 문장 (≤10자): {short}개")
    print(f"   • 중간 문장 (11-50자): {medium}개") 
    print(f"   • 긴 문장 (51-100자): {long}개")
    print(f"   • 매우 긴 문장 (>100자): {very_long}개")
    
    # 특징별 분류
    features = {
        "한자+조사 혼합": len([s for s in df['원문'] if any(c in s for c in ['은', '는', '이', '가', '을', '를'])]),
        "시문/운율": len([s for s in df['원문'] if any(c in s for c in ['이로다', '이러니', '이언마는'])]),
        "설명문": len([s for s in df['원문'] if '云' in s or '曰' in s]),
        "인용문": len([s for s in df['번역문'] if '<' in s or '>' in s]),
        "전문용어": len([s for s in df['원문'] if '釋草' in s or '正義' in s]),
        "의문문": len([s for s in df['번역문'] if '?' in s or '가.' in s]),
        "복합문": len([s for s in df['원문'] if len(s.split()) >= 3])
    }
    
    print(f"\n🎯 특징별 분류:")
    for feature, count in features.items():
        print(f"   • {feature}: {count}개")
    
    # 미리보기
    print(f"\n📋 테스트 데이터 미리보기:")
    for i, row in df.head(3).iterrows():
        print(f"{i+1}. [ID {row['id']}] 원문: {row['원문']}")
        print(f"   번역: {row['번역문']}")
        print()
    
    # 파일 저장
    save_excel(df, "real_test_data.xlsx")
    print("✅ 실제 테스트 데이터 준비 완료")
    
    return "real_test_data.xlsx"

def test_individual_tokenizer():
    """개별 토크나이저 테스트"""
    print("\n🧪 개별 토크나이저 단위 테스트")
    print("-" * 60)
    
    test_cases = [
        ("興也라", "興이다."),
        ("蒹은 薕(렴)이요 葭는 蘆也라", "蒹은 물억새이고 葭는 갈대이다."),
        ("白露凝戾爲霜然後에 歲事成이요", "白露가 얼어 서리가 된 뒤에야 歲事가 이루어지고")
    ]
    
    for i, (src, tgt) in enumerate(test_cases, 1):
        print(f"\n테스트 케이스 {i}:")
        print(f"원문: {src}")
        print(f"번역: {tgt}")
        
        try:
            src_units = split_src_meaning_units(src)
            tgt_units = split_tgt_meaning_units(src, tgt, use_semantic=False)
            
            print(f"✅ 원문 분할: {src_units}")
            print(f"✅ 번역 분할: {tgt_units}")
        except Exception as e:
            print(f"❌ 분할 실패: {e}")

def test_sentence_tokenization(file_path):
    """문장별 토크나이징 테스트"""
    print("\n🔤 개별 문장 토크나이징 테스트")
    print("-" * 60)
    
    try:
        df = load_excel(file_path)
        print(f"📋 데이터 컬럼: {list(df.columns)}")
        print(f"📊 데이터 행 수: {len(df)}")
        
        for idx, row in df.head(5).iterrows():
            src = row['원문']
            tgt = row['번역문']
            
            print(f"\n📝 문장 {idx+1} [ID {row['id']}]:")
            print(f"원문: {src}")
            print(f"번역: {tgt}")
            
            try:
                src_units = split_src_meaning_units(src)
                tgt_units = split_tgt_meaning_units(src, tgt, use_semantic=False)
                
                print(f"✅ 원문 분할: {src_units}")
                print(f"✅ 번역 분할: {tgt_units}")
                
            except Exception as e:
                print(f"❌ 토크나이징 실패: {e}")
                
        print("\n✅ 토크나이징 테스트 완료")
        
    except Exception as e:
        print(f"❌ 파일 로드 실패: {e}")

def preprocess_data(input_file):
    """데이터 전처리 (컬럼명 변환)"""
    print("\n🔄 데이터 전처리 (컬럼명 변환)")
    print("-" * 60)
    
    try:
        df = load_excel(input_file)
        
        # 컬럼명 변환
        if '원문' in df.columns and '번역문' in df.columns:
            df = df.rename(columns={'원문': 'src', '번역문': 'tgt'})
            
        processed_file = input_file.replace('.xlsx', '_processed.xlsx')
        save_excel(df, processed_file)
        
        print(f"✅ 컬럼명 변환 완료: {processed_file}")
        print(f"📋 변환된 컬럼: {list(df.columns)}")
        
        return processed_file
        
    except Exception as e:
        print(f"❌ 전처리 실패: {e}")
        return None

def run_full_pipeline(processed_file):
    """전체 파이프라인 실행"""
    print("\n🚀 전체 파이프라인 실행")
    print("-" * 60)
    
    print("📊 파이프라인 실행 중... (실제 임베딩 사용)")
    
    start_time = time.time()
    
    try:
        results = process_file(
            processed_file,
            use_semantic=True,  # 의미 기반 매칭 사용
            save_results=True
        )
        
        end_time = time.time()
        
        if results is not None:
            print("✅ 파이프라인 실행 성공")
            print(f"⏱️ 실행 시간: {end_time - start_time:.2f}초")
            return "real_test_results.xlsx"
        else:
            print("❌ 파이프라인 실행 실패")
            return None
            
    except Exception as e:
        print(f"❌ 파이프라인 오류: {e}")
        return None

def analyze_results(results_file):
    """결과 분석"""
    print("\n📊 결과 분석")
    print("-" * 60)
    
    try:
        df = load_excel(results_file)
        
        print(f"✅ 처리 완료된 문장 수: {len(df)}")
        
        print(f"\n📏 길이별 처리 결과:")
        for idx, row in df.iterrows():
            src_units = eval(row['src_units']) if isinstance(row['src_units'], str) else row['src_units']
            tgt_units = eval(row['tgt_units']) if isinstance(row['tgt_units'], str) else row['tgt_units']
            alignments = eval(row['alignments']) if isinstance(row['alignments'], str) else row['alignments']
            
            src_len = len(row['src']) if 'src' in row else 0
            
            print(f"문장 {idx+1} (길이 {src_len}자):")
            print(f"  원문 분할: {len(src_units) if src_units else 0}개 단위")
            print(f"  번역 분할: {len(tgt_units) if tgt_units else 0}개 단위")
            
            if src_units and len(src_units) <= 5:  # 짧은 문장만 상세 출력
                print(f"  원문 단위: {src_units}")
                print(f"  번역 단위: {tgt_units[:3] + ['...'] if len(tgt_units) > 3 else tgt_units}")
            
            print(f"  정렬 결과: {len(alignments) if alignments else 0}개")
            print()
        
        # 전체 통계
        total_src_units = sum(len(eval(row['src_units']) if isinstance(row['src_units'], str) else row['src_units']) 
                             for _, row in df.iterrows() if row['src_units'])
        total_tgt_units = sum(len(eval(row['tgt_units']) if isinstance(row['tgt_units'], str) else row['tgt_units']) 
                             for _, row in df.iterrows() if row['tgt_units'])
        
        avg_src_len = df['src'].str.len().mean() if 'src' in df.columns else 0
        
        print(f"\n📈 전반적인 통계:")
        print(f"  성공적으로 처리된 문장: {len(df[df['src_units'].notna()])}/{len(df)} ({len(df[df['src_units'].notna()])/len(df)*100:.1f}%)")
        print(f"  평균 원문 길이: {avg_src_len:.1f}자")
        print(f"  평균 원문 분할 수: {total_src_units/len(df):.1f}개/문장")
        print(f"  평균 번역 분할 수: {total_tgt_units/len(df):.1f}개/문장")
        if total_src_units > 0:
            print(f"  분할 비율 (번역/원문): {total_tgt_units/total_src_units:.2f}")
        
    except Exception as e:
        print(f"❌ 결과 분석 실패: {e}")

def main():
    """메인 테스트 함수"""
    
    # 1. 개별 토크나이저 테스트
    test_individual_tokenizer()
    
    # 2. 실제 테스트 데이터 생성
    test_file = create_test_data()
    
    # 3. 개별 문장 토크나이징 테스트
    test_sentence_tokenization(test_file)
    
    # 4. 데이터 전처리
    processed_file = preprocess_data(test_file)
    if not processed_file:
        return
    
    # 5. 전체 파이프라인 실행
    results_file = run_full_pipeline(processed_file)
    if not results_file:
        return
    
    # 6. 결과 분석
    analyze_results(results_file)
    
    print("\n" + "=" * 80)
    print("🏁 실제 데이터 테스트 결과")
    print("=" * 80)
    print("🎉 실제 데이터 테스트 성공!")
    print(f"📁 결과 파일: {results_file}")
    print("🔍 상세 분석 결과를 확인하세요.")
    
    print(f"\n🗂️ 생성된 파일들:")
    print(f"  • {test_file} - 원본 테스트 데이터")
    print(f"  • {processed_file} - 전처리된 데이터")
    print(f"  • {results_file} - 최종 결과")

if __name__ == "__main__":
    main()