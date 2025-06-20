"""실제 데이터에서 선별한 테스트 문장들"""

import pandas as pd
from pathlib import Path
import logging
import re

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_real_test_data():
    """첨부된 데이터에서 선별한 테스트 문장들"""
    
    # 다양한 길이와 패턴의 문장들을 선별 (컬럼명을 시스템 표준에 맞춤)
    test_data = [
        # 1. 짧은 한문장 (기본 테스트)
        {
            "id": 14,
            "원문": "興也라",
            "번역문": "興이다."
        },
        
        # 2. 중간 길이 (조사/어미 포함)
        {
            "id": 15,
            "원문": "蒹은 薕(렴)이요 葭는 蘆也라",
            "번역문": "蒹은 물억새이고 葭는 갈대이다."
        },
        
        # 3. 복합문 (접속사 포함)
        {
            "id": 17,
            "원문": "白露凝戾爲霜然後에 歲事成이요 國家待禮然後興이라",
            "번역문": "白露가 얼어 서리가 된 뒤에야 歲事가 이루어지고 國家는 禮가 행해진 뒤에야 흥성한다."
        },
        
        # 4. 긴 설명문 (복잡한 구조)
        {
            "id": 18,
            "원문": "箋云 蒹葭在衆草之中에 蒼蒼然彊盛이라가 至白露凝戾爲霜이면 則成而黃이라",
            "번역문": "箋云： 갈대는 여러 풀 가운데에 푸르게 무성했다가 白露가 얼어 서리가 되면 다 자라 누래진다."
        },
        
        # 5. 비유문 (은유적 표현)
        {
            "id": 19,
            "원문": "興者는 喩衆民之不從襄公政令者는 得周禮以敎之면 則服이라",
            "번역문": "興한 것은 襄公의 政令을 따르지 않는 백성들은 <군주가> 周禮를 따라 교화시키면 복종한다는 것을 비유한 것이다."
        },
        
        # 6. 시문 (운율이 있는 원문)
        {
            "id": 13,
            "원문": "蒹葭蒼蒼이러니 白露爲霜이로다",
            "번역문": "갈대 무성하더니 白露 서리가 되었네"
        },
        
        # 7. 의문문
        {
            "id": 20,
            "원문": "所謂伊人이 在水一方이언마는",
            "번역문": "이른바 그 분이 강물 저쪽에 있건만"
        },
        
        # 8. 매우 긴 복합문 (최고 난이도) - 문자열 분할하여 연결
        {
            "id": 41,
            "원문": ("若逆流遡洄而往從之, 則道險阻且長遠, 不可得至, 言逆禮以治國, 則無得人道, 終不可至. " +
                    "若順流遡游而往從之, 則宛然在於水之中央, 言順禮治國, 則得人之道, 自來迎己, 正近在禮樂之內."),
            "번역문": ("만일 물살을 거슬러 올라가서 따른다면 길이 험하고 막히며 멀어서 도달할 수 없다는 것은, " +
                     "禮에 어긋나게 나라를 다스리면 사람을 얻는 방도가 없어서 끝내 이를 수 없음을 말한 것이고, " +
                     "물살에 순응하며 따라 내려가 만나려 하면 宛然히 물 가운데 있다는 것은, " +
                     "禮에 따라 나라를 다스리면 사람을 얻는 道이니 절로 와서 나를 맞이함이 바로 禮樂의 안에 가까이 있음을 말한 것이다.")
        },
        
        # 9. 인용문 (따옴표 포함)
        {
            "id": 42,
            "원문": "然則非禮, 必不得人, 得人, 必能固國, 君何以不求用周禮乎.",
            "번역문": "그러니 禮가 아니면 반드시 사람을 얻을 수 없고, 사람을 얻어야 반드시 나라를 견고하게 할 수 있는데, 군주는 어찌하여 周禮 따름을 추구하지 않는가."
        },
        
        # 10. 전문 용어 설명 (사전식 정의) - 인용부호 이스케이프 처리
        {
            "id": 52,
            "원문": "正義曰：'蒹 薕 葭 蘆', 釋草文, 郭璞曰\"蒹, 似萑而細, 高數尺.",
            "번역문": "正義曰：'蒹, 薕', '葭, 蘆'는 ≪爾雅≫ <釋草>의 글인데 郭璞은 \"蒹은 물억새(萑)와 비슷한데 가늘고 키가 數尺이다."
        }
    ]
    
    # 데이터 검증
    for i, row in enumerate(test_data):
        if not isinstance(row['id'], int):
            logger.warning(f"Row {i}: ID가 정수가 아닙니다: {row['id']}")
            row['id'] = int(row['id'])
        
        if not row['원문'].strip():
            logger.error(f"Row {i}: 원문이 비어있습니다")
            
        if not row['번역문'].strip():
            logger.error(f"Row {i}: 번역문이 비어있습니다")
    
    try:
        # DataFrame 생성
        df = pd.DataFrame(test_data)
        
        # 데이터 타입 명시적 설정
        df['id'] = df['id'].astype(int)
        df['원문'] = df['원문'].astype(str)
        df['번역문'] = df['번역문'].astype(str)
        
        # 중복 ID 확인
        if df['id'].duplicated().any():
            logger.warning("중복된 ID가 발견되었습니다:")
            duplicates = df[df['id'].duplicated(keep=False)]['id'].unique()
            logger.warning(f"중복 ID: {duplicates}")
        
        # 엑셀 파일로 저장
        output_path = Path("real_test_data.xlsx")
        df.to_excel(output_path, index=False, engine='openpyxl')
        
        print(f"✅ 실제 테스트 데이터 생성: {output_path}")
        print(f"📊 문장 수: {len(test_data)}개")
        
        # 안전한 길이별 통계 계산
        src_lengths = [len(str(row['원문'])) for row in test_data if row['원문']]
        tgt_lengths = [len(str(row['번역문'])) for row in test_data if row['번역문']]
        
        if src_lengths and tgt_lengths:
            print(f"📏 길이 분포:")
            print(f"   원문 길이: 최소 {min(src_lengths)}, 최대 {max(src_lengths)}, 평균 {sum(src_lengths)/len(src_lengths):.1f}")
            print(f"   번역 길이: 최소 {min(tgt_lengths)}, 최대 {max(tgt_lengths)}, 평균 {sum(tgt_lengths)/len(tgt_lengths):.1f}")
            print(f"   평균 확장 비율: {(sum(tgt_lengths)/sum(src_lengths)):.2f}")
        
        # 길이별 분류 개선
        length_categories = {
            'short': [i for i, length in enumerate(src_lengths) if length <= 10],
            'medium': [i for i, length in enumerate(src_lengths) if 10 < length <= 50],
            'long': [i for i, length in enumerate(src_lengths) if 50 < length <= 100],
            'very_long': [i for i, length in enumerate(src_lengths) if length > 100]
        }
        
        print(f"\n📝 길이별 분류:")
        print(f"   • 짧은 문장 (≤10자): {len(length_categories['short'])}개")
        print(f"   • 중간 문장 (11-50자): {len(length_categories['medium'])}개") 
        print(f"   • 긴 문장 (51-100자): {len(length_categories['long'])}개")
        print(f"   • 매우 긴 문장 (>100자): {len(length_categories['very_long'])}개")
        
        # 특징별 분류 개선
        feature_analysis = analyze_text_features(test_data)
        print(f"\n🎯 특징별 분류:")
        for feature, count in feature_analysis.items():
            print(f"   • {feature}: {count}개")
        
        # 미리보기
        print(f"\n📋 테스트 데이터 미리보기:")
        for i, row in enumerate(test_data[:3], 1):
            src_preview = str(row['원문'])[:50] + ('...' if len(str(row['원문'])) > 50 else '')
            tgt_preview = str(row['번역문'])[:50] + ('...' if len(str(row['번역문'])) > 50 else '')
            print(f"{i}. [ID {row['id']}] 원문: {src_preview}")
            print(f"   번역: {tgt_preview}\n")
        
        return output_path
        
    except Exception as e:
        logger.error(f"❌ 파일 생성 중 오류 발생: {e}")
        raise

def analyze_text_features(test_data):
    """텍스트 특징 분석"""
    features = {
        '한자+조사 혼합': 0,
        '시문/운율': 0,
        '설명문': 0,
        '인용문': 0,
        '전문용어': 0,
        '의문문': 0,
        '복합문': 0
    }
    
    for row in test_data:
        src = str(row['원문'])
        tgt = str(row['번역문'])
        
        # 한자+조사 혼합 (한자 뒤에 한글 조사)
        if re.search(r'[\u4e00-\u9fff][\uac00-\ud7af]{1,2}(?=\s|[\u4e00-\u9fff]|$)', src):
            features['한자+조사 혼합'] += 1
        
        # 시문/운율 (특정 어미나 감탄사)
        if re.search(r'[이]?로다|[이]?러니|哉|也', src):
            features['시문/운율'] += 1
        
        # 설명문 (箋云, 正義曰 등)
        if re.search(r'箋云|正義曰|釋.*文', src):
            features['설명문'] += 1
        
        # 인용문 (따옴표나 인용 표시)
        if '"' in src or '"' in tgt or '曰' in src:
            features['인용문'] += 1
        
        # 전문용어 (釋草, 爾雅 등)
        if re.search(r'釋草|爾雅|郭璞', src + tgt):
            features['전문용어'] += 1
        
        # 의문문
        if re.search(r'乎[.?]?$|가\?|은가', tgt):
            features['의문문'] += 1
        
        # 복합문 (접속사나 복문 구조)
        if re.search(r'然後|則|若.*則|而|且', src) or len(src) > 50:
            features['복합문'] += 1
    
    return features

def validate_test_data():
    """생성된 테스트 데이터 검증"""
    try:
        output_path = Path("real_test_data.xlsx")
        if not output_path.exists():
            print(f"❌ 테스트 파일이 존재하지 않습니다: {output_path}")
            return False
        
        df = pd.read_excel(output_path)
        
        # 필수 컬럼 확인
        required_columns = ['id', '원문', '번역문']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"❌ 필수 컬럼 누락: {missing_columns}")
            return False
        
        # 데이터 무결성 확인
        empty_src = df[df['원문'].isna() | (df['원문'].str.strip() == '')].index.tolist()
        empty_tgt = df[df['번역문'].isna() | (df['번역문'].str.strip() == '')].index.tolist()
        
        if empty_src:
            print(f"❌ 빈 원문이 있는 행: {empty_src}")
            return False
        
        if empty_tgt:
            print(f"❌ 빈 번역문이 있는 행: {empty_tgt}")
            return False
        
        print(f"✅ 테스트 데이터 검증 완료: {len(df)}개 행, 모든 검사 통과")
        return True
        
    except Exception as e:
        print(f"❌ 데이터 검증 중 오류: {e}")
        return False

if __name__ == "__main__":
    try:
        # 테스트 데이터 생성
        output_path = create_real_test_data()
        
        # 검증 수행
        if validate_test_data():
            print(f"🎉 테스트 데이터 생성 및 검증 완료!")
            print(f"📁 파일 위치: {output_path.absolute()}")
        else:
            print("❌ 데이터 검증 실패")
    
    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {e}")