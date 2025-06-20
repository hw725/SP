"""mecab + jieba 통합 테스트"""

def test_tokenizer_integration():
    """토크나이저 통합 테스트"""
    from tokenizer import split_src_meaning_units, split_tgt_meaning_units
    
    # 테스트 데이터
    src_text = "子曰學而時習之不亦說乎"
    tgt_text = "공자께서 말씀하셨다 배우고 때때로 익히면 또한 기쁘지 아니한가"
    
    print("🧪 토크나이저 통합 테스트")
    print(f"원문: {src_text}")
    print(f"번역: {tgt_text}")
    
    # 원문 분할 테스트
    try:
        src_units = split_src_meaning_units(src_text)
        print(f"✅ 원문 분할 성공: {src_units}")
    except Exception as e:
        print(f"❌ 원문 분할 실패: {e}")
        return False
    
    # 번역문 분할 테스트 (더미 임베딩 사용)
    try:
        def dummy_embed_func(texts):
            import numpy as np
            return [np.random.randn(100) for _ in texts]
        
        tgt_units = split_tgt_meaning_units(
            src_text, tgt_text, 
            embed_func=dummy_embed_func,
            use_semantic=False  # 단순 모드로 테스트
        )
        print(f"✅ 번역 분할 성공: {tgt_units}")
    except Exception as e:
        print(f"❌ 번역 분할 실패: {e}")
        return False
    
    return True

def test_mecab_functionality():
    """MeCab 기능 테스트"""
    from tokenizer import calculate_mecab_completeness
    
    test_texts = [
        "공자께서",
        "말씀하셨다", 
        "배우고",
        "때때로 익히면"
    ]
    
    print("\n🧪 MeCab 기능 테스트")
    for text in test_texts:
        try:
            score = calculate_mecab_completeness(text)
            print(f"✅ '{text}' → 완전성 점수: {score:.3f}")
        except Exception as e:
            print(f"❌ '{text}' → 오류: {e}")
            return False
    
    return True

def test_jieba_functionality():
    """jieba 기능 테스트"""
    from tokenizer import get_jieba_boundaries
    
    test_texts = [
        "子曰學而時習之",
        "不亦說乎",
        "學而時習"
    ]
    
    print("\n🧪 jieba 기능 테스트")
    for text in test_texts:
        try:
            boundaries = get_jieba_boundaries(text)
            print(f"✅ '{text}' → 경계: {boundaries}")
        except Exception as e:
            print(f"❌ '{text}' → 오류: {e}")
            return False
    
    return True

def main():
    """전체 통합 테스트"""
    print("🔬 MeCab + jieba 통합 테스트 시작")
    
    tests = [
        ("토크나이저 통합", test_tokenizer_integration),
        ("MeCab 기능", test_mecab_functionality), 
        ("jieba 기능", test_jieba_functionality),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"테스트: {name}")
        print(f"{'='*50}")
        
        try:
            result = test_func()
            results.append((name, result))
            print(f"✅ {name}: {'통과' if result else '실패'}")
        except Exception as e:
            results.append((name, False))
            print(f"❌ {name}: 예외 발생 - {e}")
    
    # 결과 요약
    print(f"\n{'='*50}")
    print("🏁 테스트 결과 요약")
    print(f"{'='*50}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 통과" if result else "❌ 실패" 
        print(f"{name}: {status}")
    
    print(f"\n총 {total}개 테스트 중 {passed}개 통과 ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 모든 테스트 통과!")
        return True
    else:
        print("⚠️ 일부 테스트 실패")
        return False

if __name__ == "__main__":
    main()