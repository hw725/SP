"""Prototype02 통합 상태 철저 점검"""
import sys
import os
sys.path.append('src')

def check_prototype02_integration():
    """Prototype02 통합 완성도 검사"""
    print("=== Prototype02 통합 상태 점검 ===\n")
    
    issues = []
    
    # 1. 핵심 클래스 임포트 테스트
    try:
        from src.text_alignment import (
            TextMasker, 
            SourceTextSplitter, 
            TargetTextAligner,
            TextAlignmentProcessor
        )
        print("✓ 핵심 클래스들 임포트 성공")
    except ImportError as e:
        issues.append(f"✗ 핵심 클래스 임포트 실패: {e}")
    
    # 2. 하위 호환성 함수 테스트
    try:
        from src.text_alignment import (
            split_src_meaning_units,
            split_tgt_meaning_units,
            mask_brackets,
            restore_masks
        )
        print("✓ 하위 호환성 함수들 임포트 성공")
    except ImportError as e:
        issues.append(f"✗ 하위 호환성 함수 임포트 실패: {e}")
    
    # 3. 컴포넌트 레지스트리 테스트
    try:
        from src.components import get_embedder, get_tokenizer
        print("✓ 컴포넌트 팩토리 임포트 성공")
    except ImportError as e:
        issues.append(f"✗ 컴포넌트 팩토리 임포트 실패: {e}")
    
    # 4. 파이프라인 테스트
    try:
        from src.pipeline import process_single_row
        print("✓ 파이프라인 함수 임포트 성공")
    except ImportError as e:
        issues.append(f"✗ 파이프라인 함수 임포트 실패: {e}")
    
    # 5. 마스킹 기능 테스트
    try:
        masker = TextMasker()
        test_text = "이것은 (테스트) 문장입니다."
        masked, masks = masker.mask(test_text, "source")
        restored = masker.restore(masked, masks)
        
        if restored == test_text:
            print("✓ 마스킹/언마스킹 기능 정상")
        else:
            issues.append(f"✗ 마스킹/언마스킹 불일치: '{test_text}' != '{restored}'")
    except Exception as e:
        issues.append(f"✗ 마스킹 기능 테스트 실패: {e}")
    
    # 6. 소스 텍스트 분할 테스트
    try:
        splitter = SourceTextSplitter()
        test_text = "中國人民 解放軍은 強力한 軍隊입니다."
        units = splitter.split(test_text)
        
        if units and len(units) > 1:
            print(f"✓ 원문 분할 기능 정상: {len(units)}개 단위")
        else:
            issues.append(f"✗ 원문 분할 결과 부적절: {units}")
    except Exception as e:
        issues.append(f"✗ 원문 분할 테스트 실패: {e}")
    
    # 7. 의존성 검사
    required_modules = ['soynlp', 'regex', 'numpy']
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module} 모듈 사용 가능")
        except ImportError:
            issues.append(f"✗ {module} 모듈 누락")
    
    # 결과 요약
    print(f"\n=== 점검 결과 ===")
    if issues:
        print(f"발견된 문제: {len(issues)}개")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("✓ 모든 점검 통과 - Prototype02 통합 완료!")
        return True

if __name__ == "__main__":
    check_prototype02_integration()