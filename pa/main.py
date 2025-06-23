"""PA 메인 실행기 - 완전 버전"""

import sys
import os
import argparse

def check_dependencies():
    """의존성 확인"""
    
    missing = []
    
    try:
        import pandas
    except ImportError:
        missing.append("pandas")
    
    try:
        import spacy
    except ImportError:
        missing.append("spacy")
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    # ✅ sentence_transformers 체크 추가
    try:
        import sentence_transformers
    except ImportError:
        missing.append("sentence_transformers")
    
    if missing:
        print(f"❌ 필수 패키지 누락: {', '.join(missing)}")
        print("설치 명령: pip install " + " ".join(missing))
        return False
    
    # SA 모듈 확인
    sa_path = os.path.join(os.path.dirname(__file__), '..', 'sa')
    if not os.path.exists(sa_path):
        print(f"❌ SA 모듈 경로가 없습니다: {sa_path}")
        return False
    
    return True

def main():
    print("🚀 PA (Paragraph Aligner) 시작")
    
    # 의존성 확인
    if not check_dependencies():
        return
    
    parser = argparse.ArgumentParser(description="PA: Paragraph Aligner")
    parser.add_argument("input_file", help="입력 파일 (Excel) - 컬럼: 원문, 번역문")
    parser.add_argument("output_file", help="출력 파일 (Excel) - 컬럼: 문단식별자, 원문, 번역문")
    parser.add_argument("--embedder", default="bge", choices=["bge", "st", "openai"])
    parser.add_argument("--threshold", type=float, default=0.3, help="유사도 임계값")
    parser.add_argument("--max-length", type=int, default=150, help="최대 문장 길이")
    parser.add_argument("--parallel", action="store_true", help="병렬 처리 (미구현)")  # ✅ 추가
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--device", default="cuda", help="임베더 연산 디바이스 (cuda/gpu/cpu, 기본값: cuda)")  # 기본값을 cuda로 변경
    parser.add_argument("--splitter", default="spacy", choices=["spacy", "stanza"], help="문장 분할기 선택")

    args = parser.parse_args()
    
    # 병렬 처리 경고
    if args.parallel:
        print("⚠️ --parallel 옵션은 현재 미구현 상태입니다.")
    
    # 파일 존재 확인
    if not os.path.exists(args.input_file):
        print(f"❌ 입력 파일이 없습니다: {args.input_file}")
        return
    
    # 출력 디렉토리 생성
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 출력 디렉토리 생성: {output_dir}")
    
    try:
        from processor import process_paragraph_file
        
        result_df = process_paragraph_file(
            args.input_file,
            args.output_file,
            embedder_name=args.embedder,
            max_length=args.max_length,
            similarity_threshold=args.threshold,
            device=args.device,
            splitter=args.splitter   # splitter 인자 전달
        )
        
        if result_df is not None:
            print(f"\n✅ PA 처리 완료!")
            print(f"입력: {args.input_file}")
            print(f"출력: {args.output_file}")
            print(f"결과: {len(result_df)}개 문장 쌍")
        else:
            print("\n❌ PA 처리 실패!")
            
    except Exception as e:
        print(f"\n❌ 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()