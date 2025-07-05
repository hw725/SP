"""SA 정렬 시스템 메인 실행기"""

import argparse
import logging
import time
import sys
import os
from typing import Optional

import sys
sys.stdout.reconfigure(encoding='utf-8')
from core.io_utils import IOManager

def setup_logging(verbose: bool = False):
    """로깅 설정"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s:%(name)s:%(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('sa_processing.log', encoding='utf-8')
        ]
    )

def get_tokenizer_module(tokenizer_name: str):
    """토크나이저 모듈 동적 로드 - jieba(원문), mecab(번역문)만 지원"""
    tokenizer_map = {
        'jieba': 'core.tokenizers.jieba_mecab',
        'mecab': 'core.tokenizers.jieba_mecab',
    }
    if tokenizer_name not in tokenizer_map:
        raise ValueError(f"지원하지 않는 토크나이저: {tokenizer_name}. 지원: jieba(원문), mecab(번역문)")
    module_name = tokenizer_map[tokenizer_name]
    try:
        import importlib
        return importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"토크나이저 모듈 로드 실패: {module_name} ({e})")

def get_embedder_module(embedder_name: str):
    """임베더 모듈 동적 로드"""
    embedder_map = {
        'openai': 'core.embedders.openai',
        'bge': 'core.embedders.bge',
    }
    
    if embedder_name not in embedder_map:
        raise ValueError(f"지원하지 않는 임베더: {embedder_name}. 지원: openai, bge")
    
    module_name = embedder_map[embedder_name]
    
    try:
        import importlib
        return importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"임베더 모듈 로드 실패: {module_name} ({e})")

def process_single_file(
    input_file: str,
    output_file: str,
    tokenizer_name: str = 'jieba',
    embedder_name: str = 'bge',  # 기본값을 bge로 변경
    use_semantic: bool = True,
    min_tokens: int = 1,
    max_tokens: int = 10,
    parallel: bool = True,  # 병렬 처리 기본값 True로 변경
    openai_model: str = "text-embedding-3-large",
    openai_api_key: str = None,
    progress_callback=None,
    stop_flag=None,
    **kwargs
) -> bool:
    """단일 파일 처리 (병렬 옵션 지원)"""
    import time
    start_time = time.time()  # ⏱️ 처리 시작 시간 기록
    print(f"🚀 파일 처리 시작: {input_file}")
    print(f"📊 설정:")
    print(f"   토크나이저: {tokenizer_name}")
    print(f"   임베더: {embedder_name}")
    print(f"   의미 매칭: {use_semantic}")
    print(f"   병렬 처리: {parallel} (기본값: True)")
    print(f"   토큰 범위: {min_tokens}-{max_tokens}")
    try:
        if parallel:
            print("⚡ 병렬 처리 모드로 실행합니다.")
            from core.processor import process_file
            # 병렬 처리 함수 호출 - embedder_name 전달
            results_df = process_file(
                input_file,
                use_semantic=use_semantic,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                save_results=True,
                output_file=output_file,
                openai_model=openai_model,
                openai_api_key=openai_api_key,
                embedder_name=embedder_name
            )
            if results_df is not None:
                print(f"🎉 병렬 처리 완료! 결과: {len(results_df)}개 구")
                return True
            else:
                print(f"❌ 병렬 처리 실패")
                return False
        # 항상 동적 모듈 로딩 경로 사용
        print("✅ 동적 모듈 로딩...")
        tokenizer_module = get_tokenizer_module(tokenizer_name)
        embedder_module = get_embedder_module(embedder_name)
        print(f"✅ 모듈 로드 완료")
        from core.processor import process_file_with_modules
        results = process_file_with_modules(
            input_file, output_file,
            tokenizer_module, embedder_module,
            embedder_name,  # 추가!
            use_semantic, min_tokens, max_tokens,
            openai_model=openai_model,
            openai_api_key=openai_api_key
        )

        end_time = time.time()  # ⏱️ 처리 종료 시간 기록

        if results is not None:
            print(f"🎉 처리 완료!")
            print(f"⏱️  처리 시간: {end_time - start_time:.2f}초")
            print(f"📊 처리 결과: {len(results)}개 문장")
            print(f"📁 출력 파일: {output_file}")

            # 기본 형식 저장
            output_file_basic = output_file

            # 구 단위 형식 저장
            output_file_phrase = output_file.replace('.xlsx', '_phrase.xlsx').replace('.csv', '_phrase.csv').replace('.txt', '_phrase.txt')

            from sa.io_utils import save_phrase_format_results
            save_phrase_format_results(results, output_file_phrase)

            print(f"📁 기본 출력: {output_file_basic}")
            print(f"📁 구 단위 출력: {output_file_phrase}")

            return True
        else:
            print(f"❌ 처리 실패")
            return False

    except Exception as e:
        print(f"💥 처리 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_dependencies():
    """의존성 및 환경 점검
    - 필수 패키지, torch, transformers, sentence-transformers, mecab-python3 등
    - 주요 경로/환경변수
    """
    missing = []
    try:
        import pandas
    except ImportError:
        missing.append("pandas")
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    try:
        import torch
    except ImportError:
        missing.append("torch")
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    try:
        import sentence_transformers
    except ImportError:
        missing.append("sentence_transformers")
    try:
        import mecab
    except ImportError:
        print("\u26a0\ufe0f mecab-python3가 설치되어 있지 않습니다. mecab 관련 기능은 동작하지 않습니다.")
    if missing:
        print(f"\u274c 필수 패키지 누락: {', '.join(missing)}")
        print("설치 명령: pip install " + " ".join(missing))
        return False
    return True

def main():
    """메인 함수"""
    if not check_dependencies():
        print("\u274c 의존성/환경 점검 실패. 필수 패키지 또는 경로를 확인하세요.")
        return
    parser = argparse.ArgumentParser(
        description='SA 정렬 시스템 - 문장 단위 토큰 정렬',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 처리
  python main.py input.xlsx output.xlsx
  
  # 토크나이저/임베더 지정
  python main.py input.xlsx output.xlsx --tokenizer soy --embedder openai
  
  # 병렬 처리
  python main.py input.xlsx output.xlsx --parallel
  
  # 상세 설정
  python main.py input.xlsx output.xlsx --tokenizer jieba --embedder bge --min-tokens 2 --max-tokens 15 --no-semantic
  
지원 토크나이저: jieba, soy, kkma
지원 임베더: openai, bge
        """
    )
    
    # 필수 인자
    parser.add_argument('input_file', help='입력 Excel 파일 (문장식별자, 원문, 번역문 컬럼)')
    parser.add_argument('output_file', help='출력 Excel 파일')
    
    # 선택 인자
    parser.add_argument('--tokenizer', '-t', default='jieba', 
                       choices=['jieba', 'mecab'],
                       help='토크나이저 선택 (jieba: 원문, mecab: 번역문, 기본: jieba)')
    
    parser.add_argument('--embedder', '-e', default='bge',
                       choices=['openai', 'bge'], 
                       help='임베더 선택 (기본: bge)')
    
    parser.add_argument('--parallel', '-p', action='store_true', default=True,
                       help='병렬 처리 활성화 (기본: 활성화)')
    
    parser.add_argument('--no-semantic', action='store_true',
                       help='의미 기반 매칭 비활성화')
    
    parser.add_argument('--min-tokens', type=int, default=1,
                       help='최소 토큰 수 (기본: 1)')
    
    parser.add_argument('--max-tokens', type=int, default=10,
                       help='최대 토큰 수 (기본: 10)')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='상세 로그 출력')
    
    parser.add_argument('--openai-model', default="text-embedding-3-large", help='OpenAI 임베딩 모델명')
    parser.add_argument('--openai-api-key', default=None, help='OpenAI API 키')
    parser.add_argument('--save-phrase', action='store_true', default=True, help='구 단위 결과도 저장')

    args = parser.parse_args()
    
    # 로깅 설정
    setup_logging(args.verbose)
    
    # 파일 존재 확인
    if not os.path.exists(args.input_file):
        print(f"❌ 입력 파일을 찾을 수 없습니다: {args.input_file}")
        sys.exit(1)
    
    # 처리 실행
    success = process_single_file(
        input_file=args.input_file,
        output_file=args.output_file,
        tokenizer_name=args.tokenizer,
        embedder_name=args.embedder,
        use_semantic=not args.no_semantic,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        parallel=args.parallel,
        openai_model=args.openai_model,
        openai_api_key=args.openai_api_key
        # progress_callback, stop_flag 인자 제거
    )
    
    if success:
        print("\n🎉 처리 성공!")
        sys.exit(0)
    else:
        print("\n❌ 처리 실패!")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SA: Sentence Aligner")
    parser.add_argument('input_file', help='입력 파일 (Excel)')
    parser.add_argument('output_file', help='출력 파일 (Excel)')
    parser.add_argument('--tokenizer', default='jieba', help='원문 토크나이저 (기본: jieba)')
    parser.add_argument('--embedder', default='bge', help='임베더 (기본: bge)')
    parser.add_argument('--parallel', '-p', action='store_true', default=True, help='병렬 처리 활성화 (기본: 활성화)')
    parser.add_argument('--no-semantic', action='store_true', help='의미 기반 매칭 비활성화')
    parser.add_argument('--min-tokens', type=int, default=1, help='최소 토큰 수 (기본: 1)')
    parser.add_argument('--max-tokens', type=int, default=10, help='최대 토큰 수 (기본: 10)')
    parser.add_argument('--verbose', '-v', action='store_true', help='상세 로그 출력')
    parser.add_argument('--openai-model', default="text-embedding-3-large", help='OpenAI 임베딩 모델명')
    parser.add_argument('--openai-api-key', default=None, help='OpenAI API 키')
    parser.add_argument('--save-phrase', action='store_true', default=True, help='구 단위 결과도 저장')
    args = parser.parse_args()
    # 로깅 설정
    setup_logging(args.verbose)
    
    # 파일 존재 확인
    if not os.path.exists(args.input_file):
        print(f"❌ 입력 파일을 찾을 수 없습니다: {args.input_file}")
        sys.exit(1)
    
    # 처리 실행
    success = process_single_file(
        input_file=args.input_file,
        output_file=args.output_file,
        tokenizer_name=args.tokenizer,
        embedder_name=args.embedder,
        use_semantic=not args.no_semantic,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        parallel=args.parallel,
        openai_model=args.openai_model,
        openai_api_key=args.openai_api_key
        # progress_callback, stop_flag 인자 제거
    )
    
    if success:
        print("\n🎉 처리 성공!")
        sys.exit(0)
    else:
        print("\n❌ 처리 실패!")
        sys.exit(1)