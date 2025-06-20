"""SA 정렬 시스템 메인 실행기"""

import argparse
import logging
import time
import sys
import os
from typing import Optional

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
    """토크나이저 모듈 동적 로드"""
    tokenizer_map = {
        'jieba': 'sa_tokenizers.jieba_mecab',
        'mecab': 'sa_tokenizers.jieba_mecab', 
        'soy': 'sa_tokenizers.soynlp',
        'kkma': 'sa_tokenizers.kkma'
    }
    
    if tokenizer_name not in tokenizer_map:
        raise ValueError(f"지원하지 않는 토크나이저: {tokenizer_name}")
    
    module_name = tokenizer_map[tokenizer_name]
    
    try:
        import importlib
        module = importlib.import_module(module_name)
        return module
    except ImportError as e:
        raise ImportError(f"토크나이저 모듈 로드 실패 {tokenizer_name}: {e}")

def get_embedder_module(embedder_name: str):
    """임베더 모듈 동적 로드"""
    embedder_map = {
        'sentence_transformer': 'sa_embedders.sentence_transformer',
        'st': 'sa_embedders.sentence_transformer',
        'openai': 'sa_embedders.openai',
        'bge': 'sa_embedders.bge',
        'hf': 'sa_embedders.hf'
    }
    
    if embedder_name not in embedder_map:
        raise ValueError(f"지원하지 않는 임베더: {embedder_name}")
    
    module_name = embedder_map[embedder_name]
    
    try:
        import importlib
        module = importlib.import_module(module_name)
        return module
    except ImportError as e:
        raise ImportError(f"임베더 모듈 로드 실패 {embedder_name}: {e}")

def process_single_file(
    input_file: str,
    output_file: str,
    tokenizer_name: str = 'jieba',
    embedder_name: str = 'st',
    use_semantic: bool = True,
    min_tokens: int = 1,
    max_tokens: int = 10,
    parallel: bool = False,
    **kwargs
) -> bool:
    """단일 파일 처리"""
    
    print(f"🚀 파일 처리 시작: {input_file}")
    print(f"📊 설정:")
    print(f"   토크나이저: {tokenizer_name}")
    print(f"   임베더: {embedder_name}")
    print(f"   의미 매칭: {use_semantic}")
    print(f"   병렬 처리: {parallel}")
    print(f"   토큰 범위: {min_tokens}-{max_tokens}")
    
    try:
        # 🔧 수정: 기본 토크나이저는 동적 로딩 없이 바로 처리
        if tokenizer_name == 'jieba' and embedder_name == 'st':
            print("✅ 기본 모듈 사용 (jieba + sentence_transformer)")
            
            from processor import process_file
            
            start_time = time.time()
            
            results = process_file(
                input_file,
                use_semantic=use_semantic,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                save_results=True,
                output_file=output_file
            )
            
        else:
            print("✅ 동적 모듈 로딩...")
            
            # 동적 모듈 로드
            tokenizer_module = get_tokenizer_module(tokenizer_name)
            embedder_module = get_embedder_module(embedder_name)
            
            print(f"✅ 모듈 로드 완료")
            
            from processor import process_file_with_modules
            
            start_time = time.time()
            
            results = process_file_with_modules(
                input_file, output_file,
                tokenizer_module, embedder_module,
                use_semantic, min_tokens, max_tokens
            )
        
        end_time = time.time()
        
        if results is not None:
            print(f"🎉 처리 완료!")
            print(f"⏱️  처리 시간: {end_time - start_time:.2f}초")
            print(f"📊 처리 결과: {len(results)}개 문장")
            print(f"📁 출력 파일: {output_file}")
            return True
        else:
            print(f"❌ 처리 실패")
            return False
            
    except Exception as e:
        print(f"💥 처리 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 함수"""
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
지원 임베더: st, openai, bge, hf
        """
    )
    
    # 필수 인자
    parser.add_argument('input_file', help='입력 Excel 파일 (문장식별자, 원문, 번역문 컬럼)')
    parser.add_argument('output_file', help='출력 Excel 파일')
    
    # 선택 인자
    parser.add_argument('--tokenizer', '-t', default='jieba', 
                       choices=['jieba', 'mecab', 'soy', 'kkma'],
                       help='토크나이저 선택 (기본: jieba)')
    
    parser.add_argument('--embedder', '-e', default='st',
                       choices=['st', 'sentence_transformer', 'openai', 'bge', 'hf'], 
                       help='임베더 선택 (기본: st)')
    
    parser.add_argument('--parallel', '-p', action='store_true',
                       help='병렬 처리 활성화')
    
    parser.add_argument('--no-semantic', action='store_true',
                       help='의미 기반 매칭 비활성화')
    
    parser.add_argument('--min-tokens', type=int, default=1,
                       help='최소 토큰 수 (기본: 1)')
    
    parser.add_argument('--max-tokens', type=int, default=10,
                       help='최대 토큰 수 (기본: 10)')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='상세 로그 출력')
    
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
        parallel=args.parallel
    )
    
    if success:
        print("\n🎉 처리 성공!")
        sys.exit(0)
    else:
        print("\n❌ 처리 실패!")
        sys.exit(1)

if __name__ == "__main__":
    main()