"""메인 실행 파일 - parallel 옵션 추가"""

import argparse
import sys
import os
import shutil
from pathlib import Path
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_files(target_tokenizer: str, embedder: str):
    """파일 교체 설정"""
    
    # 토크나이저 파일 매핑
    tokenizer_files = {
        'mecab': 'tokenizer.py',      # 현재 파일 그대로 사용
        'soy': 'tokenizer_soy.py',
        'kkma': 'tokenizer_kkma.py'
    }
    
    # 임베더 파일 매핑  
    embedder_files = {
        'bge': 'embedder_bge.py',
        'openai': 'embedder_openai.py',
        'hf': 'embedder_hf.py'
    }
    
    # 토크나이저 교체
    if target_tokenizer != 'mecab':
        tokenizer_source = tokenizer_files.get(target_tokenizer)
        if tokenizer_source and Path(tokenizer_source).exists():
            shutil.copy2(tokenizer_source, 'tokenizer.py')
            logger.info(f"토크나이저를 {target_tokenizer}로 교체")
        else:
            logger.error(f"토크나이저 파일을 찾을 수 없음: {tokenizer_source}")
            return False
    else:
        logger.info("기본 mecab 토크나이저 사용")
    
    # 임베더 교체
    if embedder != 'bge':
        embedder_source = embedder_files.get(embedder)
        if embedder_source and Path(embedder_source).exists():
            shutil.copy2(embedder_source, 'embedder.py')
            logger.info(f"임베더를 {embedder}로 교체")
        else:
            logger.error(f"임베더 파일을 찾을 수 없음: {embedder_source}")
            return False
    else:
        logger.info("기본 BGE 임베더 사용")
        
    return True

def main():
    parser = argparse.ArgumentParser(description="Prototype02 텍스트 정렬 시스템")
    parser.add_argument("input_path", help="입력 파일 경로 (.xlsx)")
    parser.add_argument("output_path", help="출력 파일 경로 (.xlsx)")
    
    # 번역문 토크나이저 선택
    parser.add_argument("--target-tokenizer", 
                       choices=['mecab', 'soy', 'kkma'], 
                       default='mecab',
                       help="번역문 토크나이저 (기본값: mecab)")
    
    # 임베더 선택
    parser.add_argument("--embedder",
                       choices=['bge', 'openai', 'hf'],
                       default='bge', 
                       help="임베더 선택 (기본값: bge)")
    
    # 추가 옵션들
    parser.add_argument("--verbose", "-v", action="store_true", help="상세 로그")
    parser.add_argument("--parallel", action="store_true", help="병렬 처리 사용")
    parser.add_argument("--workers", type=int, default=4, help="병렬 워커 수 (기본값: 4)")
    parser.add_argument("--batch-size", type=int, default=20, help="임베딩 배치 크기 (기본값: 20)")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 입력 파일 확인
    if not Path(args.input_path).exists():
        logger.error(f"입력 파일을 찾을 수 없습니다: {args.input_path}")
        sys.exit(1)
    
    # 파일 설정
    logger.info("=== 파일 설정 시작 ===")
    logger.info(f"설정: 원문=한자+한글(고정), 번역문={args.target_tokenizer}, 임베더={args.embedder}")
    if args.parallel:
        logger.info(f"병렬 처리: {args.workers}개 워커")
    logger.info(f"배치 크기: {args.batch_size}")
    
    if not setup_files(args.target_tokenizer, args.embedder):
        logger.error("파일 설정 실패")
        sys.exit(1)
    
    # 처리 실행
    logger.info("=== 처리 시작 ===")
    try:
        # 동적 임포트 (파일 교체 후)
        import importlib
        
        # 모듈 재로드
        modules_to_reload = ['io_manager', 'tokenizer', 'embedder', 'aligner', 'punctuation']
        for module_name in modules_to_reload:
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
            
        from io_manager import process_file
        
        logger.info(f"입력: {args.input_path}")
        logger.info(f"출력: {args.output_path}")
        
        # 옵션 전달
        process_file(
            args.input_path, 
            args.output_path,
            parallel=args.parallel,
            workers=args.workers,
            batch_size=args.batch_size
        )
        
        logger.info("=== 처리 완료 ===")
        
    except Exception as e:
        logger.error(f"처리 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()