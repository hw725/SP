# main.py
import argparse
import logging
import os
from src.config import Config
from src.orchestrator import run_processing

def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('alignment.log', mode='w', encoding='utf-8')
        ]
    )

def main():
    parser = argparse.ArgumentParser(description="텍스트 정렬 파이프라인 (Prototype02 통합)")
    
    # 기본 인자들
    parser.add_argument("input", help="입력 파일 경로")
    parser.add_argument("output", help="출력 파일 경로")
    
    parser.add_argument("--source-tokenizer", 
                       default=Config.DEFAULT_SOURCE_TOKENIZER,
                       help=f"원문 토크나이저 타입 (기본값: {Config.DEFAULT_SOURCE_TOKENIZER}) - 실제로는 Prototype02 로직 사용")
    
    parser.add_argument("--target-tokenizer", 
                       default=Config.DEFAULT_TARGET_TOKENIZER,
                       help=f"번역문 토크나이저 타입 (기본값: {Config.DEFAULT_TARGET_TOKENIZER}) - 실제로는 Prototype02 로직 사용")
    
    parser.add_argument("--embedder", 
                       default=Config.DEFAULT_EMBEDDER_TYPE,
                       help=f"임베더 타입 (기본값: {Config.DEFAULT_EMBEDDER_TYPE})")
    
    parser.add_argument("--parallel", action="store_true", help="병렬 처리 사용")
    parser.add_argument("--workers", type=int, default=Config.DEFAULT_NUM_WORKERS,
                       help=f"워커 프로세스 수 (기본값: {Config.DEFAULT_NUM_WORKERS})")
    parser.add_argument("--chunk-size", type=int, default=Config.DEFAULT_CHUNK_SIZE,
                       help=f"청크 크기 (기본값: {Config.DEFAULT_CHUNK_SIZE})")
    parser.add_argument("--verbose", action="store_true", help="상세 로깅")
    
    args = parser.parse_args()
    
    # 로깅 설정
    setup_logging(args.verbose)
    
    # 입출력 경로 검증
    if not os.path.exists(args.input):
        logging.error(f"입력 파일을 찾을 수 없습니다: {args.input}")
        return
    
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logging.info(f"출력 디렉토리 생성: {output_dir}")
        except Exception as e:
            logging.error(f"출력 디렉토리 생성 실패: {e}")
            return
    
    # Config 생성
    config = Config(
        input_path=args.input,
        output_path=args.output,
        source_tokenizer_type=args.source_tokenizer,
        target_tokenizer_type=args.target_tokenizer,
        embedder_type=args.embedder,
        use_parallel=args.parallel,
        num_workers=args.workers,
        chunk_size=args.chunk_size,
        verbose=args.verbose
    )
    
    # 설정 로깅
    logging.info("=== 텍스트 정렬 파이프라인 시작 (Prototype02 통합) ===")
    logging.info(f"입력: {config.input_path}")
    logging.info(f"출력: {config.output_path}")
    logging.info(f"알고리즘: Prototype02 고품질 로직")
    logging.info(f"임베더: {config.embedder_type}")
    logging.info(f"병렬 처리: {config.use_parallel}")
    logging.info(f"워커 수: {config.num_workers}")
    logging.info(f"청크 크기: {config.chunk_size}")
    
    try:
        # 처리 실행
        run_processing(config)
        logging.info("=== 텍스트 정렬 파이프라인 완료 ===")
    except Exception as e:
        logging.error(f"파이프라인 실행 중 오류 발생: {e}", exc_info=True)

if __name__ == "__main__":
    main()