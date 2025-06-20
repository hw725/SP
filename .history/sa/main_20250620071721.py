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
    
    # 로깅 레벨 조정
    if not verbose:
        logging.getLogger('src.text_alignment').setLevel(logging.WARNING)
        logging.getLogger('soynlp').setLevel(logging.WARNING)
        
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('alignment.log', mode='w', encoding='utf-8')
        ]
    )

def main():
    parser = argparse.ArgumentParser(description="텍스트 정렬 파이프라인 (Prototype02 완전 통합)")
    
    parser.add_argument("input", help="입력 파일 경로")
    parser.add_argument("output", help="출력 파일 경로")
    parser.add_argument("--embedder", default="bge-m3", help="임베더 타입 (기본값: bge-m3)")
    parser.add_argument("--parallel", action="store_true", help="병렬 처리 사용")
    parser.add_argument("--workers", type=int, default=4, help="워커 프로세스 수")
    parser.add_argument("--chunk-size", type=int, default=50, help="청크 크기")
    parser.add_argument("--verbose", action="store_true", help="상세 로깅")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
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
    
    config = Config(
        input_path=args.input,
        output_path=args.output,
        source_tokenizer_type="prototype02",
        target_tokenizer_type="prototype02", 
        embedder_type=args.embedder,
        use_parallel=args.parallel,
        num_workers=args.workers,
        chunk_size=args.chunk_size,
        verbose=args.verbose
    )
    
    logging.info("=== Prototype02 완전 통합 파이프라인 시작 ===")
    logging.info(f"입력: {config.input_path}")
    logging.info(f"출력: {config.output_path}")
    logging.info(f"임베더: {config.embedder_type}")
    logging.info(f"병렬 처리: {config.use_parallel}")
    
    try:
        run_processing(config)
        logging.info("=== 파이프라인 완료 ===")
    except Exception as e:
        logging.error(f"파이프라인 실행 중 오류 발생: {e}", exc_info=True)

if __name__ == "__main__":
    main()