# main.py
import argparse
import logging
import os
# import multiprocessing # freeze_support 사용 시 필요
from src.config import Config
from src.orchestrator import run_processing
# from src.components import ALIGNER_REGISTRY, EMBEDDER_REGISTRY, TOKENIZER_REGISTRY # 직접 사용 안 함

# logger 정의 (루트 로거 사용 또는 특정 이름 지정)
# logging.getLogger()는 루트 로거를 가져오며, basicConfig에 의해 설정됨.
# 특정 모듈 로거를 사용하려면 logger = logging.getLogger(__name__) 사용.
# 여기서는 basicConfig가 루트 로거를 설정하므로, 이후 logging.info 등으로 사용 가능.

def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    # 기존 핸들러 제거 (특히 Jupyter Notebook 등에서 중복 로깅 방지)
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('alignment.log', mode='w', encoding='utf-8') # 매번 새로 쓰기
        ]
    )

def main():
    parser = argparse.ArgumentParser(description="텍스트 정렬 파이프라인")
    
    # 기본 인자들
    parser.add_argument("input", help="입력 파일 경로")
    parser.add_argument("output", help="출력 파일 경로")
    
    # Config 클래스 변수에서 기본값 가져오기
    parser.add_argument("--source-tokenizer", 
                       default=Config.DEFAULT_SOURCE_TOKENIZER,
                       help=f"원문 토크나이저 타입 (기본값: {Config.DEFAULT_SOURCE_TOKENIZER})")
    
    parser.add_argument("--target-tokenizer", 
                       default=Config.DEFAULT_TARGET_TOKENIZER,
                       help=f"번역문 토크나이저 타입 (기본값: {Config.DEFAULT_TARGET_TOKENIZER})")
    
    parser.add_argument("--embedder", 
                       default=Config.DEFAULT_EMBEDDER_TYPE,
                       help=f"임베더 타입 (기본값: {Config.DEFAULT_EMBEDDER_TYPE})")
    
    parser.add_argument("--parallel", action="store_true", help="병렬 처리 사용")
    parser.add_argument("--workers", type=int, default=Config.DEFAULT_NUM_WORKERS,
                       help=f"워커 프로세스 수 (기본값: {Config.DEFAULT_NUM_WORKERS})")
    parser.add_argument("--chunk-size", type=int, default=Config.DEFAULT_CHUNK_SIZE,
                       help=f"청크 크기 (기본값: {Config.DEFAULT_CHUNK_SIZE})")
    
    args = parser.parse_args()
    
    # Config 생성
    config = Config(
        input_path=args.input,
        output_path=args.output,
        source_tokenizer_type=args.source_tokenizer,
        target_tokenizer_type=args.target_tokenizer,
        embedder_type=args.embedder,
        use_parallel=args.parallel,
        num_workers=args.workers,
        chunk_size=args.chunk_size
    )
    
    # 설정 로깅
    logging.info(f"입력: {config.input_path}")
    logging.info(f"출력: {config.output_path}")
    logging.info(f"원문 토크나이저: {config.source_tokenizer_type}")
    logging.info(f"번역문 토크나이저: {config.target_tokenizer_type}")
    logging.info(f"임베더: {config.embedder_type}")
    logging.info(f"병렬 처리: {config.use_parallel}")
    
    # 처리 실행
    from src.orchestrator import run_processing
    run_processing(config)

if __name__ == "__main__":
    # if os.name == 'nt': # Windows에서 멀티프로세싱 사용 시
    #     multiprocessing.freeze_support()
    main()