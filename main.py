"""메인 실행 모듈"""

import argparse
import logging
import sys
import os
from typing import Optional

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_logging(verbose: bool = False):
    """로깅 설정"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )

def validate_file_paths(input_path: str, output_path: str) -> bool:
    """파일 경로 검증"""
    if not os.path.exists(input_path):
        logger.error(f"입력 파일이 존재하지 않습니다: {input_path}")
        return False
    
    if not input_path.lower().endswith(('.xlsx', '.xls')):
        logger.error(f"지원되지 않는 파일 형식입니다: {input_path}")
        return False
    
    # 출력 디렉토리 확인
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"출력 디렉토리 생성: {output_dir}")
        except Exception as e:
            logger.error(f"출력 디렉토리 생성 실패: {e}")
            return False
    
    return True

def process_with_options(
    input_path: str,
    output_path: str,
    parallel: bool = False,
    workers: int = 4,
    chunk_size: int = 50,
    gpu_memory_fraction: float = 0.8,
    verbose: bool = False
):
    """옵션에 따른 처리 실행"""
    try:
        if parallel:
            logger.info(f"병렬 처리 시작 (워커: {workers}, 청크 크기: {chunk_size})")
            from io_manager import process_file_parallel
            
            process_file_parallel(
                input_path=input_path,
                output_path=output_path,
                num_workers=workers,
                chunk_size=chunk_size,
                gpu_memory_fraction=gpu_memory_fraction,
                verbose=verbose
            )
        else:
            logger.info("단일 프로세스 처리 시작")
            from io_manager import process_file
            
            process_file(
                input_path=input_path,
                output_path=output_path,
                verbose=verbose
            )
        
        logger.info(f"처리 완료: {output_path}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise

def main(
    input_path: Optional[str] = None,
    output_path: Optional[str] = None,
    parallel: bool = False,
    workers: int = 4,
    chunk_size: int = 50,
    gpu_memory_fraction: float = 0.8,
    verbose: bool = False
):
    """메인 함수"""
    try:
        # 로깅 설정
        setup_logging(verbose)
        
        # 명령행 인자 처리 (함수 인자가 None인 경우)
        if input_path is None or output_path is None:
            parser = argparse.ArgumentParser(description="한문-한국어 구 단위 정렬 시스템")
            parser.add_argument("input_path", help="입력 Excel 파일 경로")
            parser.add_argument("output_path", help="출력 Excel 파일 경로")
            parser.add_argument("--parallel", action="store_true", help="병렬 처리 사용")
            parser.add_argument("--workers", type=int, default=4, help="워커 프로세스 수 (기본값: 4)")
            parser.add_argument("--chunk-size", type=int, default=50, help="청크 크기 (기본값: 50)")
            parser.add_argument("--gpu-memory", type=float, default=0.8, help="GPU 메모리 사용 비율 (기본값: 0.8)")
            parser.add_argument("--verbose", action="store_true", help="상세 출력")
            
            args = parser.parse_args()
            
            input_path = args.input_path
            output_path = args.output_path
            parallel = args.parallel
            workers = args.workers
            chunk_size = args.chunk_size
            gpu_memory_fraction = args.gpu_memory
            verbose = args.verbose
        
        # 파일 경로 검증
        if not validate_file_paths(input_path, output_path):
            sys.exit(1)
        
        logger.info(f"입력 파일: {input_path}")
        logger.info(f"출력 파일: {output_path}")
        logger.info(f"병렬 처리: {'사용' if parallel else '미사용'}")
        
        if parallel:
            logger.info(f"워커 수: {workers}")
            logger.info(f"청크 크기: {chunk_size}")
            logger.info(f"GPU 메모리 비율: {gpu_memory_fraction}")
        
        # 처리 실행
        process_with_options(
            input_path=input_path,
            output_path=output_path,
            parallel=parallel,
            workers=workers,
            chunk_size=chunk_size,
            gpu_memory_fraction=gpu_memory_fraction,
            verbose=verbose
        )
        
        logger.info("프로그램이 성공적으로 완료되었습니다.")
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 프로그램이 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Critical error occurred during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()