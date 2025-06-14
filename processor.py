from .io_utils import load_excel_file, save_alignment_results  # ← 추가
from .punctuation import normalize_sentences                  # ← 추가
from .aligner import align_by_embedding          

def process_with_options(
    input_path: str, 
    output_path: str, 
    use_parallel: bool = False,
    num_workers: int = None,
    chunk_size: int = 20,
    gpu_strategy: str = "single",
    verbose: bool = False
) -> None:
    """
    옵션에 따라 병렬 또는 비병렬 처리를 선택하는 통합 인터페이스
    
    Args:
        input_path: 입력 파일 경로
        output_path: 출력 파일 경로
        use_parallel: 병렬 처리 사용 여부
        num_workers: 병렬 워커 수 (None=자동)
        chunk_size: 데이터 청크 크기
        gpu_strategy: GPU 활용 전략
            - "single": 첫 번째 워커만 GPU 사용 (안전)
            - "shared": 모든 워커가 GPU 공유 (고성능 GPU 필요)
            - "multi": 여러 GPU에 분산 (다중 GPU 환경)
            - "none": GPU 사용 안 함 (CPU만 사용)
        verbose: 상세 로깅 여부
    """
    import time
    import logging
    import os
    
    # 로거 확인
    logger = logging.getLogger(__name__)
    
    # 시작 시간
    start_time = time.time()
    
    # 파일 경로 검증
    if not os.path.exists(input_path):
        logger.error(f"입력 파일을 찾을 수 없습니다: {input_path}")
        return
        
    # 출력 디렉토리 존재 확인
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"출력 디렉토리 생성: {output_dir}")
        except Exception as e:
            logger.error(f"출력 디렉토리 생성 실패: {e}")
            return
    
    try:
        # GPU 상태 확인
        gpu_available = False
        gpu_count = 0
        gpu_info = []
        
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                gpu_count = torch.cuda.device_count()
                for i in range(gpu_count):
                    try:
                        gpu_name = torch.cuda.get_device_name(i)
                        total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
                        gpu_info.append(f"GPU {i}: {gpu_name} ({total_memory:.2f} GB)")
                    except:
                        gpu_info.append(f"GPU {i}: Information unavailable")
                
                logger.info(f"Available GPUs: {gpu_count}")
                for info in gpu_info:
                    logger.info(info)
            else:
                logger.info("No GPU available, using CPU only")
                if gpu_strategy != "none":
                    logger.warning("GPU strategy requested but no GPU available, falling back to CPU")
                    gpu_strategy = "none"
        except ImportError:
            logger.warning("torch module not available, using CPU only")
            gpu_strategy = "none"
        
        # 처리 방식 선택
        if use_parallel:
            # 병렬 처리 설정 로깅
            worker_info = "auto" if num_workers is None else str(num_workers)
            logger.info(f"Using parallel processing with {worker_info} workers, chunk_size={chunk_size}, gpu_strategy={gpu_strategy}")
            
            # 병렬 처리 실행
            process_file_parallel(
                input_path=input_path,
                output_path=output_path,
                num_workers=num_workers,
                chunk_size=chunk_size,
                gpu_strategy=gpu_strategy
            )
        else:
            # 비병렬 처리 로깅
            logger.info("Using sequential processing")
            
            # 기존 process_file 함수가 있는지 확인
            if 'process_file' in globals():
                # 기존 함수 호출
                process_file(
                    input_path=input_path,
                    output_path=output_path,
                    batch_size=128,
                    verbose=verbose
                )
            else:
                logger.error("Sequential processing function (process_file) not found")
                return
        
        # 처리 시간 기록
        elapsed_time = time.time() - start_time
        logger.info(f"Processing completed in {elapsed_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def main(input_file, output_file, verbose=False, use_parallel=False, num_workers=None, chunk_size=20, gpu_strategy="single"):
    """
    Execute the pipeline with optional parallel processing.
    
    Args:
        input_file: Path to input Excel file
        output_file: Path to output Excel file
        verbose: Enable verbose logging
        use_parallel: Enable parallel processing
        num_workers: Number of worker processes (None = auto)
        chunk_size: Size of data chunks for parallel processing
        gpu_strategy: GPU strategy ("single", "shared", "multi", or "none")
    """
    if not os.path.isfile(input_file):
        logger.error(f"Input file does not exist: {input_file}")
        return
    if not input_file.lower().endswith(('.xls', '.xlsx')):
        logger.error("Input file must have .xls/.xlsx extension.")
        return
    if not output_file.lower().endswith('.xlsx'):
        logger.error("Output file must have .xlsx extension.")
        return

    if verbose:
        logging.getLogger().setLevel(logging.INFO)
        logger.info("Verbose mode activated: INFO level logging enabled.")
    
    # Log parallel processing options if enabled
    if use_parallel:
        worker_info = "auto" if num_workers is None else num_workers
        logger.info(f"Parallel processing enabled: workers={worker_info}, chunk_size={chunk_size}, gpu_strategy={gpu_strategy}")

    try:
        # Use the unified interface function instead of direct process_file call
        process_with_options(
            input_path=input_file,
            output_path=output_file,
            use_parallel=use_parallel,
            num_workers=num_workers,
            chunk_size=chunk_size,
            gpu_strategy=gpu_strategy,
            verbose=verbose
        )
        logger.info(f"Processing completed: {output_file}")
    except Exception as e:
        logger.error(f"Critical error occurred during pipeline execution: {e}", exc_info=True)
