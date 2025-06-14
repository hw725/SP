"""Excel 파일 읽기/쓰기 및 파일 처리 모듈"""

import os
import logging
import pandas as pd
import torch
import gc
import multiprocessing as mp
from typing import List, Dict, Any, Tuple
from tqdm.notebook import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# 기본 설정값
DEFAULT_CHUNK_SIZE = 50
DEFAULT_MAX_CACHE_SIZE = 10000

def process_file(
    input_path: str, 
    output_path: str, 
    batch_size: int = 128, 
    verbose: bool = False
) -> None:
    """청크 단위로 최적화된 파일 처리 함수"""
    # 지연 임포트로 순환 참조 방지
    from punctuation import mask_brackets, restore_masks
    from tokenizer import split_src_meaning_units, split_tgt_meaning_units
    from aligner import align_src_tgt
    from embedder import compute_embeddings_with_cache, get_embedding_manager
    
    # 임베딩 매니저 인스턴스 가져오기
    embedding_manager = get_embedding_manager()
    
    try:
        df = pd.read_excel(input_path, engine='openpyxl')
    except Exception as e:
        logger.error(f"[IO] Failed to read Excel file: {e}")
        return

    if '원문' not in df.columns or '번역문' not in df.columns:
        logger.error("[IO] Missing '원문' or '번역문' columns.")
        return

    outputs: List[Dict[str, Any]] = []
    total_rows = len(df)
    
    chunk_size = min(DEFAULT_CHUNK_SIZE, total_rows)
    
    for chunk_start in tqdm(range(0, total_rows, chunk_size), desc="Processing chunks"):
        chunk_end = min(chunk_start + chunk_size, total_rows)
        chunk_df = df.iloc[chunk_start:chunk_end]
        
        # 메모리 관리
        if len(embedding_manager._cache) > DEFAULT_MAX_CACHE_SIZE:
            embedding_manager.clear_cache()
        
        for idx, row in enumerate(chunk_df.itertuples(index=False), start=chunk_start+1):
            src_text = str(getattr(row, '원문', '') or '')
            tgt_text = str(getattr(row, '번역문', '') or '')
            
            if verbose:
                print(f"\n[========= ROW {idx} =========]")
                print("Source (input):", src_text)
                print("Target (input):", tgt_text)

            try:
                masked_src, src_masks = mask_brackets(src_text, text_type="source")
                masked_tgt, tgt_masks = mask_brackets(tgt_text, text_type="target")

                src_units = split_src_meaning_units(masked_src)
                tgt_units = split_tgt_meaning_units(
                    masked_src, masked_tgt, use_semantic=True, min_tokens=1
                )

                restored_src_units = [restore_masks(unit, src_masks) for unit in src_units]
                restored_tgt_units = [restore_masks(unit, tgt_masks) for unit in tgt_units]

                aligned_pairs = align_src_tgt(
                    restored_src_units, restored_tgt_units, compute_embeddings_with_cache
                )
                
                if not aligned_pairs:
                    # 정렬 결과가 없으면 원본 텍스트 사용
                    outputs.append({
                        "문장식별자": idx,
                        "구식별자": 1,
                        "원문구": src_text,
                        "번역구": tgt_text,
                    })
                    continue
                
                aligned_src_units, aligned_tgt_units = zip(*aligned_pairs)

                if verbose:
                    print("Alignment result: (source to target comparison)")
                    for src_gu, tgt_gu in zip(aligned_src_units, aligned_tgt_units):
                        print(f"SRC: {src_gu} | TGT: {tgt_gu}")

                for gu_idx, (src_gu, tgt_gu) in enumerate(zip(aligned_src_units, aligned_tgt_units), start=1):
                    outputs.append({
                        "문장식별자": idx,
                        "구식별자": gu_idx,
                        "원문구": src_gu,
                        "번역구": tgt_gu,
                    })

            except Exception as e:
                logger.warning(f"[IO] Failed to process row {idx}: {e}")
                outputs.append({
                    "문장식별자": idx,
                    "구식별자": 1,
                    "원문구": src_text,
                    "번역구": tgt_text,
                })

    try:
        output_df = pd.DataFrame(outputs, columns=["문장식별자", "구식별자", "원문구", "번역구"])
        output_df.to_excel(output_path, index=False, engine='openpyxl')
        if verbose:
            logger.info(f"[IO] Results saved successfully: {output_path}")
    except Exception as e:
        logger.error(f"[IO] Failed to save results: {e}")

def process_chunk(chunk_data, gpu_idx=-1, memory_fraction=0.95):
    """
    하나의 데이터 청크를 처리하는 워커 함수 - GPU 최적화 버전
    
    Args:
        chunk_data: (인덱스, 원문, 번역문) 튜플의 리스트
        gpu_idx: 사용할 GPU 인덱스 (-1: CPU만 사용)
        memory_fraction: GPU 메모리 사용 비율 (0.0 ~ 1.0)
    
    Returns:
        처리된 결과 딕셔너리 리스트
    """
    import os
    import torch
    import logging
    import gc
    from typing import List, Dict, Any
    
    # 로거 초기화
    logger = logging.getLogger(__name__)
    
    # 필수 모듈 및 함수 임포트 검증
    if not _validate_dependencies():
        logger.error("필수 모듈 또는 함수가 누락되었습니다. 원본 텍스트를 그대로 반환합니다.")
        return _create_fallback_results(chunk_data)
    
    # GPU/CPU 환경 설정
    _setup_device_environment(gpu_idx, memory_fraction, logger)
    
    # 모델 초기화
    model, embedding_cache = _initialize_model(gpu_idx, logger)
    if model is None:
        logger.error("모델 로드 실패. 원본 텍스트를 반환합니다.")
        return _create_fallback_results(chunk_data)
    
    # 청크 데이터 처리
    chunk_results = []
    for idx, src_text, tgt_text in chunk_data:
        try:
            result = _process_single_row(
                idx, src_text, tgt_text, 
                embedding_cache, logger
            )
            chunk_results.extend(result)
        except Exception as e:
            logger.error(f"행 {idx} 처리 오류: {e}")
            chunk_results.append(_create_error_result(idx, src_text, tgt_text))
    
    # 리소스 정리
    _cleanup_resources(model, embedding_cache, gpu_idx)
    
    return chunk_results

def process_file_parallel(
    input_path: str,
    output_path: str,
    num_workers: int = 4,
    chunk_size: int = 50,
    gpu_memory_fraction: float = 0.8,
    verbose: bool = False
) -> None:
    """
    병렬 처리를 사용한 파일 처리 함수
    
    Args:
        input_path: 입력 Excel 파일 경로
        output_path: 출력 Excel 파일 경로
        num_workers: 워커 프로세스 수
        chunk_size: 청크 크기
        gpu_memory_fraction: GPU 메모리 사용 비율
        verbose: 상세 출력 여부
    """
    try:
        # 데이터 로드
        df = pd.read_excel(input_path, engine='openpyxl')
        logger.info(f"Loaded {len(df)} rows from {input_path}")
        
        if '원문' not in df.columns or '번역문' not in df.columns:
            logger.error("Missing '원문' or '번역문' columns.")
            return
        
        # 데이터를 청크로 분할
        chunks = []
        for i in range(0, len(df), chunk_size):
            chunk_df = df.iloc[i:i + chunk_size]
            chunk_data = []
            for idx, row in enumerate(chunk_df.itertuples(index=False), start=i+1):
                src_text = str(getattr(row, '원문', '') or '')
                tgt_text = str(getattr(row, '번역문', '') or '')
                chunk_data.append((idx, src_text, tgt_text))
            chunks.append(chunk_data)
        
        logger.info(f"Split data into {len(chunks)} chunks")
        
        # GPU 사용 가능 여부 확인
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        logger.info(f"Available GPUs: {num_gpus}")
        
        # 병렬 처리 실행
        all_results = []
        
        if num_workers == 1:
            # 단일 프로세스 처리
            logger.info("Using single process mode")
            for i, chunk_data in enumerate(tqdm(chunks, desc="Processing chunks")):
                gpu_idx = i % num_gpus if num_gpus > 0 else -1
                chunk_results = process_chunk(
                    chunk_data, 
                    gpu_idx=gpu_idx, 
                    memory_fraction=gpu_memory_fraction
                )
                all_results.extend(chunk_results)
        else:
            # 멀티프로세스 처리
            logger.info(f"Using multiprocess mode with {num_workers} workers")
            
            # 워커 인수 준비
            chunk_args = []
            for i, chunk_data in enumerate(chunks):
                gpu_idx = i % num_gpus if num_gpus > 0 else -1
                chunk_args.append((chunk_data, gpu_idx, gpu_memory_fraction))
            
            # ProcessPoolExecutor 사용
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # 작업 제출
                future_to_chunk = {
                    executor.submit(process_chunk, *args): i 
                    for i, args in enumerate(chunk_args)
                }
                
                # 결과 수집
                with tqdm(total=len(chunks), desc="Processing chunks") as pbar:
                    for future in as_completed(future_to_chunk):
                        chunk_idx = future_to_chunk[future]
                        try:
                            chunk_results = future.result()
                            all_results.extend(chunk_results)
                            pbar.update(1)
                            
                            if verbose:
                                logger.info(f"Completed chunk {chunk_idx}")
                                
                        except Exception as e:
                            logger.error(f"Chunk {chunk_idx} failed: {e}")
                            # 실패한 청크에 대해 폴백 결과 생성
                            chunk_data = chunk_args[chunk_idx][0]
                            fallback_results = _create_fallback_results(chunk_data)
                            all_results.extend(fallback_results)
                            pbar.update(1)
        
        # 결과 정렬 (문장식별자 기준)
        all_results.sort(key=lambda x: (x['문장식별자'], x['구식별자']))
        
        # 결과 저장
        output_df = pd.DataFrame(all_results, columns=["문장식별자", "구식별자", "원문구", "번역구"])
        output_df.to_excel(output_path, index=False, engine='openpyxl')
        
        logger.info(f"Parallel processing completed. Results saved to {output_path}")
        logger.info(f"Total processed results: {len(all_results)}")
        
    except Exception as e:
        logger.error(f"Parallel processing failed: {e}")
        import traceback
        traceback.print_exc()

def _validate_dependencies() -> bool:
    """필수 의존성 검증"""
    try:
        # 지연 임포트로 순환 참조 방지
        from punctuation import mask_brackets, restore_masks
        from tokenizer import split_src_meaning_units, split_tgt_meaning_units
        from aligner import align_src_tgt
        from embedder import compute_embeddings_with_cache
        return True
    except ImportError as e:
        logging.getLogger(__name__).error(f"의존성 임포트 실패: {e}")
        return False

def _create_fallback_results(chunk_data) -> List[Dict[str, Any]]:
    """의존성 실패 시 대체 결과 생성"""
    return [
        {
            "문장식별자": idx,
            "구식별자": 1,
            "원문구": src,
            "번역구": tgt
        }
        for idx, src, tgt in chunk_data
    ]


def _create_error_result(idx: int, src_text: str, tgt_text: str) -> Dict[str, Any]:
    """오류 발생 시 기본 결과 생성"""
    return {
        "문장식별자": idx,
        "구식별자": 1,
        "원문구": src_text,
        "번역구": tgt_text,
    }

def _setup_device_environment(gpu_idx: int, memory_fraction: float, logger):
    """GPU/CPU 환경 설정"""
    if gpu_idx < 0:
        # CPU 모드
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        logger.info("Worker using CPU mode")
    else:
        # GPU 모드
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
        logger.info(f"Worker using GPU {gpu_idx} with memory fraction {memory_fraction}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.set_per_process_memory_fraction(memory_fraction, 0)
            except (AttributeError, RuntimeError) as e:
                logger.warning(f"GPU 메모리 설정 실패: {e}")
                gc.collect()

def _initialize_model(gpu_idx: int, logger):
    """모델 초기화 및 로드"""
    try:
        # 임베딩 매니저 초기화
        from embedder import get_embedding_manager
        embedding_manager = get_embedding_manager()
        embedding_cache = {}
        
        # GPU 설정
        if gpu_idx >= 0 and torch.cuda.is_available():
            try:
                # GPU 모드로 모델 설정 (더미 모드에서는 실제로는 사용되지 않음)
                logger.info("GPU 모드로 모델 초기화 완료")
            except RuntimeError as e:
                logger.error(f"GPU 모드 설정 오류, CPU로 대체: {e}")
        else:
            # CPU 모드
            logger.info("CPU 모드로 모델 초기화 완료")
            
        return embedding_manager, embedding_cache
        
    except Exception as e:
        logger.error(f"모델 로드 오류: {e}")
        return None, None

def _process_single_row(
    idx: int, 
    src_text: str, 
    tgt_text: str, 
    embedding_cache: dict,
    logger
) -> List[Dict[str, Any]]:
    """단일 행 처리"""
    # 지연 임포트
    from punctuation import mask_brackets, restore_masks
    from tokenizer import split_src_meaning_units, split_tgt_meaning_units
    from aligner import align_src_tgt
    from embedder import compute_embeddings_with_cache
    
    # 빈 텍스트 처리
    if not src_text or not tgt_text:
        return [_create_error_result(idx, src_text, tgt_text)]
    
    try:
        # 1. 괄호 마스킹
        masked_src, src_masks = mask_brackets(src_text, text_type="source")
        masked_tgt, tgt_masks = mask_brackets(tgt_text, text_type="target")
        
        # 2. 의미 단위 분할
        src_units = split_src_meaning_units(masked_src)
        tgt_units = split_tgt_meaning_units(
            masked_src, masked_tgt, 
            use_semantic=True, min_tokens=1
        )
        
        # 분할 결과 검증
        if not src_units or not tgt_units:
            logger.warning(f"행 {idx}: 의미 단위 분할 결과가 비어 있습니다.")
            return [_create_error_result(idx, src_text, tgt_text)]
        
        # 3. 마스크 복원
        restored_src_units = [restore_masks(unit, src_masks) for unit in src_units]
        restored_tgt_units = [restore_masks(unit, tgt_masks) for unit in tgt_units]
        
        # 4. 정렬 수행
        aligned_pairs = align_src_tgt(
            restored_src_units, 
            restored_tgt_units, 
            compute_embeddings_with_cache
        )
        
        if not aligned_pairs:
            return [_create_error_result(idx, src_text, tgt_text)]
        
        # 5. 결과 생성
        results = []
        aligned_src_units, aligned_tgt_units = zip(*aligned_pairs)
        
        for gu_idx, (src_gu, tgt_gu) in enumerate(zip(aligned_src_units, aligned_tgt_units), start=1):
            # 빈 결과 필터링
            if src_gu.strip() and tgt_gu.strip():
                results.append({
                    "문장식별자": idx,
                    "구식별자": gu_idx,
                    "원문구": src_gu,
                    "번역구": tgt_gu,
                })
        
        # 결과가 없으면 원본 반환
        if not results:
            results = [_create_error_result(idx, src_text, tgt_text)]
            
        return results
        
    except Exception as e:
        logger.error(f"행 {idx} 상세 처리 오류: {e}")
        return [_create_error_result(idx, src_text, tgt_text)]

def _cleanup_resources(model, embedding_cache: dict, gpu_idx: int):
    """리소스 정리"""
    try:
        # 캐시 정리
        if embedding_cache:
            embedding_cache.clear()
        
        # 모델 정리
        if model is not None:
            del model
        
        # 메모리 정리
        gc.collect()
        
        # GPU 메모리 정리
        if gpu_idx >= 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        # 정리 중 오류는 로깅만 하고 넘어감
        logging.getLogger(__name__).warning(f"리소스 정리 중 오류: {e}")

# 추가: 메모리 효율적인 배치 처리를 위한 헬퍼 함수
def _process_chunk_in_batches(
    chunk_data, 
    batch_size: int = 5,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    대용량 청크를 작은 배치로 나누어 처리하여 메모리 효율성 개선
    
    Args:
        chunk_data: 청크 데이터
        batch_size: 배치 크기
        **kwargs: process_chunk에 전달할 추가 인자
    
    Returns:
        처리 결과 리스트
    """
    all_results = []
    
    for i in range(0, len(chunk_data), batch_size):
        batch = chunk_data[i:i + batch_size]
        batch_results = process_chunk(batch, **kwargs)
        all_results.extend(batch_results)
        
        # 배치 간 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return all_results