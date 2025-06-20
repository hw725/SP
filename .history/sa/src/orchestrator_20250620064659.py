# src/orchestrator.py
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any
from tqdm import tqdm

from .config import Config
from .pipeline import process_single_row
from .components import get_embedder  # *** get_tokenizer 제거 ***
from .file_io import load_data, save_data

logger = logging.getLogger(__name__)

def process_chunk(chunk_data: List[Dict[str, Any]], config: Config) -> List[Dict[str, Any]]:
    """청크 처리 - 토크나이저 생성 제거"""
    # *** 토크나이저는 생성하지 않음 (Prototype02 로직이 내장되어 있음) ***
    worker_embedder = get_embedder(config.embedder_type, **config.embedder_config)
    
    processed_rows = []
    for row_data in chunk_data:
        try:
            processed_result_dict = process_single_row(
                row_data, 
                None,  # *** source_tokenizer = None ***
                None,  # *** target_tokenizer = None ***
                worker_embedder
            )
            
            merged_row = {**row_data, **processed_result_dict} 
            processed_rows.append(merged_row)
        except Exception as e:
            logger.error(f"Row 처리 실패: {e}", exc_info=True)
            error_row = {
                **row_data, 
                'processing_info': {'error': str(e), 'status': 'error'}, 
                'aligned_source': row_data.get('원문', ''), 
                'aligned_target': row_data.get('번역문', '')
            }
            processed_rows.append(error_row)
    return processed_rows

def run_processing(config: Config):
    logger.info("데이터 로드 시작...")
    try:
        data = load_data(config.input_path)
    except Exception as e:
        logger.error(f"데이터 로드 중 치명적 에러: {e}", exc_info=True)
        return

    # 원본 순서 보존
    for idx, row in enumerate(data):
        row['__row_order'] = idx

    logger.info(f"총 {len(data)}개 행 로드됨")

    if not data:
        logger.warning("처리할 데이터가 없습니다.")
        return

    processed_data: List[Dict[str, Any]] = [] 
    
    if config.use_parallel and config.num_workers > 0 and len(data) > config.chunk_size:
        logger.info(f"병렬 처리 모드: {config.num_workers}개 워커, 청크 크기: {config.chunk_size}")
        chunks = [data[i:i + config.chunk_size] for i in range(0, len(data), config.chunk_size)]
        
        with ProcessPoolExecutor(max_workers=config.num_workers) as executor:
            future_to_chunk = {executor.submit(process_chunk, chunk, config): chunk for chunk in chunks}
            with tqdm(total=len(chunks), desc="청크 처리 중") as pbar:
                for future in as_completed(future_to_chunk):
                    try:
                        chunk_results = future.result()
                        processed_data.extend(chunk_results)
                    except Exception as e: 
                        logger.error(f"청크 결과 처리 중 에러: {e}", exc_info=True)
                        failed_chunk_data = future_to_chunk[future]
                        for row_data in failed_chunk_data: 
                            error_row = {
                                **row_data, 
                                'processing_info': {'error': f"Chunk processing failed: {str(e)}", 'status': 'error'},
                                'aligned_source': row_data.get('원문', ''), 
                                'aligned_target': row_data.get('번역문', '')
                            }
                            processed_data.append(error_row)
                    finally:
                        pbar.update(1)
    else:
        logger.info("순차 처리 모드")
        # *** 토크나이저 생성 완전 제거 ***
        embedder = get_embedder(config.embedder_type, **config.embedder_config) 
        
        with tqdm(total=len(data), desc="행 처리 중") as pbar:
            for row_data in data:
                try:
                    processed_result_dict = process_single_row(
                        row_data,
                        None,  # *** source_tokenizer = None ***
                        None,  # *** target_tokenizer = None ***
                        embedder
                    )
                    
                    merged_row = {**row_data, **processed_result_dict}
                    processed_data.append(merged_row)
                except Exception as e:
                    logger.error(f"Row 처리 실패: {e}", exc_info=True)
                    error_row = {
                        **row_data, 
                        'processing_info': {'error': str(e), 'status': 'error'},
                        'aligned_source': row_data.get('원문', ''), 
                        'aligned_target': row_data.get('번역문', '')
                    }
                    processed_data.append(error_row)
                finally:
                    pbar.update(1)
    
    # 원본 순서 복원
    processed_data.sort(key=lambda x: x.get('__row_order', 0))
    
    logger.info(f"처리 완료: {len(processed_data)}개 결과")
    if processed_data:
        logger.info("결과 저장 중...")
        try:
            save_data(processed_data, config.output_path) 
            logger.info(f"결과가 {config.output_path}에 저장됨")
        except Exception as e:
            logger.error(f"결과 저장 중 에러: {e}", exc_info=True)
    else:
        logger.warning("저장할 처리 결과가 없습니다.")