# src/orchestrator.py
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any
from tqdm import tqdm

from .config import Config
from .pipeline import process_single_row
from .components import get_tokenizer, get_embedder, get_text_masker
from .file_io import load_data, save_data

logger = logging.getLogger(__name__)

def _unmask_texts_in_result(
    processed_result_dict: Dict[str, Any], 
    text_masker: Any,
    unmask_type: str
) -> Dict[str, Any]:
    """
    처리 결과 딕셔너리 내의 aligned_source와 aligned_target을 언마스킹합니다.
    masking_map은 processing_info에서 가져옵니다.
    """
    if not processed_result_dict or processed_result_dict.get('processing_info', {}).get('status') != 'success':
        return processed_result_dict # 변경 없이 반환

    processing_info = processed_result_dict['processing_info']
    masking_map_source = processing_info.get('masking_map_source')
    masking_map_target = processing_info.get('masking_map_target')

    current_aligned_source = processed_result_dict.get('aligned_source')
    current_aligned_target = processed_result_dict.get('aligned_target')

    try:
        if masking_map_source and isinstance(current_aligned_source, str):
            processed_result_dict['aligned_source'] = text_masker.restore(
                current_aligned_source,
                masking_map_source
            )
        elif isinstance(current_aligned_source, str) and "__PAREN_" in current_aligned_source : # 맵이 없는데 플레이스홀더가 있는 경우 경고
             logger.warning(f"Source text for ID {processed_result_dict.get('id', 'N/A')} contains placeholders but no masking_map_source found.")


        if masking_map_target and isinstance(current_aligned_target, str):
            processed_result_dict['aligned_target'] = text_masker.restore(
                current_aligned_target,
                masking_map_target
            )
        elif isinstance(current_aligned_target, str) and "__PAREN_" in current_aligned_target: # 맵이 없는데 플레이스홀더가 있는 경우 경고
            logger.warning(f"Target text for ID {processed_result_dict.get('id', 'N/A')} contains placeholders but no masking_map_target found.")

    except Exception as e:
        row_id = processed_result_dict.get('id', 'N/A') # 원본 row_data에서 id를 가져올 수 있다면 사용
        logger.error(f"언마스킹 중 오류 발생 (ID: {row_id}): {e}", exc_info=True)
        # 오류 발생 시 원본 마스킹된 텍스트를 유지할 수 있도록 별도 처리는 하지 않음.
        # 필요하다면 processing_info에 오류 기록 추가 가능
        if 'processing_info' in processed_result_dict:
            processed_result_dict['processing_info']['unmask_error'] = str(e)

    return processed_result_dict

def process_chunk(chunk_data: List[Dict[str, Any]], config: Config) -> List[Dict[str, Any]]:
    source_tokenizer = get_tokenizer(config.source_tokenizer_type, **config.source_tokenizer_config)
    target_tokenizer = get_tokenizer(config.target_tokenizer_type, **config.target_tokenizer_config)
    worker_embedder = get_embedder(config.embedder_type, **config.embedder_config)
    # worker_aligner = get_aligner(config.aligner_type, **config.aligner_config)  # 이 줄은 이미 주석 처리되어 있음 (좋음)
    text_masker = get_text_masker(**getattr(config, 'text_masker_config', {}))
    
    processed_rows = []
    for row_data in chunk_data:
        try:
            processed_result_dict = process_single_row(
                row_data, source_tokenizer, target_tokenizer, 
                worker_embedder  # aligner 인자 없음 (좋음)
            )
            
            # 언마스킹 수행
            unmasked_result_dict = _unmask_texts_in_result(
                processed_result_dict, 
                text_masker, 
                getattr(config, 'output_unmask_type', 'original')  # 기본값 'original' 제공
            )
            
            merged_row = {**row_data, **unmasked_result_dict} 
            processed_rows.append(merged_row)
        except Exception as e:
            logger.error(f"Row 처리 실패 (in process_chunk, id: {row_data.get('id', 'N/A')}): {e}", exc_info=True)
            error_row = {**row_data, 'processing_info': {'error': str(e), 'status': 'error'}, 
                         'aligned_source': row_data.get('원문', ''), 
                         'aligned_target': row_data.get('번역문', '')}
            processed_rows.append(error_row)
    return processed_rows

def run_processing(config: Config):
    logger.info("데이터 로드 시작...")
    try:
        data = load_data(config.input_path)
    except Exception as e:
        logger.error(f"데이터 로드 중 치명적 에러: {e}", exc_info=True)
        return

    # ← 추가: 원본 순서를 보존할 수 있도록 메타 정보 삽입
    for idx, row in enumerate(data):
        row['__row_order'] = idx

    logger.info(f"총 {len(data)}개 행 로드됨")

    if not data:
        logger.warning("처리할 데이터가 없습니다.")
        return

    processed_data: List[Dict[str, Any]] = [] 
    
    if config.use_parallel and config.num_workers > 0 and len(data) > config.chunk_size :
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
                            error_row = {**row_data, 'processing_info': {'error': f"Chunk processing failed: {str(e)}", 'status': 'error'},
                                         'aligned_source': row_data.get('원문', ''), 
                                         'aligned_target': row_data.get('번역문', '')}
                            processed_data.append(error_row)
                    finally:
                        pbar.update(1)
    else:
        logger.info("순차 처리 모드")
        source_tokenizer = get_tokenizer(config.source_tokenizer_type, **config.source_tokenizer_config)
        target_tokenizer = get_tokenizer(config.target_tokenizer_type, **config.target_tokenizer_config)
        embedder = get_embedder(config.embedder_type, **config.embedder_config) 
        # aligner = get_aligner(config.aligner_type, **config.aligner_config)  # 이 줄은 이미 주석 처리되어 있음 (좋음)
        text_masker = get_text_masker(**getattr(config, 'text_masker_config', {}))
        
        with tqdm(total=len(data), desc="행 처리 중") as pbar:
            for row_data in data:
                try:
                    processed_result_dict = process_single_row(
                        row_data,
                        source_tokenizer,
                        target_tokenizer,
                        embedder
                    )
                    
                    # 언마스킹 수행
                    unmasked_result_dict = _unmask_texts_in_result(
                        processed_result_dict,
                        text_masker,
                        getattr(config, 'output_unmask_type', 'original')
                    )
                    
                    merged_row = {**row_data, **unmasked_result_dict}
                    processed_data.append(merged_row)
                except Exception as e:
                    logger.error(f"Row 처리 실패 (in sequential, id: {row_data.get('id', 'N/A')}): {e}", exc_info=True)
                    error_row = {**row_data, 'processing_info': {'error': str(e), 'status': 'error'},
                                 'aligned_source': row_data.get('원문', ''), 
                                 'aligned_target': row_data.get('번역문', '')}
                    processed_data.append(error_row)
                finally:
                    pbar.update(1)
    
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