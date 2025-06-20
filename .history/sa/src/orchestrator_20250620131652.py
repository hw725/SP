# src/orchestrator.py
"""오케스트레이터"""
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import concurrent.futures
from tqdm import tqdm

from .config import Config
from .file_io import load_data, save_results
from .components import get_tokenizer, get_embedder
from .pipeline import process_single_row

logger = logging.getLogger(__name__)

def run_processing(config: Config):
    """메인 처리 실행 함수"""
    
    # 1. 데이터 로드
    logger.info(f"파일 로드 중: {config.input_path}")
    df = load_data(config.input_path)
    logger.info(f"로드된 데이터 크기: {df.shape}")
    logger.info(f"컬럼들: {list(df.columns)}")
    
    # 2. 컴포넌트 초기화
    source_tokenizer = get_tokenizer(
        config.source_tokenizer_type,
        min_tokens=config.min_tokens
    )
    
    target_tokenizer = get_tokenizer(
        config.target_tokenizer_type,
        min_tokens=config.min_tokens
    )
    
    embedder = get_embedder(
        config.embedder_type,
        cache_dir=config.cache_dir,
        use_fp16=config.use_fp16
    )
    
    logger.info("컴포넌트 초기화 완료:")
    logger.info(f"  - 원문 토크나이저: {config.source_tokenizer_type}")
    logger.info(f"  - 번역문 토크나이저: {config.target_tokenizer_type}")
    logger.info(f"  - 임베더: {config.embedder_type}")
    
    # 3. 처리 실행
    if config.use_parallel:
        logger.info("병렬 처리 시작")
        results = _parallel_process(df, source_tokenizer, target_tokenizer, embedder, config)
    else:
        logger.info("순차 처리 시작")
        results = _sequential_process(df, source_tokenizer, target_tokenizer, embedder, config)
    
    # 4. 결과 DataFrame 생성
    results_df = pd.DataFrame(results)
    original_df = df.copy()
    
    # 원본 컬럼들 유지
    for col in ['aligned_source', 'aligned_target', 'processing_info']:
        if col in results_df.columns:
            original_df[col] = results_df[col]
    
    logger.info(f"처리 완료: {len(results)}개 행")
    
    # 5. 결과 저장
    logger.info(f"결과 저장 완료: {config.output_path}")
    save_results(original_df, config.output_path)
    logger.info(f"출력 컬럼들: {list(original_df.columns)}")
    logger.info(f"성공적으로 처리된 행: {len([r for r in results if r.get('processing_info', {}).get('status') == 'success'])}/{len(results)}")

def _sequential_process(df: pd.DataFrame, source_tokenizer, target_tokenizer, embedder, config: Config) -> List[Dict[str, Any]]:
    """순차 처리"""
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="행 처리 중"):
        result = process_single_row(
            row.to_dict(),
            source_tokenizer,
            target_tokenizer,
            embedder
        )
        results.append(result)
        
        if config.verbose and idx < 3:  # 처음 3개 행만 상세 로그
            logger.debug(f"행 {idx} 처리 결과: {result}")
    
    return results

def _parallel_process(df: pd.DataFrame, source_tokenizer, target_tokenizer, embedder, config: Config) -> List[Dict[str, Any]]:
    """병렬 처리"""
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.num_workers) as executor:
        # 청크 단위로 작업 분할
        chunks = [df[i:i+config.chunk_size] for i in range(0, len(df), config.chunk_size)]
        
        futures = []
        for chunk_idx, chunk in enumerate(chunks):
            future = executor.submit(_process_chunk, chunk, source_tokenizer, target_tokenizer, embedder, chunk_idx)
            futures.append(future)
        
        # 결과 수집
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="청크 처리 중"):
            try:
                chunk_results = future.result()
                results.extend(chunk_results)
            except Exception as e:
                logger.error(f"청크 처리 실패: {e}")
    
    return results

def _process_chunk(chunk: pd.DataFrame, source_tokenizer, target_tokenizer, embedder, chunk_idx: int) -> List[Dict[str, Any]]:
    """청크 처리"""
    results = []
    
    for idx, row in chunk.iterrows():
        try:
            result = process_single_row(
                row.to_dict(),
                source_tokenizer,
                target_tokenizer,
                embedder
            )
            results.append(result)
        except Exception as e:
            logger.error(f"청크 {chunk_idx}, 행 {idx} 처리 실패: {e}")
            results.append({
                'aligned_source': str(row.get('원문', '')),
                'aligned_target': str(row.get('번역문', '')),
                'processing_info': {'error': str(e)}
            })
    
    return results