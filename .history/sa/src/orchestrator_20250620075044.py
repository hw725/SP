# src/orchestrator.py
"""오케스트레이터 - 병렬 처리 안전성 강화"""
import logging
import pandas as pd
from typing import Dict, Any, List
from tqdm import tqdm

from .config import Config
from .components import get_tokenizer, get_embedder
from .pipeline import process_single_row

logger = logging.getLogger(__name__)

def run_processing(config: Config):
    """메인 처리 함수"""
    logger.info(f"파일 로드 중: {config.input_path}")
    
    # 입력 파일 로드
    try:
        if config.input_path.endswith('.xlsx'):
            df = pd.read_excel(config.input_path)
        elif config.input_path.endswith('.csv'):
            df = pd.read_csv(config.input_path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {config.input_path}")
    except Exception as e:
        logger.error(f"파일 로드 실패: {e}")
        raise
    
    logger.info(f"로드된 데이터 크기: {df.shape}")
    logger.info(f"컬럼들: {list(df.columns)}")
    
    # 필수 컬럼 확인
    required_columns = ['원문', '번역문']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"필수 컬럼 누락: {missing_columns}")
    
    # 컴포넌트 초기화
    try:
        source_tokenizer = get_tokenizer(config.source_tokenizer_type)
        target_tokenizer = get_tokenizer(config.target_tokenizer_type)
        embedder = get_embedder(config.embedder_type)
        
        logger.info(f"컴포넌트 초기화 완료:")
        logger.info(f"  - 원문 토크나이저: {config.source_tokenizer_type}")
        logger.info(f"  - 번역문 토크나이저: {config.target_tokenizer_type}")
        logger.info(f"  - 임베더: {config.embedder_type}")
        
    except Exception as e:
        logger.error(f"컴포넌트 초기화 실패: {e}")
        raise
    
    # 데이터 처리
    try:
        if config.use_parallel:
            logger.info(f"병렬 처리 시작 (워커: {config.num_workers}, 청크: {config.chunk_size})")
            try:
                from .parallel_processor import ParallelProcessor
                processor = ParallelProcessor(
                    source_tokenizer=source_tokenizer,
                    target_tokenizer=target_tokenizer,
                    embedder=embedder,
                    num_workers=config.num_workers,
                    chunk_size=config.chunk_size
                )
                results = processor.process_dataframe(df)
            except Exception as parallel_error:
                logger.error(f"병렬 처리 실패: {parallel_error}")
                logger.info("순차 처리로 대체합니다.")
                config.use_parallel = False
        
        if not config.use_parallel:
            logger.info("순차 처리 시작")
            results = []
            
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="행 처리 중"):
                try:
                    result = process_single_row(
                        row.to_dict(),
                        source_tokenizer,
                        target_tokenizer,
                        embedder
                    )
                    results.append(result)
                except Exception as e:
                    logger.error(f"행 {idx} 처리 실패: {e}")
                    # 실패한 경우 원본 데이터 보존
                    results.append({
                        'aligned_source': str(row.get('원문', '')),
                        'aligned_target': str(row.get('번역문', '')),
                        'processing_info': {'error': str(e)}
                    })
        
        logger.info(f"처리 완료: {len(results)}개 행")
        
    except Exception as e:
        logger.error(f"데이터 처리 실패: {e}")
        raise
    
    # 결과 저장
    try:
        # 원본 데이터에 결과 추가
        result_df = df.copy()
        
        # 결과 컬럼 추가
        aligned_sources = []
        aligned_targets = []
        processing_infos = []
        
        for result in results:
            aligned_sources.append(result.get('aligned_source', ''))
            aligned_targets.append(result.get('aligned_target', ''))
            processing_infos.append(str(result.get('processing_info', {})))
        
        # 올바른 컬럼명으로 추가
        result_df['aligned_source'] = aligned_sources
        result_df['aligned_target'] = aligned_targets
        result_df['processing_info'] = processing_infos
        
        # 저장
        if config.output_path.endswith('.xlsx'):
            result_df.to_excel(config.output_path, index=False)
        elif config.output_path.endswith('.csv'):
            result_df.to_csv(config.output_path, index=False)
        else:
            raise ValueError(f"지원하지 않는 출력 형식: {config.output_path}")
        
        logger.info(f"결과 저장 완료: {config.output_path}")
        logger.info(f"출력 파일 크기: {result_df.shape}")
        logger.info(f"출력 컬럼들: {list(result_df.columns)}")
        
        # 성공한 처리 개수 확인
        success_count = sum(1 for result in results if result.get('processing_info', {}).get('status') == 'success')
        logger.info(f"성공적으로 처리된 행: {success_count}/{len(results)}")
        
    except Exception as e:
        logger.error(f"결과 저장 실패: {e}")
        raise