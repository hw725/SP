"""병렬 처리 프로세서"""
import logging
import pandas as pd
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from .components import BaseTokenizer, BaseEmbedder
from .pipeline import process_single_row

logger = logging.getLogger(__name__)

def process_chunk(chunk_data: List[Dict[str, Any]], 
                 source_tokenizer_type: str,
                 target_tokenizer_type: str, 
                 embedder_type: str) -> List[Dict[str, Any]]:
    """청크 단위 처리 함수 (별도 프로세스에서 실행)"""
    try:
        # 각 프로세스에서 독립적으로 컴포넌트 초기화
        from .components import get_tokenizer, get_embedder
        
        source_tokenizer = get_tokenizer(source_tokenizer_type)
        target_tokenizer = get_tokenizer(target_tokenizer_type)
        embedder = get_embedder(embedder_type)
        
        results = []
        for row_data in chunk_data:
            try:
                result = process_single_row(
                    row_data,
                    source_tokenizer,
                    target_tokenizer,
                    embedder
                )
                results.append(result)
            except Exception as e:
                logger.error(f"행 처리 실패: {e}")
                results.append({
                    'aligned_source': str(row_data.get('원문', '')),
                    'aligned_target': str(row_data.get('번역문', '')),
                    'processing_info': {'error': str(e)}
                })
        
        return results
        
    except Exception as e:
        logger.error(f"청크 처리 실패: {e}")
        # 실패한 경우 원본 데이터 반환
        return [{
            'aligned_source': str(row.get('원문', '')),
            'aligned_target': str(row.get('번역문', '')),
            'processing_info': {'error': str(e)}
        } for row in chunk_data]

class ParallelProcessor:
    """병렬 처리 클래스"""
    
    def __init__(self, 
                 source_tokenizer: BaseTokenizer,
                 target_tokenizer: BaseTokenizer,
                 embedder: BaseEmbedder,
                 num_workers: int = 4,
                 chunk_size: int = 50):
        self.source_tokenizer_type = getattr(source_tokenizer, '__class__', 'prototype02').__name__.lower()
        self.target_tokenizer_type = getattr(target_tokenizer, '__class__', 'prototype02').__name__.lower()
        self.embedder_type = getattr(embedder, '__class__', 'bge-m3').__name__.lower()
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        
        # 타입 매핑
        type_mapping = {
            'prototype02sourcetokenizer': 'prototype02',
            'hanjahangultokenizer': 'hanja_hangul',
            'mecabtokenizer': 'mecab',
            'jiebatokenizer': 'jieba',
            'bgem3embedder': 'bge-m3',
            'sentencetransformerembedder': 'sentence-transformer',
            'openaiembedder': 'openai',
            'cohereembedder': 'cohere'
        }
        
        self.source_tokenizer_type = type_mapping.get(self.source_tokenizer_type, 'prototype02')
        self.target_tokenizer_type = type_mapping.get(self.target_tokenizer_type, 'prototype02')
        self.embedder_type = type_mapping.get(self.embedder_type, 'bge-m3')
    
    def process_dataframe(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """데이터프레임 병렬 처리"""
        # 데이터를 청크로 분할
        chunks = []
        for i in range(0, len(df), self.chunk_size):
            chunk = df.iloc[i:i+self.chunk_size].to_dict('records')
            chunks.append(chunk)
        
        logger.info(f"데이터를 {len(chunks)}개 청크로 분할 (청크 크기: {self.chunk_size})")
        
        results = []
        
        try:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                # 작업 제출
                future_to_chunk = {
                    executor.submit(
                        process_chunk, 
                        chunk,
                        self.source_tokenizer_type,
                        self.target_tokenizer_type,
                        self.embedder_type
                    ): chunk for chunk in chunks
                }
                
                # 결과 수집
                for future in tqdm(as_completed(future_to_chunk), 
                                 total=len(chunks), 
                                 desc="청크 처리 중"):
                    try:
                        chunk_results = future.result()
                        results.extend(chunk_results)
                    except Exception as e:
                        logger.error(f"청크 처리 실패: {e}")
                        # 실패한 청크의 원본 데이터 복구
                        failed_chunk = future_to_chunk[future]
                        for row in failed_chunk:
                            results.append({
                                'aligned_source': str(row.get('원문', '')),
                                'aligned_target': str(row.get('번역문', '')),
                                'processing_info': {'error': str(e)}
                            })
        
        except Exception as e:
            logger.error(f"병렬 처리 실패: {e}")
            # 순차 처리로 폴백
            logger.info("순차 처리로 대체합니다.")
            from .pipeline import process_single_row
            from .components import get_tokenizer, get_embedder
            
            source_tokenizer = get_tokenizer(self.source_tokenizer_type)
            target_tokenizer = get_tokenizer(self.target_tokenizer_type)
            embedder = get_embedder(self.embedder_type)
            
            for _, row in tqdm(df.iterrows(), total=len(df), desc="순차 처리 중"):
                try:
                    result = process_single_row(
                        row.to_dict(),
                        source_tokenizer,
                        target_tokenizer,
                        embedder
                    )
                    results.append(result)
                except Exception as row_e:
                    logger.error(f"행 처리 실패: {row_e}")
                    results.append({
                        'aligned_source': str(row.get('원문', '')),
                        'aligned_target': str(row.get('번역문', '')),
                        'processing_info': {'error': str(row_e)}
                    })
        
        return results