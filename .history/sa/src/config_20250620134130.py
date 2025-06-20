# src/config.py
import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class Config:
    """설정 클래스 - 모든 속성 완전 추가"""
    # 필수 인자들
    input_path: str
    output_path: str
    
    # 토크나이저 설정
    source_tokenizer_type: str = "default"  # prototype02-mecab
    target_tokenizer_type: str = "prototype02"
    min_tokens: int = 1
    
    # 임베더 설정
    embedder_type: str = "bge-m3"
    cache_dir: str = "./.cache"
    use_fp16: bool = True
    
    # 처리 설정
    use_parallel: bool = False
    num_workers: int = 4
    chunk_size: int = 50
    
    # 기타
    verbose: bool = False

    def __post_init__(self):
        """설정 검증 및 로깅"""
        
        # 경로 검증
        input_path = Path(self.input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {self.input_path}")
        
        # 출력 디렉토리 생성
        output_path = Path(self.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 캐시 디렉토리 생성
        cache_path = Path(self.cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # min_tokens 검증
        if self.min_tokens < 1:
            logger.warning(f"min_tokens가 1보다 작습니다: {self.min_tokens}, 1로 설정")
            self.min_tokens = 1
        
        # 설정 로깅
        logger.info("Prototype02 통합 알고리즘 사용")
        if self.verbose:
            logger.info(f"상세 설정:")
            logger.info(f"  - 입력: {self.input_path}")
            logger.info(f"  - 출력: {self.output_path}")
            logger.info(f"  - 원문 토크나이저: {self.source_tokenizer_type}")
            logger.info(f"  - 번역문 토크나이저: {self.target_tokenizer_type}")
            logger.info(f"  - 최소 토큰 수: {self.min_tokens}")
            logger.info(f"  - 임베더: {self.embedder_type}")
            logger.info(f"  - 병렬 처리: {self.use_parallel}")
            logger.info(f"  - 워커 수: {self.num_workers}")