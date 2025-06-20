# src/config.py
import os
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class Config:
    """Prototype02 통합 시스템 설정"""
    # 필수 인자들
    input_path: str
    output_path: str
    
    # *** 기본값 고정: 원문 jieba, 번역문 mecab ***
    source_tokenizer_type: str = "jieba-mecab"  # 실제로는 원문에 jieba 적용
    target_tokenizer_type: str = "jieba-mecab"   # 번역문에 mecab 적용
    embedder_type: str = "bge-m3"
    
    # 병렬 처리 설정
    use_parallel: bool = False
    num_workers: int = 4
    chunk_size: int = 50
    
    # 기타 설정
    verbose: bool = False
    min_tokens: int = 1
    
    def __post_init__(self):
        logger.info("Prototype02 + 분석기 통합 알고리즘 사용")
        
        # 기본값 검증 및 설정
        if not self.source_tokenizer_type or self.source_tokenizer_type == "default":
            self.source_tokenizer_type = "default"  # jieba-mecab 조합
            
        if not self.target_tokenizer_type or self.target_tokenizer_type == "default":
            self.target_tokenizer_type = "default"  # jieba-mecab 조합
        
        logger.info(f"기본 설정 적용: {self.source_tokenizer_type} (원문), {self.target_tokenizer_type} (번역문)")