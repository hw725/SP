# src/config.py
import os
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class Config:
    """단순화된 설정 클래스 (Prototype02 통합)"""
    # 필수 인자들
    input_path: str
    output_path: str
    
    # 클래스 변수로 기본값 정의
    DEFAULT_SOURCE_TOKENIZER = "prototype02"  # 실제로는 내장 로직 사용
    DEFAULT_TARGET_TOKENIZER = "prototype02"  # 실제로는 내장 로직 사용
    DEFAULT_EMBEDDER_TYPE = "bge-m3"
    DEFAULT_NUM_WORKERS = 4
    DEFAULT_CHUNK_SIZE = 100
    
    # 실제 필드들
    source_tokenizer_type: str = DEFAULT_SOURCE_TOKENIZER
    target_tokenizer_type: str = DEFAULT_TARGET_TOKENIZER
    embedder_type: str = DEFAULT_EMBEDDER_TYPE
    use_parallel: bool = False
    num_workers: int = DEFAULT_NUM_WORKERS
    chunk_size: int = DEFAULT_CHUNK_SIZE
    
    # 컴포넌트 설정
    source_tokenizer_config: Dict[str, Any] = field(default_factory=dict)
    target_tokenizer_config: Dict[str, Any] = field(default_factory=dict)
    embedder_config: Dict[str, Any] = field(default_factory=lambda: {
        "use_fp16": True,
    })
    
    # 기타
    verbose: bool = False
    default_cache_root: Optional[str] = ".cache/embeddings"

    def __post_init__(self):
        """초기화 후 검증"""
        # API 키 확인 및 설정
        final_embedder_config = dict(self.embedder_config)

        if self.embedder_type == "openai":
            api_key = final_embedder_config.get("api_key") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI 임베더를 사용하려면 API 키가 필요합니다.")
            final_embedder_config['api_key'] = api_key
        
        elif self.embedder_type == "cohere":
            api_key = final_embedder_config.get("api_key") or os.getenv("COHERE_API_KEY")
            if not api_key:
                raise ValueError("Cohere 임베더를 사용하려면 API 키가 필요합니다.")
            final_embedder_config['api_key'] = api_key

        object.__setattr__(self, 'embedder_config', final_embedder_config)
        
        logger.info("Prototype02 통합 알고리즘 사용")