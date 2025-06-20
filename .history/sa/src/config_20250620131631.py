# src/config.py
import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class Config:
    """설정 클래스"""
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
    output_format: str = "phrase_unit"  # "phrase_unit" or "compact"
    cache_dir: str = "./.cache"
    min_tokens: int = 1  # *** 추가된 속성 ***

    def __post_init__(self):
        """설정 검증 및 로깅"""
        
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
            logger.info(f"  - 출력 포맷: {self.output_format}")
            logger.info(f"  - 원문 토크나이저: {self.source_tokenizer_type}")
            logger.info(f"  - 번역문 토크나이저: {self.target_tokenizer_type}")
            logger.info(f"  - 최소 토큰 수: {self.min_tokens}")
            logger.info(f"  - 임베더: {self.embedder_type}")
            logger.info(f"  - 병렬 처리: {self.use_parallel}")
            logger.info(f"  - 워커 수: {self.num_workers}")