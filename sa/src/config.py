# src/config.py
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional # Optional 추가
import logging # logging 추가

logger = logging.getLogger(__name__) # 로거 추가

@dataclass(frozen=True)
class Config:
    """설정 클래스"""
    # 파일 경로
    input_path: str
    output_path: str
    
    # 컴포넌트 타입
    source_tokenizer_type: str = "jieba"  # 원문용 토크나이저
    target_tokenizer_type: str = "mecab"  # 번역문용 토크나이저
    embedder_type: str = "bge-m3"  # 임베더 타입 (기본 BGE-M3)
    aligner_type: str = "strict" # 기본 정렬기 변경 가능
    
    # 컴포넌트 설정
    source_tokenizer_config: Dict[str, Any] = field(default_factory=dict)
    target_tokenizer_config: Dict[str, Any] = field(default_factory=dict)
    embedder_config: Dict[str, Any] = field(default_factory=lambda: {
        "use_fp16": True,  # BGE-M3 기본값
    })
    aligner_config: Dict[str, Any] = field(default_factory=dict)
    
    # 처리 옵션
    use_parallel: bool = False
    num_workers: int = os.cpu_count() or 2 # 기본 워커 수 개선
    chunk_size: int = 50 # 기본 청크 크기 조정
    
    # 기타
    verbose: bool = False
    default_cache_root: Optional[str] = ".cache/embeddings" # 공통 캐시 루트

    def __post_init__(self):
        """초기화 후 검증"""
        # API 키 확인 및 설정 (환경 변수 우선)
        final_embedder_config = dict(self.embedder_config) # 수정 가능한 복사본 생성

        if self.embedder_type == "openai":
            api_key = final_embedder_config.get("api_key") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI 임베더를 사용하려면 API 키가 필요합니다. embedder_config에 'api_key'를 설정하거나 OPENAI_API_KEY 환경 변수를 사용하세요.")
            final_embedder_config['api_key'] = api_key
        
        elif self.embedder_type == "cohere":
            api_key = final_embedder_config.get("api_key") or os.getenv("COHERE_API_KEY")
            if not api_key:
                raise ValueError("Cohere 임베더를 사용하려면 API 키가 필요합니다. embedder_config에 'api_key'를 설정하거나 COHERE_API_KEY 환경 변수를 사용하세요.")
            final_embedder_config['api_key'] = api_key
        
        # embedder_config에 cache_dir가 명시적으로 없고, default_cache_root가 설정되어 있다면 동적 생성
        # 이 로직은 get_embedder에서 처리하도록 이전했으므로, 여기서는 선택 사항.
        # 만약 Config에서 최종 결정하고 싶다면 주석 해제.
        # if 'cache_dir' not in final_embedder_config and self.default_cache_root:
        #     cache_path = os.path.join(self.default_cache_root, self.embedder_type.replace('-', '_'))
        #     final_embedder_config['cache_dir'] = cache_path
        #     logger.info(f"Config: Embedder '{self.embedder_type}'의 캐시 디렉토리 설정: {cache_path}")

        # frozen=True이므로 object.__setattr__을 사용하여 수정된 config 반영
        object.__setattr__(self, 'embedder_config', final_embedder_config)