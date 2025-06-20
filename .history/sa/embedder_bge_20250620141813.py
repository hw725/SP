"""텍스트 임베딩 계산 및 캐시 관리 모듈"""

import logging
import numpy as np
import torch
from typing import List, Optional
from tqdm import tqdm  # tqdm.notebook 대신 tqdm 사용

logger = logging.getLogger(__name__)

# 기본 설정값
DEFAULT_BATCH_SIZE = 20
DEFAULT_EMBEDDING_MODEL = 'BAAI/bge-m3'
DEFAULT_MAX_CACHE_SIZE = 10000

class EmbeddingManager:
    """임베딩 계산 및 캐시 관리 클래스"""
    
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, fallback_to_dummy: bool = True):
        self.model_name = model_name
        self.model = None  # 지연 로딩
        self._cache = {}
        self._model_loaded = False
        self._use_dummy = False
        self._fallback_to_dummy = fallback_to_dummy
    
    def _load_model(self):
        """모델을 지연 로딩"""
        if self._model_loaded:
            return
            
        try:
            from FlagEmbedding import BGEM3FlagModel
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = BGEM3FlagModel(self.model_name, use_fp16=True)
            self._model_loaded = True
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            
            if self._fallback_to_dummy:
                logger.warning("Falling back to dummy embeddings for testing")
                self._use_dummy = True
                self._model_loaded = True
            else:
                raise RuntimeError(f"Could not initialize embedding model: {e}")
    
    def _generate_dummy_embedding(self, text: str) -> np.ndarray:
        """더미 임베딩 생성"""
        seed = hash(text) % (2**31)
        np.random.seed(seed)
        dummy_emb = np.random.randn(1024).astype(np.float32)
        dummy_emb = dummy_emb / (np.linalg.norm(dummy_emb) + 1e-8)
        return dummy_emb
    
    def compute_embeddings_with_cache(
        self,
        texts: List[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
        show_batch_progress: bool = False
    ) -> np.ndarray:
        """향상된 임베딩 캐싱 기능"""
        # 모델 지연 로딩
        if not self._model_loaded:
            self._load_model()
            
        result_list: List[Optional[np.ndarray]] = [None] * len(texts)
        to_embed: List[str] = []
        indices_to_embed: List[int] = []

        # 캐시 확인
        for i, txt in enumerate(texts):
            if txt in self._cache:
                result_list[i] = self._cache[txt]
            else:
                to_embed.append(txt)
                indices_to_embed.append(i)

        # 새 임베딩 계산 필요시
        if to_embed:
            if self._use_dummy:
                # 더미 임베딩 사용
                embeddings = [self._generate_dummy_embedding(text) for text in to_embed]
            else:
                # 실제 모델 사용
                embeddings = []
                it = range(0, len(to_embed), batch_size)
                if show_batch_progress:
                    it = tqdm(it, desc="Embedding batches", ncols=80)
                
                # GPU 메모리 정리
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                for start in it:
                    batch = to_embed[start:start + batch_size]
                    try:
                        output = self.model.encode(
                            batch,
                            return_dense=True,
                            return_sparse=False,
                            return_colbert_vecs=False
                        )
                        dense = output['dense_vecs']
                    except RuntimeError as e:
                        # OOM 오류 시 배치 크기 줄여서 재시도
                        if "out of memory" in str(e) and batch_size > 1:
                            reduced_batch = batch_size // 2
                            logger.warning(f"메모리 부족, 배치 크기 축소: {batch_size} -> {reduced_batch}")
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            
                            return self.compute_embeddings_with_cache(
                                texts, 
                                batch_size=reduced_batch,
                                show_batch_progress=show_batch_progress
                            )
                        else:
                            raise e
                            
                    embeddings.extend(dense)

            # 캐시 업데이트
            for i, (txt, emb) in enumerate(zip(to_embed, embeddings)):
                self._cache[txt] = emb
                result_list[indices_to_embed[i]] = emb

        return np.array(result_list)
    
    def clear_cache(self):
        """캐시 초기화"""
        self._cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def is_loaded(self) -> bool:
        """모델 로드 상태 확인"""
        return self._model_loaded
    
    def is_using_dummy(self) -> bool:
        """더미 모드 사용 여부 확인"""
        return self._use_dummy

# 전역 인스턴스 및 함수들
_embedding_manager = None

def get_embedding_manager() -> EmbeddingManager:
    """임베딩 매니저 싱글톤 인스턴스 반환"""
    global _embedding_manager
    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager(fallback_to_dummy=True)
    return _embedding_manager

def compute_embeddings_with_cache(texts: List[str], **kwargs) -> np.ndarray:
    """하위 호환성을 위한 함수"""
    manager = get_embedding_manager()
    return manager.compute_embeddings_with_cache(texts, **kwargs)

# 기존 코드와의 호환성을 위한 전역 변수 (느긋한 평가)
class EmbeddingManagerProxy:
    """임베딩 매니저 프록시 클래스"""
    def __getattr__(self, name):
        manager = get_embedding_manager()
        return getattr(manager, name)

embedding_manager = EmbeddingManagerProxy()