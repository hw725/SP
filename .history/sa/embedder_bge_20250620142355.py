"""BGE 임베더 - 병렬 처리 안전 버전"""

import logging
import numpy as np
import torch
from typing import List, Optional, Callable
from tqdm import tqdm
import threading

logger = logging.getLogger(__name__)

# 전역 설정
DEFAULT_BATCH_SIZE = 20
DEFAULT_EMBEDDING_MODEL = 'BAAI/bge-m3'
DEFAULT_MAX_CACHE_SIZE = 10000

# 스레드 로컬 스토리지 (병렬 처리 안전)
_thread_local = threading.local()

class EmbeddingManager:
    """임베딩 계산 및 캐시 관리 클래스 - 병렬 처리 안전"""
    
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, fallback_to_dummy: bool = True):
        self.model_name = model_name
        self._cache = {}
        self._fallback_to_dummy = fallback_to_dummy
        self._lock = threading.Lock()  # 캐시 접근 동기화
    
    def _get_model(self):
        """스레드 로컬 모델 가져오기"""
        if not hasattr(_thread_local, 'model'):
            try:
                from FlagEmbedding import BGEM3FlagModel
                logger.info(f"스레드 {threading.current_thread().name}: BGE 모델 로딩 중...")
                
                # 디바이스 설정
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                
                _thread_local.model = BGEM3FlagModel(
                    self.model_name, 
                    use_fp16=True,
                    device=device
                )
                _thread_local.use_dummy = False
                logger.info(f"스레드 {threading.current_thread().name}: BGE 모델 로딩 완료")
                
            except Exception as e:
                logger.error(f"스레드 {threading.current_thread().name}: BGE 모델 로딩 실패: {e}")
                
                if self._fallback_to_dummy:
                    logger.warning(f"스레드 {threading.current_thread().name}: 더미 모드로 전환")
                    _thread_local.model = None
                    _thread_local.use_dummy = True
                else:
                    raise RuntimeError(f"BGE 모델 초기화 실패: {e}")
        
        return _thread_local.model, getattr(_thread_local, 'use_dummy', False)
    
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
        """병렬 처리 안전한 임베딩 계산"""
        
        if not texts:
            return np.array([])
        
        # 모델 가져오기 (스레드 로컬)
        model, use_dummy = self._get_model()
        
        result_list: List[Optional[np.ndarray]] = [None] * len(texts)
        to_embed: List[str] = []
        indices_to_embed: List[int] = []

        # 캐시 확인 (스레드 안전)
        with self._lock:
            for i, txt in enumerate(texts):
                if txt in self._cache:
                    result_list[i] = self._cache[txt]
                else:
                    to_embed.append(txt)
                    indices_to_embed.append(i)

        # 새 임베딩 계산
        if to_embed:
            if use_dummy:
                # 더미 임베딩 사용
                embeddings = [self._generate_dummy_embedding(text) for text in to_embed]
            else:
                # 실제 BGE 모델 사용
                embeddings = []
                
                # 배치 처리
                for start in range(0, len(to_embed), batch_size):
                    batch = to_embed[start:start + batch_size]
                    
                    try:
                        # GPU 메모리 정리
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        # 임베딩 계산
                        output = model.encode(
                            batch,
                            return_dense=True,
                            return_sparse=False,
                            return_colbert_vecs=False
                        )
                        dense = output['dense_vecs']
                        embeddings.extend(dense)
                        
                    except RuntimeError as e:
                        # OOM 오류 처리
                        if "out of memory" in str(e) and batch_size > 1:
                            logger.warning(f"메모리 부족, 배치 크기 축소: {batch_size} -> {batch_size//2}")
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            
                            # 재귀 호출로 배치 크기 줄여서 재시도
                            return self.compute_embeddings_with_cache(
                                texts, 
                                batch_size=batch_size//2,
                                show_batch_progress=show_batch_progress
                            )
                        else:
                            logger.error(f"임베딩 계산 실패: {e}")
                            # 실패한 배치는 더미로 대체
                            embeddings.extend([self._generate_dummy_embedding(text) for text in batch])

            # 캐시 업데이트 (스레드 안전)
            with self._lock:
                for i, (txt, emb) in enumerate(zip(to_embed, embeddings)):
                    self._cache[txt] = emb
                    result_list[indices_to_embed[i]] = emb

        return np.array(result_list)
    
    def clear_cache(self):
        """캐시 초기화"""
        with self._lock:
            self._cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# 전역 인스턴스
_embedding_manager = EmbeddingManager(fallback_to_dummy=True)

def compute_embeddings_with_cache(texts: List[str], **kwargs) -> np.ndarray:
    """하위 호환성 함수"""
    return _embedding_manager.compute_embeddings_with_cache(texts, **kwargs)

def get_embed_func() -> Callable:
    """임베딩 함수 반환"""
    return compute_embeddings_with_cache

def get_embedding_manager() -> EmbeddingManager:
    """임베딩 매니저 반환"""
    return _embedding_manager

# 프록시 클래스
class EmbeddingManagerProxy:
    def __getattr__(self, name):
        return getattr(_embedding_manager, name)

embedding_manager = EmbeddingManagerProxy()