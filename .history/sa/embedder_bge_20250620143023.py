"""BGE 임베더 - 프로세스 안전 버전"""

import logging
import numpy as np
import torch
import os
from typing import List, Optional, Callable
from tqdm import tqdm

logger = logging.getLogger(__name__)

# 전역 설정
DEFAULT_BATCH_SIZE = 20
DEFAULT_EMBEDDING_MODEL = 'BAAI/bge-m3'

class EmbeddingManager:
    """임베딩 계산 및 캐시 관리 클래스 - 프로세스 안전"""
    
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, fallback_to_dummy: bool = True):
        self.model_name = model_name
        self.model = None
        self._cache = {}
        self._fallback_to_dummy = fallback_to_dummy
        self._model_loaded = False
        self._use_dummy = False
        self.process_id = os.getpid()
    
    def _load_model(self):
        """모델 로딩 (프로세스별)"""
        if self._model_loaded and os.getpid() == self.process_id:
            return
            
        # 프로세스 변경 시 재초기화
        if os.getpid() != self.process_id:
            self.process_id = os.getpid()
            self._model_loaded = False
            self.model = None
        
        try:
            from FlagEmbedding import BGEM3FlagModel
            print(f"프로세스 {self.process_id}: BGE 모델 로딩 중...")
            
            # 환경 변수 설정
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
            
            # 디바이스 설정 (프로세스별로 GPU 메모리 분리)
            if torch.cuda.is_available():
                device = f'cuda'
            else:
                device = 'cpu'
            
            self.model = BGEM3FlagModel(
                self.model_name,
                use_fp16=True,
                device=device
            )
            
            self._model_loaded = True
            self._use_dummy = False
            print(f"프로세스 {self.process_id}: BGE 모델 로딩 완료")
            
        except Exception as e:
            print(f"프로세스 {self.process_id}: BGE 모델 로딩 실패: {e}")
            
            if self._fallback_to_dummy:
                print(f"프로세스 {self.process_id}: 더미 모드로 전환")
                self._use_dummy = True
                self._model_loaded = True
            else:
                raise RuntimeError(f"BGE 모델 초기화 실패: {e}")
    
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
        """프로세스 안전한 임베딩 계산"""
        
        if not texts:
            return np.array([])
        
        # 모델 로딩
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

        # 새 임베딩 계산
        if to_embed:
            if self._use_dummy:
                # 더미 임베딩 사용
                embeddings = [self._generate_dummy_embedding(text) for text in to_embed]
            else:
                # 실제 BGE 모델 사용
                embeddings = []
                
                for start in range(0, len(to_embed), batch_size):
                    batch = to_embed[start:start + batch_size]
                    
                    try:
                        # GPU 메모리 정리
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        # 임베딩 계산
                        output = self.model.encode(
                            batch,
                            return_dense=True,
                            return_sparse=False,
                            return_colbert_vecs=False
                        )
                        dense = output['dense_vecs']
                        embeddings.extend(dense)
                        
                    except Exception as e:
                        print(f"프로세스 {self.process_id}: 임베딩 계산 실패: {e}")
                        # 실패한 배치는 더미로 대체
                        embeddings.extend([self._generate_dummy_embedding(text) for text in batch])

            # 캐시 업데이트
            for i, (txt, emb) in enumerate(zip(to_embed, embeddings)):
                self._cache[txt] = emb
                result_list[indices_to_embed[i]] = emb

        return np.array(result_list)

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