import logging
import numpy as np
import hashlib
import os
from typing import List, Optional, Dict  # Dict 추가
from pathlib import Path
import torch

from .interfaces import BaseEmbedder

logger = logging.getLogger(__name__)

class CachedEmbedder(BaseEmbedder):
    """캐시 기능이 있는 임베더"""
    def __init__(self, cache_dir: Optional[str] = None, **kwargs):
        # super().__init__(**kwargs)  # ← 제거
        self.cache_dir = cache_dir or os.getenv("EMBED_CACHE_DIR", "./.cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_enabled = True  # cache 사용 여부

    def _get_cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _load_from_cache(self, key: str) -> Optional[np.ndarray]:
        cache_file = Path(self.cache_dir) / f"{key}.npy"
        if cache_file.exists():
            try:
                return np.load(cache_file)
            except Exception as e:
                logger.warning(f"Cache load failed ({cache_file}): {e}")
        return None

    def _save_to_cache(self, key: str, emb: np.ndarray):
        cache_file = Path(self.cache_dir) / f"{key}.npy"
        try:
            np.save(cache_file, emb)
        except Exception as e:
            logger.warning(f"Cache save failed ({cache_file}): {e}")

    def embed(self, texts: List[str]) -> np.ndarray:
        """기본 캐시 로직: cache_enabled=True인 경우 시도"""
        embeddings = []
        for text in texts:
            key = self._get_cache_key(text)
            emb = None
            if self.cache_enabled:
                emb = self._load_from_cache(key)
            if emb is None:
                # 실제 임베딩 (서브클래스 구현 호출)
                emb = self._encode(text)
                if self.cache_enabled:
                    self._save_to_cache(key, emb)
            embeddings.append(emb)
        return np.vstack(embeddings)

    def _encode(self, text: str) -> np.ndarray:
        """서브클래스에서 override"""
        raise NotImplementedError

class SentenceTransformerEmbedder(CachedEmbedder):
    """SentenceTransformer 임베더"""
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 
                 cache_dir: Optional[str] = None, device: Optional[str] = None, **kwargs):
        super().__init__(cache_dir=cache_dir, **kwargs)
        self.model_name = model_name
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"SentenceTransformer 모델 로드 완료: {self.model_name}")
        except ImportError:
            logger.error("sentence-transformers 패키지가 설치되지 않았습니다")
            raise
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            raise
    
    def _encode(self, text: str) -> np.ndarray:
        # 실제 임베딩 구현
        pass

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])

        all_embeddings_map: Dict[int, np.ndarray] = {}
        texts_to_encode_indices: List[int] = []
        texts_to_encode_values: List[str] = []

        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            cached_embedding = self._load_from_cache(cache_key)
            if cached_embedding is not None:
                all_embeddings_map[i] = cached_embedding
            else:
                texts_to_encode_indices.append(i)
                texts_to_encode_values.append(text)
        
        if texts_to_encode_values:
            logger.debug(f"{self.__class__.__name__}: {len(texts_to_encode_values)}개 텍스트 인코딩, {len(texts) - len(texts_to_encode_values)}개 캐시 로드.")
            try:
                new_embeddings_array = self.model.encode(texts_to_encode_values, convert_to_numpy=True)
                for i, new_embedding in enumerate(new_embeddings_array):
                    original_idx = texts_to_encode_indices[i]
                    text_content = texts_to_encode_values[i]
                    current_cache_key = self._get_cache_key(text_content)
                    all_embeddings_map[original_idx] = new_embedding
                    self._save_to_cache(current_cache_key, new_embedding)
            except Exception as e:
                logger.error(f"임베딩 계산 실패: {e}")
                raise
        else:
            logger.debug(f"{self.__class__.__name__}: 모든 {len(texts)}개 텍스트 캐시 로드.")
        
        final_embeddings_list = [all_embeddings_map[i] for i in range(len(texts))]
        if not final_embeddings_list: return np.array([])
        return np.stack(final_embeddings_list)

class OpenAIEmbedder(CachedEmbedder):
    """OpenAI API 임베더"""
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-large", # text-embedding-ada-002
                 cache_dir: Optional[str] = None, **kwargs):
        super().__init__(cache_dir=cache_dir, **kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API 키가 필요합니다. 인자로 전달하거나 OPENAI_API_KEY 환경 변수를 설정하세요.")
        self.model = model
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            logger.error("openai 패키지가 설치되지 않았습니다. `pip install openai`")
            raise
    
    def _encode(self, text: str) -> np.ndarray:
        # 실제 임베딩 구현
        pass

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts: return np.array([])

        all_embeddings_map: Dict[int, np.ndarray] = {}
        texts_to_encode_indices: List[int] = []
        texts_to_encode_values: List[str] = []

        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            cached_embedding = self._load_from_cache(cache_key)
            if cached_embedding is not None:
                all_embeddings_map[i] = cached_embedding
            else:
                texts_to_encode_indices.append(i)
                texts_to_encode_values.append(text)
        
        if texts_to_encode_values:
            logger.debug(f"{self.__class__.__name__}: {len(texts_to_encode_values)}개 텍스트 인코딩, {len(texts) - len(texts_to_encode_values)}개 캐시 로드.")
            try:
                # OpenAI API는 배치 처리를 지원하므로, texts_to_encode_values를 한 번에 보냄
                response = self.client.embeddings.create(input=texts_to_encode_values, model=self.model)
                new_embeddings_data = response.data
                if len(new_embeddings_data) != len(texts_to_encode_values):
                    raise ValueError("OpenAI API 응답의 임베딩 수가 요청한 텍스트 수와 다릅니다.")

                for i, data_item in enumerate(new_embeddings_data):
                    original_idx = texts_to_encode_indices[i]
                    text_content = texts_to_encode_values[i] # API 응답 순서와 요청 순서가 같다고 가정
                    current_cache_key = self._get_cache_key(text_content)
                    embedding = np.array(data_item.embedding)
                    all_embeddings_map[original_idx] = embedding
                    self._save_to_cache(current_cache_key, embedding)
            except Exception as e:
                logger.error(f"OpenAI 임베딩 실패: {e}")
                raise
        else:
            logger.debug(f"{self.__class__.__name__}: 모든 {len(texts)}개 텍스트 캐시 로드.")

        final_embeddings_list = [all_embeddings_map[i] for i in range(len(texts))]
        if not final_embeddings_list: return np.array([])
        return np.stack(final_embeddings_list)

class CohereEmbedder(CachedEmbedder):
    """Cohere API 임베더"""
    def __init__(self, api_key: Optional[str] = None, model: str = "embed-multilingual-v2.0", 
                 cache_dir: Optional[str] = None, **kwargs):
        super().__init__(cache_dir=cache_dir, **kwargs)
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("Cohere API 키가 필요합니다. 인자로 전달하거나 COHERE_API_KEY 환경 변수를 설정하세요.")
        self.model = model
        try:
            import cohere
            self.client = cohere.Client(self.api_key)
        except ImportError:
            logger.error("cohere 패키지가 설치되지 않았습니다. `pip install cohere`")
            raise
    
    def _encode(self, text: str) -> np.ndarray:
        pass

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts: return np.array([])

        all_embeddings_map: Dict[int, np.ndarray] = {}
        texts_to_encode_indices: List[int] = []
        texts_to_encode_values: List[str] = []

        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            cached_embedding = self._load_from_cache(cache_key)
            if cached_embedding is not None:
                all_embeddings_map[i] = cached_embedding
            else:
                texts_to_encode_indices.append(i)
                texts_to_encode_values.append(text)
        
        if texts_to_encode_values:
            logger.debug(f"{self.__class__.__name__}: {len(texts_to_encode_values)}개 텍스트 인코딩, {len(texts) - len(texts_to_encode_values)}개 캐시 로드.")
            try:
                response = self.client.embed(texts=texts_to_encode_values, model=self.model)
                new_embeddings_array = np.array(response.embeddings)
                if new_embeddings_array.shape[0] != len(texts_to_encode_values):
                     raise ValueError("Cohere API 응답의 임베딩 수가 요청한 텍스트 수와 다릅니다.")

                for i, new_embedding in enumerate(new_embeddings_array):
                    original_idx = texts_to_encode_indices[i]
                    text_content = texts_to_encode_values[i]
                    current_cache_key = self._get_cache_key(text_content)
                    all_embeddings_map[original_idx] = new_embedding
                    self._save_to_cache(current_cache_key, new_embedding)
            except Exception as e:
                logger.error(f"Cohere 임베딩 실패: {e}")
                raise
        else:
            logger.debug(f"{self.__class__.__name__}: 모든 {len(texts)}개 텍스트 캐시 로드.")

        final_embeddings_list = [all_embeddings_map[i] for i in range(len(texts))]
        if not final_embeddings_list: return np.array([])
        return np.stack(final_embeddings_list)

class BGEM3Embedder(CachedEmbedder):
    """BGEM3 임베더"""
    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool = True, cache_dir: Optional[str] = None, **kwargs):
        super().__init__(cache_dir=cache_dir, **kwargs)
        try:
            from FlagEmbedding import BGEM3FlagModel as FlagModel 
        except ImportError:
            logger.error("FlagEmbedding 라이브러리를 찾을 수 없습니다. `pip install FlagEmbedding`으로 설치해주세요.")
            raise
        
        self.model = FlagModel(model_name, use_fp16=use_fp16) 
        logger.info(f"BGEM3Embedder 초기화 완료: 모델={model_name}, FP16 사용={use_fp16}, 캐시 디렉토리='{self.cache_dir if self.cache_enabled else '비활성화됨'}'")

    def _normalize_l2(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 정규화 수행"""
        if not isinstance(embeddings, np.ndarray):
            logger.error(f"정규화 대상이 NumPy 배열이 아닙니다. 타입: {type(embeddings)}")
            # 혹은 여기서 예외를 발생시킬 수도 있습니다.
            # raise TypeError("정규화 대상은 NumPy 배열이어야 합니다.")
            return np.array([]) # 빈 배열 또는 적절한 오류 처리

        if embeddings.ndim == 1:
            norm = np.linalg.norm(embeddings)
            if norm == 0:
                return embeddings
            return embeddings / norm
        elif embeddings.ndim == 2:
            norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
            return np.divide(embeddings, norm, out=np.zeros_like(embeddings), where=norm!=0)
        else:
            logger.warning(f"지원하지 않는 임베딩 차원입니다: {embeddings.ndim}. 정규화 없이 반환합니다.")
            return embeddings # 또는 예외 발생

    def _encode(self, text: str) -> np.ndarray:
        pass

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])

        all_embeddings_map: Dict[int, np.ndarray] = {}
        texts_to_encode_indices: List[int] = []
        texts_to_encode_values: List[str] = []

        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            embedding = self._load_from_cache(cache_key)
            if embedding is not None:
                all_embeddings_map[i] = embedding
            else:
                texts_to_encode_indices.append(i)
                texts_to_encode_values.append(text)

        if texts_to_encode_values:
            logger.debug(f"BGEM3: {len(texts_to_encode_values)}개 텍스트 인코딩 수행, {len(texts) - len(texts_to_encode_values)}개는 캐시에서 로드.")
            try:
                # FlagModel.encode 호출
                # BAAI/bge-m3 모델은 다양한 타입의 임베딩을 반환할 수 있으므로,
                # 명시적으로 dense 벡터를 요청하거나, 반환된 딕셔너리에서 추출해야 합니다.
                # encode_kwargs를 사용하여 dense 벡터만 요청하는 것이 더 안전할 수 있습니다.
                # 예: encode_kwargs = {'return_dense': True, 'return_sparse': False, 'return_colbert_vecs': False}
                # output = self.model.encode(texts_to_encode_values, **encode_kwargs)
                
                output = self.model.encode(texts_to_encode_values) # 이전과 동일하게 호출

                if isinstance(output, dict):
                    # BGE-M3의 경우 dense vector는 'dense_vecs' 키에 저장될 가능성이 높음
                    if 'dense_vecs' in output:
                        new_embeddings_unnormalized = output['dense_vecs']
                        logger.debug("BGEM3 encode 결과가 dict이며, 'dense_vecs' 키에서 임베딩 추출.")
                    else:
                        logger.error(f"BGEM3 encode 결과가 dict이지만 'dense_vecs' 키를 찾을 수 없습니다. 사용 가능한 키: {output.keys()}")
                        # 이 경우, 빈 배열을 반환하거나 예외를 발생시켜야 합니다.
                        # 여기서는 빈 배열을 만들고 아래에서 처리되도록 합니다.
                        new_embeddings_unnormalized = np.array([]) 
                elif isinstance(output, np.ndarray):
                    new_embeddings_unnormalized = output
                    logger.debug("BGEM3 encode 결과가 NumPy 배열입니다.")
                else:
                    logger.error(f"BGEM3 encode 결과가 예상치 못한 타입입니다: {type(output)}")
                    new_embeddings_unnormalized = np.array([])

                if not isinstance(new_embeddings_unnormalized, np.ndarray) or new_embeddings_unnormalized.size == 0:
                    logger.warning("임베딩 추출 실패 또는 빈 임베딩 배열. 후속 처리 중단 가능성.")
                    # 이 경우, 각 텍스트에 대해 빈 임베딩을 저장하거나 오류를 전파해야 합니다.
                    # 현재 로직에서는 빈 new_embeddings가 되어 루프를 돌지 않게 됩니다.
                else:
                    new_embeddings = self._normalize_l2(new_embeddings_unnormalized)

                    if new_embeddings.size > 0 and new_embeddings.shape[0] == len(texts_to_encode_values):
                        for i, embedding in enumerate(new_embeddings):
                            original_idx = texts_to_encode_indices[i]
                            text_content = texts_to_encode_values[i]
                            current_cache_key = self._get_cache_key(text_content)
                            self._save_to_cache(current_cache_key, embedding)
                            all_embeddings_map[original_idx] = embedding
                    elif new_embeddings.size == 0:
                         logger.warning("정규화 후 임베딩이 비어있습니다.")
                    else:
                        logger.warning(f"정규화된 임베딩 수({new_embeddings.shape[0]})와 요청 텍스트 수({len(texts_to_encode_values)})가 불일치합니다.")
            except Exception as e:
                logger.error(f"BGEM3 임베딩 실패: {e}")
                raise
        else:
            logger.debug(f"{self.__class__.__name__}: 모든 {len(texts)}개 텍스트 캐시 로드.")

        final_embeddings_list = [all_embeddings_map[i] for i in range(len(texts))]
        if not final_embeddings_list: return np.array([])
        return np.stack(final_embeddings_list)

def calculate_matching_score(...):
    # 함수 내에서 device 재정의 또는 전역 참조 명시
    current_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')