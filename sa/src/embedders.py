from typing import List, Optional, Dict, Any
import numpy as np # numpy import 확인
import hashlib
import logging
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class BaseEmbedder:
    """임베더 기본 클래스 (인터페이스 역할)"""
    def __init__(self, **kwargs): # kwargs를 받아 하위 클래스에서 super() 호출 시 문제 없도록
        pass

    def embed(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError("Embedder의 하위 클래스는 embed 메서드를 구현해야 합니다.")

class CachedEmbedder(BaseEmbedder):
    """캐시 기능이 있는 임베더"""
    def __init__(self, cache_dir: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.memory_cache: Dict[str, np.ndarray] = {}
        self.cache_enabled: bool = False # 파일 캐시 활성화 여부

        if self.cache_dir:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self.cache_enabled = True
                logger.info(f"파일 캐시 사용: {self.cache_dir.resolve()}")
            except Exception as e:
                logger.error(f"캐시 디렉토리 생성 실패: {self.cache_dir}. 파일 캐싱 비활성화. 오류: {e}")
                self.cache_dir = None # 생성 실패 시 None으로 설정
        else:
            logger.info("파일 캐시 비활성화됨 (cache_dir가 제공되지 않음).")

    def _get_cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        if cache_key in self.memory_cache:
            logger.debug(f"메모리 캐시에서 로드: {cache_key}")
            return self.memory_cache[cache_key]
        
        if self.cache_enabled and self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}.npy"
            if cache_file.exists():
                try:
                    embedding = np.load(cache_file)
                    self.memory_cache[cache_key] = embedding
                    logger.debug(f"파일 캐시에서 로드: {cache_key}")
                    return embedding
                except Exception as e:
                    logger.warning(f"캐시 파일 로드 실패 ({cache_key}): {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, embedding: np.ndarray):
        self.memory_cache[cache_key] = embedding
        logger.debug(f"메모리 캐시에 저장: {cache_key}")
        
        if self.cache_enabled and self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}.npy"
            try:
                np.save(cache_file, embedding)
                logger.debug(f"파일 캐시에 저장: {cache_key}")
            except Exception as e:
                logger.warning(f"캐시 파일 저장 실패 ({cache_key}): {e}")

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
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-ada-002", 
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
                logger.error(f"BGEM3 임베딩 실패: {e}", exc_info=True) 
                raise 
        else:
            logger.debug(f"BGEM3: 모든 {len(texts)}개 텍스트를 캐시에서 로드했습니다.")
        
        final_embeddings_list = [all_embeddings_map.get(i) for i in range(len(texts))] # .get()으로 변경하여 KeyEror 방지
        
        # None이 포함될 수 있으므로 필터링 또는 적절한 처리 필요
        # 여기서는 None이 아닌 것들만 stack
        valid_embeddings = [emb for emb in final_embeddings_list if emb is not None]
        if not valid_embeddings:
            return np.array([])
        
        # 모든 임베딩이 동일한 shape인지 확인 (스택하기 전)
        first_shape = valid_embeddings[0].shape
        if not all(emb.shape == first_shape for emb in valid_embeddings):
            logger.error("스택할 임베딩들의 차원이 일치하지 않습니다.")
            # 차원이 다른 경우 어떻게 처리할지 결정해야 함 (예: 오류 발생, 빈 배열 반환 등)
            # 여기서는 가장 흔한 경우(모두 동일 차원)를 가정하고 진행. 문제가 지속되면 상세 로깅 필요.
            # 임시로 첫 번째 임베딩과 차원이 같은 것들만 사용하거나, 오류 발생
            valid_embeddings = [emb for emb in valid_embeddings if emb.shape == first_shape]
            if not valid_embeddings: return np.array([])

        return np.stack(valid_embeddings)