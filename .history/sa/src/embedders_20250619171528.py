import os
import hashlib
import logging
import numpy as np
from typing import List, Optional, Dict
from pathlib import Path

from .interfaces import BaseEmbedder

logger = logging.getLogger(__name__)

class CachedEmbedder(BaseEmbedder):
    """캐시 기능이 있는 임베더"""
    def __init__(self, cache_dir: Optional[str] = None, **kwargs):
        self.cache_dir = cache_dir or os.getenv("EMBED_CACHE_DIR", "./.cache")
        self.model_identifier = self._get_model_identifier()  # 새로 추가
        # 모델별 캐시 디렉토리 생성
        self.model_cache_dir = os.path.join(self.cache_dir, self.model_identifier)
        os.makedirs(self.model_cache_dir, exist_ok=True)
        self.cache_enabled = True

    def _get_model_identifier(self) -> str:
        """서브클래스에서 override하여 고유 모델 식별자 반환"""
        return self.__class__.__name__

    def _get_cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _load_from_cache(self, key: str) -> Optional[np.ndarray]:
        cache_file = Path(self.model_cache_dir) / f"{key}.npy"  # model_cache_dir 사용
        if cache_file.exists():
            try:
                cached = np.load(cache_file)
                # 차원 검증 추가
                expected_dim = self._get_embedding_dimension()
                if cached.shape[0] != expected_dim:
                    logger.warning(f"캐시된 임베딩 차원({cached.shape[0]})이 현재 모델 차원({expected_dim})과 다릅니다. 캐시 무시.")
                    return None
                return cached
            except Exception as e:
                logger.warning(f"Cache load failed ({cache_file}): {e}")
        return None

    def _save_to_cache(self, key: str, emb: np.ndarray):
        cache_file = Path(self.model_cache_dir) / f"{key}.npy"  # model_cache_dir 사용
        try:
            np.save(cache_file, emb)
        except Exception as e:
            logger.warning(f"Cache save failed ({cache_file}): {e}")

    def _encode(self, text: str) -> np.ndarray:
        """서브클래스에서 override"""
        raise NotImplementedError

    def embed(self, texts: List[str]) -> np.ndarray:
        """기본 캐시 로직"""
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
                batch_embeddings = self._encode_batch(texts_to_encode_values)
                for idx, emb in zip(texts_to_encode_indices, batch_embeddings):
                    all_embeddings_map[idx] = emb
                    cache_key = self._get_cache_key(texts[idx])
                    self._save_to_cache(cache_key, emb)
            except Exception as e:
                logger.error(f"배치 인코딩 실패: {e}")
                # 개별 인코딩으로 fallback
                for idx in texts_to_encode_indices:
                    try:
                        emb = self._encode(texts[idx])
                        all_embeddings_map[idx] = emb
                        cache_key = self._get_cache_key(texts[idx])
                        self._save_to_cache(cache_key, emb)
                    except Exception as e2:
                        logger.error(f"개별 인코딩 실패 (인덱스 {idx}): {e2}")
                        # 기본값으로 설정
                        all_embeddings_map[idx] = np.zeros(self._get_embedding_dimension())
        else:
            logger.debug(f"{self.__class__.__name__}: 모든 {len(texts)}개 텍스트 캐시 로드.")

        final_embeddings_list = [all_embeddings_map[i] for i in range(len(texts))]
        if not final_embeddings_list:
            return np.array([])
        
        # Shape 검증 추가
        shapes = [emb.shape for emb in final_embeddings_list]
        if len(set(shapes)) > 1:
            logger.error(f"임베딩 shape 불일치: {shapes}")
            # 모든 임베딩을 동일한 차원으로 맞춤
            target_dim = self._get_embedding_dimension()
            normalized_embeddings = []
            for emb in final_embeddings_list:
                if emb.shape[0] != target_dim:
                    # 차원이 맞지 않으면 기본값으로 대체
                    normalized_embeddings.append(np.zeros(target_dim))
                else:
                    normalized_embeddings.append(emb)
            final_embeddings_list = normalized_embeddings
        
        return np.stack(final_embeddings_list)

    def _encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """배치 인코딩 (서브클래스에서 override 가능)"""
        return [self._encode(text) for text in texts]

    def _get_embedding_dimension(self) -> int:
        """임베딩 차원 반환 (서브클래스에서 override)"""
        return 768  # 기본값

class SentenceTransformerEmbedder(CachedEmbedder):
    """SentenceTransformer 임베더"""
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 
                 cache_dir: Optional[str] = None, device: Optional[str] = None, **kwargs):
        self.model_name = model_name
        super().__init__(cache_dir=cache_dir, **kwargs)
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
        if self.model is None:
            raise RuntimeError("모델이 로드되지 않았습니다")
        embedding = self.model.encode([text], convert_to_numpy=True)
        return embedding[0]

    def _encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        if self.model is None:
            raise RuntimeError("모델이 로드되지 않았습니다")
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [emb for emb in embeddings]

    def _get_embedding_dimension(self) -> int:
        if self.model is None:
            return 384  # 기본값
        return self.model.get_sentence_embedding_dimension()

    def _get_model_identifier(self) -> str:
        """SentenceTransformer 모델별 고유 식별자"""
        return f"st_{self.model_name.replace('/', '_').replace('-', '_')}"

class OpenAIEmbedder(CachedEmbedder):
    """OpenAI API 임베더"""
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-large",
                 cache_dir: Optional[str] = None, **kwargs):
        self.model = model
        super().__init__(cache_dir=cache_dir, **kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API 키가 필요합니다. 인자로 전달하거나 OPENAI_API_KEY 환경 변수를 설정하세요.")
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            logger.error("openai 패키지가 설치되지 않았습니다. `pip install openai`")
            raise

    def _encode(self, text: str) -> np.ndarray:
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"OpenAI 임베딩 실패: {e}")
            raise

    def _encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            return [np.array(data.embedding) for data in response.data]
        except Exception as e:
            logger.error(f"OpenAI 배치 임베딩 실패: {e}")
            return super()._encode_batch(texts)

    def _get_embedding_dimension(self) -> int:
        if "text-embedding-3-large" in self.model:
            return 3072
        elif "text-embedding-3-small" in self.model:
            return 1536
        elif "text-embedding-ada-002" in self.model:
            return 1536
        return 1536  # 기본값

    def _get_model_identifier(self) -> str:
        """OpenAI 모델별 고유 식별자"""
        return f"openai_{self.model.replace('-', '_')}"

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
        try:
            response = self.client.embed(
                texts=[text],
                model=self.model
            )
            return np.array(response.embeddings[0])
        except Exception as e:
            logger.error(f"Cohere 임베딩 실패: {e}")
            raise

    def _encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        try:
            response = self.client.embed(
                texts=texts,
                model=self.model
            )
            return [np.array(emb) for emb in response.embeddings]
        except Exception as e:
            logger.error(f"Cohere 배치 임베딩 실패: {e}")
            return super()._encode_batch(texts)

    def _get_embedding_dimension(self) -> int:
        if "embed-multilingual-v2.0" in self.model:
            return 768
        return 768  # 기본값

class BGEM3Embedder(CachedEmbedder):
    """BGEM3 임베더"""
    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool = True, cache_dir: Optional[str] = None, **kwargs):
        self.model_name = model_name
        super().__init__(cache_dir=cache_dir, **kwargs)
        try:
            from FlagEmbedding import BGEM3FlagModel
            self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        except ImportError:
            logger.error("FlagEmbedding 라이브러리를 찾을 수 없습니다. `pip install FlagEmbedding`으로 설치해주세요.")
            raise
        
        logger.info(f"BGEM3Embedder 초기화 완료: 모델={model_name}, FP16 사용={use_fp16}, 캐시 디렉토리='{self.cache_dir if self.cache_enabled else '비활성화됨'}'")

    def _normalize_l2(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 정규화 수행"""
        if not isinstance(embeddings, np.ndarray):
            logger.error(f"정규화 대상이 NumPy 배열이 아닙니다. 타입: {type(embeddings)}")
            return np.array([])

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
            return embeddings

    def _encode(self, text: str) -> np.ndarray:
        try:
            embedding = self.model.encode([text])['dense_vecs'][0]
            return self._normalize_l2(np.array(embedding))
        except Exception as e:
            logger.error(f"BGEM3 임베딩 실패: {e}")
            raise

    def _encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        try:
            result = self.model.encode(texts)
            embeddings = result['dense_vecs']
            return [self._normalize_l2(np.array(emb)) for emb in embeddings]
        except Exception as e:
            logger.error(f"BGEM3 배치 임베딩 실패: {e}")
            return super()._encode_batch(texts)

    def _get_embedding_dimension(self) -> int:
        return 1024  # BGEM3의 기본 차원

    def _get_model_identifier(self) -> str:
        """BGE 모델별 고유 식별자"""
        return f"bge_{self.model_name.replace('/', '_').replace('-', '_')}"