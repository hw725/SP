# embedder.py - 텍스트 구에 대한 임베딩 계산 및 캐시 처리 모듈

import logging
from typing import List, Optional, Dict
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)  # 기본 INFO, 필요 시 main.py에서 재설정

# 전역 캐시: 텍스트(str) → 임베딩(np.ndarray)
_embedding_cache: Dict[str, np.ndarray] = {}

# 모델은 모듈 로딩 시 단 한 번만 생성
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
try:
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)
    logger.info(f"[Embedder] Model loaded on {device}")
except Exception as e:
    logger.error(f"[Embedder] 모델 로딩 실패: {e}")
    raise


def compute_embeddings_with_cache(
    texts: List[str],
    batch_size: int = 128,
    show_batch_progress: bool = False
) -> np.ndarray:
    """
    texts 리스트 순서대로 임베딩을 계산하되,
    이미 _embedding_cache에 저장된 텍스트는 재사용.
    """
    result_list: List[Optional[np.ndarray]] = [None] * len(texts)
    to_embed: List[str] = []
    indices_to_embed: List[int] = []

    # 1. 캐시에 있는지 확인
    for i, txt in enumerate(texts):
        if txt in _embedding_cache:
            result_list[i] = _embedding_cache[txt]
        else:
            to_embed.append(txt)
            indices_to_embed.append(i)

    # 2. 캐시에 없는 텍스트만 배치 단위로 임베딩
    if to_embed:
        it = range(0, len(to_embed), batch_size)
        if show_batch_progress:
            it = tqdm(it, desc="Embedding batches", ncols=80)
        for start in it:
            batch = to_embed[start : start + batch_size]
            try:
                emb_batch = model.encode(batch, convert_to_tensor=True)
                emb_np = emb_batch.cpu().numpy()
            except Exception as e:
                logger.error(f"[Embedder] 임베딩 오류 (start={start}): {e}")
                # 실패한 경우, 해당 배치 전체를 0벡터로 대체할 수도 있음. 여기서는 예외 재발생.
                raise

            for k, vec in enumerate(emb_np):
                idx = indices_to_embed[start + k]
                _embedding_cache[texts[idx]] = vec
                result_list[idx] = vec

    # 3. numpy 배열로 쌓아서 반환
    return np.vstack(result_list)