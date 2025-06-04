# embedder.py - 텍스트 구에 대한 임베딩 계산 및 캐시 처리 모듈

import logging
from typing import List, Dict, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)
# ❌ handler 추가나 setLevel 호출은 하지 않습니다.
#    로깅 설정(basicConfig)은 main.py에서 한 번만 처리됩니다.

# 전역 캐시: 텍스트(str) -> 임베딩(np.ndarray)
_embedding_cache: Dict[str, np.ndarray] = {}

# 모델은 모듈 로딩 시 단 한 번만 생성
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
try:
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)
    logger.info(f"[Embedding] Model loaded on {device}")
except Exception as e:
    logger.error(f"[Embedding] 모델 로딩 실패: {e}")
    raise


def compute_embeddings_with_cache(
    texts: List[str],
    batch_size: int = 256,
    show_batch_progress: bool = False
) -> np.ndarray:
    """
    texts 리스트 순서대로 임베딩을 계산하되,
    이미 _embedding_cache에 저장된 텍스트는 재사용.
    """
    result_list: List[Optional[np.ndarray]] = [None] * len(texts)
    to_embed: List[str] = []
    indices_to_embed: List[int] = []

    # 1) 캐시에 있는지 확인
    for i, txt in enumerate(texts):
        if txt in _embedding_cache:
            result_list[i] = _embedding_cache[txt]
        else:
            to_embed.append(txt)
            indices_to_embed.append(i)

    # 2) 캐시에 없는 텍스트만 배치 단위로 임베딩
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
                logger.error(f"[Embedding] 임베딩 오류 (start={start}): {e}")
                raise

            for k, vec in enumerate(emb_np):
                idx = indices_to_embed[start + k]
                _embedding_cache[texts[idx]] = vec
                result_list[idx] = vec

    # 3) numpy 배열로 쌓아서 반환
    # BUG FIX: None 값이 남아있을 수 있으므로, None이 있으면 예외 발생
    if any(x is None for x in result_list):
        logger.error("[Embedding] 임베딩 결과에 None이 포함되어 있습니다.")
        raise ValueError("임베딩 결과에 None이 포함되어 있습니다.")
    return np.vstack(result_list)
