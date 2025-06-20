"""SentenceTransformer 기반 임베딩 모듈"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
import hashlib
import pickle
import os
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# 전역 변수들
_model = None
_cache = {}
_cache_file = "embedding_cache.pkl"

def get_model():
    """임베딩 모델 로드 및 캐시"""
    global _model
    if _model is None:
        logger.info("🤖 임베딩 모델 로딩 중...")
        device = "cpu"  # GPU 사용 시 "cuda"
        logger.info(f"📱 사용 디바이스: {device}")
        
        model_name = "paraphrase-multilingual-MiniLM-L12-v2"
        _model = SentenceTransformer(model_name, device=device)
        logger.info(f"✅ 임베딩 모델 로드 완료: {model_name}")
    
    return _model

def _get_cache_key(texts: List[str]) -> str:
    """텍스트 리스트로부터 캐시 키 생성"""
    text_str = '|'.join(sorted(texts))
    return hashlib.md5(text_str.encode()).hexdigest()

def _load_cache():
    """캐시 파일 로드"""
    global _cache
    if os.path.exists(_cache_file):
        try:
            with open(_cache_file, 'rb') as f:
                _cache = pickle.load(f)
            logger.info(f"📂 캐시 로드: {len(_cache)}개 항목")
        except Exception as e:
            logger.warning(f"⚠️ 캐시 로드 실패: {e}")
            _cache = {}

def _save_cache():
    """캐시 파일 저장"""
    try:
        with open(_cache_file, 'wb') as f:
            pickle.dump(_cache, f)
        logger.info(f"💾 캐시 저장: {len(_cache)}개 항목")
    except Exception as e:
        logger.warning(f"⚠️ 캐시 저장 실패: {e}")

def compute_embeddings_with_cache(
    texts: List[str],
    model_name: Optional[str] = None,
    cache_enabled: bool = True
) -> np.ndarray:
    """캐시를 활용한 임베딩 계산"""
    
    if not texts:
        return np.array([])
    
    # 캐시 키 생성
    cache_key = _get_cache_key(texts) if cache_enabled else None
    
    # 캐시 확인
    if cache_enabled:
        _load_cache()
        if cache_key in _cache:
            logger.info(f"🎯 캐시 히트: {len(texts)}개 텍스트")
            return _cache[cache_key]
    
    # 임베딩 계산
    model = get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    
    logger.info(f"✅ 임베딩 생성 완료: {len(texts)}개 → {embeddings.shape}")
    
    # 캐시 저장
    if cache_enabled and cache_key:
        _cache[cache_key] = embeddings
        _save_cache()
    
    return embeddings

def compute_similarity(text1: str, text2: str) -> float:
    """두 텍스트 간 유사도 계산"""
    embeddings = compute_embeddings_with_cache([text1, text2])
    
    if len(embeddings) != 2:
        return 0.0
        
    emb1, emb2 = embeddings[0], embeddings[1]
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
    
    return float(similarity)

def clear_cache():
    """캐시 초기화"""
    global _cache
    _cache = {}
    if os.path.exists(_cache_file):
        os.remove(_cache_file)
    logger.info("🗑️ 캐시 초기화 완료")