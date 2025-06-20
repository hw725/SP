"""텍스트 임베딩 생성기"""

import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Optional
import torch

logger = logging.getLogger(__name__)

# 전역 모델 변수 (한 번만 로드)
_model = None

def get_embedding_model():
    """임베딩 모델 가져오기 (싱글톤 패턴)"""
    global _model
    
    if _model is None:
        try:
            logger.info("🤖 임베딩 모델 로딩 중...")
            
            # GPU 사용 가능하면 GPU, 아니면 CPU
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"📱 사용 디바이스: {device}")
            
            # 다국어 지원 모델 사용
            model_name = "paraphrase-multilingual-MiniLM-L12-v2"
            _model = SentenceTransformer(model_name, device=device)
            
            logger.info(f"✅ 임베딩 모델 로드 완료: {model_name}")
            
        except Exception as e:
            logger.error(f"❌ 임베딩 모델 로드 실패: {e}")
            logger.info("🔄 기본 더미 모델로 대체...")
            _model = "dummy"  # 더미 모델 표시
    
    return _model

def get_embeddings(texts: List[str]) -> List[np.ndarray]:
    """
    텍스트 리스트를 임베딩으로 변환
    
    Args:
        texts: 텍스트 리스트
    
    Returns:
        List[np.ndarray]: 임베딩 벡터 리스트
    """
    if not texts:
        return []
    
    try:
        model = get_embedding_model()
        
        if model == "dummy":
            # 더미 임베딩 생성 (모델 로드 실패 시)
            logger.warning("⚠️ 더미 임베딩 사용 중")
            return [np.random.randn(384) for _ in texts]  # MiniLM 차원수
        
        # 실제 임베딩 생성
        embeddings = model.encode(texts, convert_to_numpy=True)
        
        logger.info(f"✅ 임베딩 생성 완료: {len(texts)}개 → {embeddings.shape}")
        
        return [emb for emb in embeddings]
        
    except Exception as e:
        logger.error(f"❌ 임베딩 생성 실패: {e}")
        logger.info("🔄 더미 임베딩으로 대체")
        
        # 오류 시 더미 임베딩 반환
        return [np.random.randn(384) for _ in texts]

def get_similarity(text1: str, text2: str) -> float:
    """
    두 텍스트 간 유사도 계산
    
    Args:
        text1: 첫 번째 텍스트
        text2: 두 번째 텍스트
    
    Returns:
        float: 유사도 (0~1)
    """
    try:
        embeddings = get_embeddings([text1, text2])
        
        if len(embeddings) != 2:
            return 0.0
        
        # 코사인 유사도 계산
        emb1, emb2 = embeddings[0], embeddings[1]
        
        # 정규화
        emb1_norm = emb1 / np.linalg.norm(emb1)
        emb2_norm = emb2 / np.linalg.norm(emb2)
        
        # 코사인 유사도
        similarity = np.dot(emb1_norm, emb2_norm)
        
        # 0~1 범위로 정규화
        similarity = (similarity + 1) / 2
        
        return float(similarity)
        
    except Exception as e:
        logger.error(f"❌ 유사도 계산 실패: {e}")
        return 0.0

def batch_similarity(texts1: List[str], texts2: List[str]) -> np.ndarray:
    """
    배치 유사도 계산
    
    Args:
        texts1: 첫 번째 텍스트 리스트
        texts2: 두 번째 텍스트 리스트
    
    Returns:
        np.ndarray: 유사도 매트릭스 (len(texts1) x len(texts2))
    """
    try:
        if not texts1 or not texts2:
            return np.zeros((len(texts1), len(texts2)))
        
        # 모든 텍스트 임베딩
        all_texts = texts1 + texts2
        all_embeddings = get_embeddings(all_texts)
        
        if len(all_embeddings) != len(all_texts):
            logger.error("❌ 임베딩 수가 일치하지 않음")
            return np.zeros((len(texts1), len(texts2)))
        
        # 분할
        emb1 = np.array(all_embeddings[:len(texts1)])
        emb2 = np.array(all_embeddings[len(texts1):])
        
        # 정규화
        emb1_norm = emb1 / np.linalg.norm(emb1, axis=1, keepdims=True)
        emb2_norm = emb2 / np.linalg.norm(emb2, axis=1, keepdims=True)
        
        # 유사도 매트릭스 계산
        similarity_matrix = np.dot(emb1_norm, emb2_norm.T)
        
        # 0~1 범위로 정규화
        similarity_matrix = (similarity_matrix + 1) / 2
        
        return similarity_matrix
        
    except Exception as e:
        logger.error(f"❌ 배치 유사도 계산 실패: {e}")
        return np.zeros((len(texts1), len(texts2)))

if __name__ == "__main__":
    # 테스트
    logging.basicConfig(level=logging.INFO)
    
    test_texts = ["興也라", "興이다.", "蒹은 薕이요"]
    
    print("🧪 임베딩 테스트")
    embeddings = get_embeddings(test_texts)
    print(f"✅ 임베딩 형태: {[emb.shape for emb in embeddings]}")
    
    print("\n🧪 유사도 테스트")
    sim = get_similarity("興也라", "興이다.")
    print(f"✅ 유사도: {sim:.3f}")
    
    print("\n🧪 배치 유사도 테스트")
    batch_sim = batch_similarity(["興也라"], ["興이다.", "蒹은 薕이요"])
    print(f"✅ 배치 유사도:\n{batch_sim}")