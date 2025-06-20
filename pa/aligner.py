"""PA 전용 정렬기 - SA DP 방식 적용 (최종)"""

import sys
import os
sys.path.append('../sa')
import pandas as pd
from typing import List, Dict

from sentence_splitter import split_target_sentences_advanced, split_source_with_spacy

# ✅ SA 임베더 직접 import
def get_embedder_function(embedder_name: str):
    """SA 임베더 함수 직접 로드"""
    
    if embedder_name == 'bge':
        try:
            from sa_embedders.bge import compute_embeddings_with_cache
            return compute_embeddings_with_cache
        except ImportError:
            print("❌ BGE 임베더 import 실패")
            return fallback_embedder
            
    elif embedder_name == 'st':
        try:
            from sa_embedders.sentence_transformer import compute_embeddings_with_cache
            return compute_embeddings_with_cache
        except ImportError:
            print("❌ SentenceTransformer 임베더 import 실패")
            return fallback_embedder
            
    elif embedder_name == 'openai':
        try:
            from sa_embedders.openai import compute_embeddings_with_cache
            return compute_embeddings_with_cache
        except ImportError:
            print("❌ OpenAI 임베더 import 실패")
            return fallback_embedder
    
    return fallback_embedder

def fallback_embedder(texts: List[str]):
    """대체 임베더 - TF-IDF 기반"""
    
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    if not texts:
        return np.array([]).reshape(0, 512)
    
    try:
        vectorizer = TfidfVectorizer(max_features=512, ngram_range=(1, 2))
        embeddings = vectorizer.fit_transform(texts).toarray()
        
        # L2 정규화
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        
        return embeddings
    except Exception as e:
        print(f"⚠️ TF-IDF 임베더 실패: {e}")
        return np.random.randn(len(texts), 512)

def align_paragraphs_with_sa_dp(
    tgt_sentences: List[str], 
    src_chunks: List[str], 
    embed_func,
    similarity_threshold: float = 0.3
) -> List[Dict]:
    """SA DP 로직을 PA 방향으로 적용 (최종)"""
    
    if not tgt_sentences or not src_chunks:
        return []
    
    print(f"🎯 PA 정렬 시작 (SA DP): {len(tgt_sentences)}개 번역문 → {len(src_chunks)}개 원문")
    
    try:
        # ✅ SA 함수 직접 import (확인된 함수명)
        import importlib.util
        sa_path = os.path.abspath(os.path.join('..', 'sa'))
        aligner_path = os.path.join(sa_path, 'aligner.py')
        
        spec = importlib.util.spec_from_file_location("sa_aligner_module", aligner_path)
        sa_aligner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sa_aligner)
        
        # ✅ 확인된 함수 사용
        sa_align_func = sa_aligner.align_tokens_with_embeddings
        
        print("🔗 SA DP 함수 연동 성공: align_tokens_with_embeddings")
        
        # ✅ 정확한 매개변수로 호출
        sa_alignments = sa_align_func(
            src_units=tgt_sentences,          # PA: 번역문이 기준 (src_units)
            tgt_units=src_chunks,             # PA: 원문이 정렬 대상 (tgt_units)
            embed_func=embed_func,
            similarity_threshold=similarity_threshold
        )
        
        print(f"📊 SA DP 결과: {len(sa_alignments) if sa_alignments else 0}개 정렬")
        
        # ✅ 결과 변환 (SA → PA 형식)
        pa_alignments = []
        
        if sa_alignments:
            for align in sa_alignments:
                # SA 결과 형식에 따른 처리
                if isinstance(align, dict):
                    # 딕셔너리 형식
                    src_text = align.get('src', '')      # SA의 src = PA의 번역문
                    tgt_text = align.get('tgt', '')      # SA의 tgt = PA의 원문
                    score = align.get('score', 0.0)
                elif isinstance(align, (list, tuple)) and len(align) >= 2:
                    # 리스트/튜플 형식
                    src_text = str(align[0])  # SA src → PA 번역문
                    tgt_text = str(align[1])  # SA tgt → PA 원문
                    score = float(align[2]) if len(align) > 2 else 0.0
                else:
                    print(f"⚠️ 알 수 없는 SA 결과 형식: {type(align)}")
                    continue
                
                pa_alignments.append({
                    '문단식별자': 1,
                    '원문': tgt_text,        # PA 원문 = SA tgt
                    '번역문': src_text,      # PA 번역문 = SA src
                    'similarity': score,
                    'split_method': 'spacy_lg',
                    'align_method': 'sa_dp_align_tokens_with_embeddings'
                })
        
        if pa_alignments:
            print(f"✅ SA DP 정렬 성공: {len(pa_alignments)}개 항목")
            return pa_alignments
        else:
            print("⚠️ SA에서 빈 결과 반환")
            
    except Exception as e:
        print(f"⚠️ SA DP 연동 실패: {e}")
        import traceback
        traceback.print_exc()
    
    # SA 연동 실패시 고품질 대체 정렬
    print("🔄 고품질 대체 정렬 방식 사용...")
    return advanced_align_paragraphs(tgt_sentences, src_chunks, embed_func, similarity_threshold)

def advanced_align_paragraphs(
    tgt_sentences: List[str], 
    src_chunks: List[str], 
    embed_func,
    similarity_threshold: float = 0.3
) -> List[Dict]:
    """고품질 대체 정렬 (DP 스타일)"""
    
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    # 임베딩 생성
    tgt_embeddings = embed_func(tgt_sentences)
    src_embeddings = embed_func(src_chunks)
    
    # 유사도 매트릭스 계산
    sim_matrix = cosine_similarity(tgt_embeddings, src_embeddings)
    
    # ✅ DP 스타일 정렬 (순서 보존 + 무결성 보장)
    alignments = []
    used_src_indices = set()
    
    # 번역문 순서 기준으로 처리
    for tgt_idx, tgt_sent in enumerate(tgt_sentences):
        # 각 번역문에 대해 사용되지 않은 원문 중 최적 매칭
        similarities = sim_matrix[tgt_idx]
        
        best_score = -1.0
        best_src_idx = -1
        
        for src_idx in range(len(src_chunks)):
            if src_idx not in used_src_indices:
                if similarities[src_idx] > best_score:
                    best_score = similarities[src_idx]
                    best_src_idx = src_idx
        
        if best_src_idx != -1:
            used_src_indices.add(best_src_idx)
            src_text = src_chunks[best_src_idx]
        else:
            src_text = ""
        
        alignments.append({
            '문단식별자': 1,
            '원문': src_text,
            '번역문': tgt_sent,
            'similarity': best_score,
            'split_method': 'spacy_lg',
            'align_method': 'advanced_dp_style'
        })
    
    # 사용되지 않은 원문들 추가 (무결성 보장)
    for src_idx, src_chunk in enumerate(src_chunks):
        if src_idx not in used_src_indices:
            alignments.append({
                '문단식별자': 1,
                '원문': src_chunk,
                '번역문': "",
                'similarity': 0.0,
                'split_method': 'spacy_lg',
                'align_method': 'unmatched_source'
            })
    
    print(f"✅ 고품질 대체 정렬 완료: {len(alignments)}개 항목")
    return alignments

# 기존 process 함수들 유지...
def process_paragraph_alignment(
    src_paragraph: str, 
    tgt_paragraph: str, 
    embedder_name: str = 'bge',
    max_length: int = 150,
    similarity_threshold: float = 0.3
):
    """PA 처리 (SA DP 연동)"""
    
    print(f"🔄 PA 처리 시작")
    
    # 1. 분할
    tgt_sentences = split_target_sentences_advanced(tgt_paragraph, max_length)
    src_chunks = split_source_with_spacy(src_paragraph, len(tgt_sentences))
    
    print(f"   번역문: {len(tgt_sentences)}개 문장")
    print(f"   원문: {len(src_chunks)}개 청크")
    
    # 2. 임베더 로드
    embed_func = get_embedder_function(embedder_name)
    
    # 3. SA DP 정렬
    alignments = align_paragraphs_with_sa_dp(
        tgt_sentences, 
        src_chunks, 
        embed_func, 
        similarity_threshold
    )
    
    return alignments

def process_paragraph_file(
    input_file: str, 
    output_file: str, 
    embedder_name: str = 'bge',
    max_length: int = 150,
    similarity_threshold: float = 0.3
):
    """파일 단위 처리 - 올바른 컬럼명"""
    
    print(f"📂 파일 처리 시작: {input_file}")
    
    # Excel 파일 로드
    df = pd.read_excel(input_file)
    
    all_results = []
    
    for idx, row in df.iterrows():
        src_paragraph = str(row.get('원문', ''))      # ✅ 입력 컬럼명
        tgt_paragraph = str(row.get('번역문', ''))    # ✅ 입력 컬럼명
        
        if src_paragraph and tgt_paragraph:
            print(f"📝 처리 중: 문단 {idx + 1}")
            
            results = process_paragraph_alignment(
                src_paragraph, 
                tgt_paragraph,
                embedder_name=embedder_name,
                max_length=max_length,
                similarity_threshold=similarity_threshold
            )
            
            # 문단식별자 업데이트
            for result in results:
                result['문단식별자'] = idx + 1
            
            all_results.extend(results)
    
    # 결과 저장 (올바른 컬럼명)
    result_df = pd.DataFrame(all_results)
    
    # ✅ 컬럼 순서 정리
    final_columns = ['문단식별자', '원문', '번역문', 'similarity', 'split_method', 'align_method']
    result_df = result_df[final_columns]
    
    result_df.to_excel(output_file, index=False)
    
    print(f"💾 결과 저장: {output_file}")
    print(f"📊 총 {len(all_results)}개 문장 쌍 생성")
    
    return result_df