"""PA 전용 정렬기 - spaCy 순차적 분할 정렬만 사용 (SA 연동 완전 제거, circular import 완전 제거)"""
import sys
import os
import importlib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import List, Dict
from sentence_splitter import split_target_sentences_advanced

# 패키지 import 방식으로 복원
from sa.sa_embedders import get_embedder

try:
    import torch
except ImportError:
    torch = None

def get_embedder_function(embedder_name: str, device: str = "cpu", openai_model: str = None, openai_api_key: str = None):
    # Robust device selection: if device=="cuda" but not available, fallback to cpu
    if device == "cuda":
        if torch is None or not torch.cuda.is_available():
            print("⚠️ torch 미설치 또는 CUDA 미지원: CPU로 전환합니다.")
            device = "cpu"
    if embedder_name == 'bge':
        return get_embedder("bge", device_id=device)
    elif embedder_name == 'openai':
        sa_openai = importlib.import_module('sa.sa_embedders.openai')
        compute_embeddings_with_cache = sa_openai.compute_embeddings_with_cache
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        def embed_func(texts):
            return compute_embeddings_with_cache(
                texts, 
                model=openai_model if openai_model else "text-embedding-3-large"
            )
        return embed_func
    else:
        raise ValueError(f"지원하지 않는 임베더: {embedder_name}. 지원: openai, bge")

# improved_align_paragraphs 직접 포함 (circular import 제거)
def improved_align_paragraphs(
    tgt_sentences: List[str], 
    src_chunks: List[str], 
    embed_func,
    similarity_threshold: float = 0.3
) -> List[Dict]:
    if not tgt_sentences:
        return []
    if isinstance(src_chunks, str):
        source_text = src_chunks
    elif isinstance(src_chunks, list) and len(src_chunks) == 1:
        source_text = src_chunks[0]
    elif isinstance(src_chunks, list):
        source_text = ' '.join(str(chunk) for chunk in src_chunks)
    else:
        source_text = str(src_chunks) if src_chunks else ""
    if not source_text.strip():
        return [{
            '원문': '',
            '번역문': tgt_sent,
            'similarity': 0.0,
            'split_method': 'whitespace',
            'align_method': 'no_source'
        } for tgt_sent in tgt_sentences]
    print(f"🔄 의미적 병합 정렬 시작: {len(tgt_sentences)}개 번역문")
    from sentence_splitter import split_source_by_whitespace_and_align
    aligned_src_chunks = split_source_by_whitespace_and_align(source_text, tgt_sentences, embed_func, similarity_threshold)
    # 임베딩 유사도 계산 및 결과 생성
    from sklearn.metrics.pairwise import cosine_similarity
    def safe_embed(texts):
        try:
            return np.array(embed_func(texts))
        except Exception as e:
            print(f"임베딩 오류: {e}")
            return np.zeros((len(texts), 768))
    tgt_embeddings = safe_embed(tgt_sentences)
    src_embeddings = safe_embed(aligned_src_chunks)
    if tgt_embeddings.shape[1] != src_embeddings.shape[1]:
        min_dim = min(tgt_embeddings.shape[1], src_embeddings.shape[1])
        tgt_embeddings = tgt_embeddings[:, :min_dim]
        src_embeddings = src_embeddings[:, :min_dim]
    sim_matrix = cosine_similarity(tgt_embeddings, src_embeddings)
    alignments = []
    for i in range(len(tgt_sentences)):
        alignments.append({
            '원문': aligned_src_chunks[i] if i < len(aligned_src_chunks) else '',
            '번역문': tgt_sentences[i],
            'similarity': sim_matrix[i, i] if i < sim_matrix.shape[0] and i < sim_matrix.shape[1] else 0.0,
            'split_method': 'whitespace',
            'align_method': 'semantic_merge'
        })
    # 남은 원문 청크가 있으면 추가
    for j in range(len(tgt_sentences), len(aligned_src_chunks)):
        alignments.append({
            '원문': aligned_src_chunks[j],
            '번역문': '',
            'similarity': 0.0,
            'split_method': 'whitespace',
            'align_method': 'semantic_merge_unmatched_src'
        })
    print(f"✅ 의미적 병합 정렬 완료: {len(alignments)}개 항목")
    return alignments

def process_paragraph_alignment(
    src_paragraph: str, 
    tgt_paragraph: str, 
    embedder_name: str = 'bge',
    max_length: int = 150,
    similarity_threshold: float = 0.3,
    device: str = "cpu"
):
    """PA 처리 (공백/구두점 기반 순차적 분할만 사용)"""
    print(f"🔄 PA 처리 시작 (공백/구두점 순차적 분할)")
    tgt_sentences = split_target_sentences_advanced(tgt_paragraph, max_length, splitter="spacy")
    # 원문 분할: spaCy 완전 배제, 공백/구두점 기준 분할만 사용
    src_chunks = src_paragraph  # improved_align_paragraphs에서 직접 분할
    print(f"   번역문: {len(tgt_sentences)}개 문장")
    print(f"   원문: {len(src_paragraph)}개 토큰")
    embed_func = get_embedder_function(embedder_name, device=device)
    alignments = improved_align_paragraphs(
        tgt_sentences, 
        src_chunks, 
        embed_func, 
        similarity_threshold
    )
    # 문단식별자 부여
    for a in alignments:
        a['문단식별자'] = 1
    return alignments


def process_paragraph_file(
    input_file: str, 
    output_file: str, 
    embedder_name: str = 'bge',
    max_length: int = 150,
    similarity_threshold: float = 0.3,
    device: str = "cpu"
):
    """파일 단위 처리 - spaCy 순차적 분할 정렬만 사용"""
    print(f"📂 파일 처리 시작: {input_file}")
    df = pd.read_excel(input_file)
    all_results = []
    for idx, row in df.iterrows():
        src_paragraph = str(row.get('원문', ''))
        tgt_paragraph = str(row.get('번역문', ''))
        if src_paragraph and tgt_paragraph:
            alignments = process_paragraph_alignment(
                src_paragraph,
                tgt_paragraph,
                embedder_name=embedder_name,
                max_length=max_length,
                similarity_threshold=similarity_threshold,
                device=device
            )
            all_results.extend(alignments)
    result_df = pd.DataFrame(all_results)
    final_columns = ['문단식별자', '원문', '번역문', 'similarity', 'split_method', 'align_method']
    result_df = result_df[final_columns]
    result_df.to_excel(output_file, index=False)
    print(f"💾 결과 저장: {output_file}")
    print(f"📊 총 {len(all_results)}개 문장 쌍 생성")
    return result_df