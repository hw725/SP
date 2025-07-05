"""PA 전용 정렬기 - spaCy 순차적 분할 정렬만 사용 (SA 연동 완전 제거, circular import 완전 제거)"""
import sys
import os
import importlib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import List, Dict
from sentence_splitter import split_target_sentences_advanced, split_source_by_whitespace_and_align

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
    src_text: str, 
    embed_func=None,
    similarity_threshold: float = 0.3
) -> List[Dict]:
    """
    순차적 1:1 정렬 (공백/포맷 100% 보존, 의미적 align 제거)
    """
    if not tgt_sentences:
        return []
    
    # 원문을 번역문 개수에 맞춰 순차적으로 분할
    aligned_src_chunks = split_source_by_whitespace_and_align(src_text, len(tgt_sentences))
    
    alignments = []
    for i in range(len(tgt_sentences)):
        alignments.append({
            '원문': aligned_src_chunks[i] if i < len(aligned_src_chunks) else '',
            '번역문': tgt_sentences[i],
            'similarity': 1.0,  # 순차적 정렬이므로 유사도는 1.0
            'split_method': 'punctuation',
            'align_method': 'sequential'
        })
    
    # 남은 원문 청크가 있으면 추가
    for j in range(len(tgt_sentences), len(aligned_src_chunks)):
        alignments.append({
            '원문': aligned_src_chunks[j],
            '번역문': '',
            'similarity': 0.0,
            'split_method': 'punctuation',
            'align_method': 'sequential_unmatched_src'
        })
    
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
    tgt_sentences = split_target_sentences_advanced(tgt_paragraph, max_length, splitter="punctuation")
    print(f"   번역문: {len(tgt_sentences)}개 문장")
    print(f"   원문 길이: {len(src_paragraph)}자")
    
    # embed_func, similarity_threshold 등은 무시 (sequential align만 사용)
    alignments = improved_align_paragraphs(
        tgt_sentences, 
        src_paragraph  # 문자열로 직접 전달
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