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
    device: str = "cpu",
    quality_threshold: float = 0.8
):
    """
    PA 처리: 순차적/의미적 정렬 모두 실행, 가중합 similarity(0.4/0.6)로 품질 기준(0.8) 이상이면 가중합 결과, 아니면 의미적 결과만 채택
    """
    print(f"🔄 PA 처리 시작 (순차적+의미적 병합)")
    # 1. 순차적 정렬
    tgt_sentences_seq = split_target_sentences_advanced(tgt_paragraph, max_length, splitter="punctuation")
    alignments_seq = improved_align_paragraphs(
        tgt_sentences_seq, 
        src_paragraph
    )
    # 2. 의미적 병합
    tgt_sentences_sem = split_target_sentences_advanced(tgt_paragraph, max_length, splitter="spacy")
    embed_func = get_embedder_function(embedder_name, device=device)
    alignments_sem = improved_align_paragraphs(
        tgt_sentences_sem,
        src_paragraph,
        embed_func,
        similarity_threshold
    )
    # 3. 쌍별 가중합 및 조건부 선택
    results = []
    max_len = max(len(alignments_seq), len(alignments_sem))
    for i in range(max_len):
        seq = alignments_seq[i] if i < len(alignments_seq) else {'원문':'','번역문':'','similarity':0.0,'split_method':'punctuation','align_method':'sequential'}
        sem = alignments_sem[i] if i < len(alignments_sem) else {'원문':'','번역문':'','similarity':0.0,'split_method':'spacy','align_method':'semantic'}
        # 가중합 similarity
        weighted_sim = seq['similarity']*0.4 + sem['similarity']*0.6
        if weighted_sim >= quality_threshold:
            # 가중합 결과 채택, 정보 병합
            result = {
                '원문': sem['원문'] if sem['원문'] else seq['원문'],
                '번역문': sem['번역문'] if sem['번역문'] else seq['번역문'],
                'similarity': weighted_sim,
                'split_method': f"seq+sem",
                'align_method': 'hybrid'
            }
        else:
            # 의미적 결과만 채택
            result = sem.copy()
            result['align_method'] = 'semantic_only'
        results.append(result)
    # 문단식별자 부여
    for a in results:
        a['문단식별자'] = 1
    return results


def process_paragraph_file(
    input_file: str, 
    output_file: str, 
    embedder_name: str = 'bge',
    max_length: int = 150,
    similarity_threshold: float = 0.3,
    device: str = "cpu",
    quality_threshold: float = 0.8
):
    """파일 단위 처리 - 순차적/의미적 정렬 모두 적용, 품질 기준 조건부 선택"""
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
                device=device,
                quality_threshold=quality_threshold
            )
            # 문단식별자 idx+1로 부여
            for a in alignments:
                a['문단식별자'] = idx + 1
            all_results.extend(alignments)
    result_df = pd.DataFrame(all_results)
    final_columns = ['문단식별자', '원문', '번역문', 'similarity', 'split_method', 'align_method']
    result_df = result_df[final_columns]

    # === 무결성 검증 및 보완 ===
    # 입력 전체 원문/번역문 연결
    input_src_all = ''.join([str(row.get('원문','')) for _, row in df.iterrows()])
    input_tgt_all = ''.join([str(row.get('번역문','')) for _, row in df.iterrows()])
    # 결과 전체 원문/번역문 연결
    output_src_all = ''.join(result_df['원문'].fillna(''))
    output_tgt_all = ''.join(result_df['번역문'].fillna(''))
    # 원문 보완
    if input_src_all != output_src_all:
        print('⚠️ 원문 무결성 불일치: 누락/중복 보정 시도')
        # 누락분 찾기
        from difflib import SequenceMatcher
        sm = SequenceMatcher(None, output_src_all, input_src_all)
        opcodes = sm.get_opcodes()
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'insert':
                # input_src_all[j1:j2]가 누락됨 → 마지막 원문에 덧붙임
                if len(result_df) > 0:
                    result_df.at[result_df.index[-1], '원문'] += input_src_all[j1:j2]
                else:
                    # 결과가 없으면 새 쌍 추가
                    result_df.loc[len(result_df)] = [df.shape[0], input_src_all[j1:j2], '', 1.0, 'integrity', 'src_patch']
            elif tag == 'delete':
                # output_src_all[i1:i2]가 중복됨 → 마지막 원문에서 제거
                if len(result_df) > 0:
                    last = result_df.at[result_df.index[-1], '원문']
                    result_df.at[result_df.index[-1], '원문'] = last.replace(output_src_all[i1:i2], '', 1)
    # 번역문 보완
    if input_tgt_all != output_tgt_all:
        print('⚠️ 번역문 무결성 불일치: 누락/중복 보정 시도')
        from difflib import SequenceMatcher
        sm = SequenceMatcher(None, output_tgt_all, input_tgt_all)
        opcodes = sm.get_opcodes()
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'insert':
                if len(result_df) > 0:
                    result_df.at[result_df.index[-1], '번역문'] += input_tgt_all[j1:j2]
                else:
                    result_df.loc[len(result_df)] = [df.shape[0], '', input_tgt_all[j1:j2], 1.0, 'integrity', 'tgt_patch']
            elif tag == 'delete':
                if len(result_df) > 0:
                    last = result_df.at[result_df.index[-1], '번역문']
                    result_df.at[result_df.index[-1], '번역문'] = last.replace(output_tgt_all[i1:i2], '', 1)
    # 최종 재정렬(컬럼 순서 보장)
    result_df = result_df[final_columns]
    result_df.to_excel(output_file, index=False)
    print(f"💾 결과 저장: {output_file}")
    print(f"📊 총 {len(all_results)}개 문장 쌍 생성")
    return result_df