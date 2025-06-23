"""PA 메인 프로세서 - 개선된 정렬"""

import pandas as pd
from typing import List, Dict
import numpy as np
from sentence_splitter import split_target_sentences_advanced, split_source_with_spacy
import torch
from aligner import get_embedder_function  # ✅ aligner의 임베더 함수만 사용

def get_device(device_preference="cuda"):
    if device_preference == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA(GPU)를 사용할 수 없습니다. CPU로 전환합니다.")
        return "cpu"
    return device_preference

def improved_align_paragraphs(
    tgt_sentences: List[str], 
    src_chunks: List[str], 
    embed_func,
    similarity_threshold: float = 0.3
) -> List[Dict]:
    """개선된 정렬 - 1:1 매칭 보장"""
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    if not tgt_sentences or not src_chunks:
        return []
    
    # 임베딩 생성 (항상 numpy array로 변환)
    tgt_embeddings = np.array(embed_func(tgt_sentences))
    src_embeddings = np.array(embed_func(src_chunks))

    # 임베딩 차원 체크
    if tgt_embeddings.shape[1] != src_embeddings.shape[1]:
        print(f"❌ 임베딩 차원 불일치: tgt={tgt_embeddings.shape}, src={src_embeddings.shape}")
        return []
    
    # 유사도 매트릭스 계산
    sim_matrix = cosine_similarity(tgt_embeddings, src_embeddings)
    
    alignments = []
    
    # ✅ 개선된 정렬: 길이에 따라 전략 선택
    if len(tgt_sentences) == len(src_chunks):
        # 1:1 순서 매칭
        for i in range(len(tgt_sentences)):
            alignments.append({
                '원문': src_chunks[i],
                '번역문': tgt_sentences[i],
                'similarity': sim_matrix[i][i] if i < len(src_chunks) else 0.0,
                'split_method': 'spacy_lg',
                'align_method': 'sequential_1to1'
            })
    
    elif len(tgt_sentences) > len(src_chunks):
        # 번역문이 더 많음: 원문을 여러 번역문에 분배
        alignments = distribute_sources_to_targets(
            tgt_sentences, src_chunks, sim_matrix, 'target_rich'
        )
    
    else:
        # 원문이 더 많음: 번역문을 여러 원문에 분배
        alignments = distribute_targets_to_sources(
            tgt_sentences, src_chunks, sim_matrix, 'source_rich'
        )
    
    return alignments

def distribute_sources_to_targets(
    tgt_sentences: List[str], 
    src_chunks: List[str], 
    sim_matrix: np.ndarray,
    method: str
) -> List[Dict]:
    """원문을 번역문에 분배"""
    
    alignments = []
    src_per_tgt = len(tgt_sentences) // len(src_chunks)
    remaining = len(tgt_sentences) % len(src_chunks)
    
    tgt_idx = 0
    
    for src_idx, src_chunk in enumerate(src_chunks):
        # 현재 원문에 할당할 번역문 개수
        assign_count = src_per_tgt + (1 if src_idx < remaining else 0)
        
        # 가장 유사한 번역문들 찾기
        if tgt_idx < len(tgt_sentences):
            end_idx = min(tgt_idx + assign_count, len(tgt_sentences))
            
            for t_idx in range(tgt_idx, end_idx):
                similarity = sim_matrix[t_idx][src_idx] if t_idx < sim_matrix.shape[0] else 0.0
                
                alignments.append({
                    '원문': src_chunk,
                    '번역문': tgt_sentences[t_idx],
                    'similarity': similarity,
                    'split_method': 'spacy_lg',
                    'align_method': method
                })
            
            tgt_idx = end_idx
    
    # 남은 번역문 처리
    while tgt_idx < len(tgt_sentences):
        alignments.append({
            '원문': "",
            '번역문': tgt_sentences[tgt_idx],
            'similarity': 0.0,
            'split_method': 'spacy_lg',
            'align_method': 'unmatched_target'
        })
        tgt_idx += 1
    
    return alignments

def distribute_targets_to_sources(
    tgt_sentences: List[str], 
    src_chunks: List[str], 
    sim_matrix: np.ndarray,
    method: str
) -> List[Dict]:
    """번역문을 원문에 분배"""
    
    alignments = []
    tgt_per_src = len(src_chunks) // len(tgt_sentences)
    remaining = len(src_chunks) % len(tgt_sentences)
    
    src_idx = 0
    
    for tgt_idx, tgt_sentence in enumerate(tgt_sentences):
        # 현재 번역문에 할당할 원문 개수
        assign_count = tgt_per_src + (1 if tgt_idx < remaining else 0)
        
        if src_idx < len(src_chunks):
            end_idx = min(src_idx + assign_count, len(src_chunks))
            
            # 첫 번째 원문과 매칭
            if src_idx < len(src_chunks):
                similarity = sim_matrix[tgt_idx][src_idx] if tgt_idx < sim_matrix.shape[0] else 0.0
                
                # 여러 원문을 합쳐서 하나의 매칭 생성
                combined_src = " ".join(src_chunks[src_idx:end_idx])
                
                alignments.append({
                    '원문': combined_src,
                    '번역문': tgt_sentence,
                    'similarity': similarity,
                    'split_method': 'spacy_lg',
                    'align_method': method
                })
            
            src_idx = end_idx
    
    # 남은 원문 처리
    while src_idx < len(src_chunks):
        alignments.append({
            '원문': src_chunks[src_idx],
            '번역문': "",
            'similarity': 0.0,
            'split_method': 'spacy_lg',
            'align_method': 'unmatched_source'
        })
        src_idx += 1
    
    return alignments

def process_paragraph_file(
    input_file: str, 
    output_file: str, 
    embedder_name: str = 'bge',
    max_length: int = 150,
    similarity_threshold: float = 0.3,
    device: str = "cuda"   # 기본값도 cuda로!
):
    """파일 단위 처리 (메인 함수)"""
    
    print(f"📂 PA 파일 처리 시작: {input_file}")
    
    try:
        # Excel 파일 로드
        df = pd.read_excel(input_file)
        print(f"📄 {len(df)}개 문단 로드됨")
        
    except FileNotFoundError:
        print(f"❌ 입력 파일을 찾을 수 없습니다: {input_file}")
        return None
    except Exception as e:
        print(f"❌ 파일 로드 에러: {e}")
        return None
    
    # 필수 컬럼 확인
    if '원문' not in df.columns or '번역문' not in df.columns:
        print(f"❌ 필수 컬럼이 없습니다: '원문', '번역문'")
        print(f"현재 컬럼: {list(df.columns)}")
        return None
    
    # 임베더 로드
    try:
        embed_func = get_embedder_function(embedder_name, device=device)
        print(f"🧠 임베더 로드 완료: {embedder_name} (device={device})")
    except Exception as e:
        print(f"❌ 임베더 로드 실패: {e}")
        from aligner import fallback_embedder_bge
        embed_func = fallback_embedder_bge(device)
    
    all_results = []
    
    for idx, row in df.iterrows():
        src_paragraph = str(row.get('원문', '')).strip()
        tgt_paragraph = str(row.get('번역문', '')).strip()
        
        if not src_paragraph or not tgt_paragraph:
            print(f"⚠️ 빈 내용 건너뜀: 행 {idx + 1}")
            continue
        
        try:
            print(f"📝 처리 중: 문단 {idx + 1}/{len(df)}")
            
            # 문장 분할
            tgt_sentences = split_target_sentences_advanced(tgt_paragraph, max_length)
            src_chunks = split_source_with_spacy(src_paragraph, tgt_sentences)
            
            print(f"   번역문: {len(tgt_sentences)}개 문장")
            print(f"   원문: {len(src_chunks)}개 청크")
            
            # ✅ 개선된 정렬 사용
            alignments = improved_align_paragraphs(
                tgt_sentences, 
                src_chunks, 
                embed_func, 
                similarity_threshold
            )
            
            # 문단식별자 업데이트
            for result in alignments:
                result['문단식별자'] = idx + 1
            
            all_results.extend(alignments)
            
        except Exception as e:
            print(f"❌ 문단 {idx + 1} 처리 실패: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_results:
        print("❌ 처리된 결과가 없습니다.")
        return None
    
    # 결과 저장
    try:
        result_df = pd.DataFrame(all_results)
        
        # 컬럼 순서 정리
        available_columns = result_df.columns.tolist()
        desired_columns = ['문단식별자', '원문', '번역문', 'similarity', 'split_method', 'align_method']
        final_columns = [col for col in desired_columns if col in available_columns]
        
        result_df = result_df[final_columns]
        result_df.to_excel(output_file, index=False)
        
        print(f"💾 결과 저장 완료: {output_file}")
        print(f"📊 총 {len(all_results)}개 문장 쌍 생성")
        
        # ✅ 결과 분석 추가
        analyze_alignment_results(result_df)
        
        return result_df
        
    except Exception as e:
        print(f"❌ 결과 저장 실패: {e}")
        return None

def analyze_alignment_results(result_df: pd.DataFrame):
    """정렬 결과 분석 (개선된 버전)"""
    
    print("\n📊 정렬 결과 분석:")
    
    # 문단별 통계
    paragraph_stats = result_df.groupby('문단식별자').agg({
        '원문': lambda x: sum(1 for text in x if str(text).strip()),
        '번역문': lambda x: sum(1 for text in x if str(text).strip()),
        'similarity': 'mean'
    }).round(3)
    
    print("📈 문단별 통계:")
    for idx, row in paragraph_stats.iterrows():
        print(f"   문단 {idx}: 원문 {row['원문']}개, 번역문 {row['번역문']}개, 유사도 {row['similarity']:.3f}")
    
    # 전체 유사도 분포
    print(f"\n🎯 전체 유사도:")
    print(f"   평균: {result_df['similarity'].mean():.3f}")
    print(f"   최고: {result_df['similarity'].max():.3f}")
    print(f"   최저: {result_df['similarity'].min():.3f}")
    
    # 고품질 매칭 비율
    high_quality = sum(1 for x in result_df['similarity'] if x > 0.7)
    medium_quality = sum(1 for x in result_df['similarity'] if 0.5 <= x <= 0.7)
    low_quality = sum(1 for x in result_df['similarity'] if x < 0.5)
    total = len(result_df)
    
    print(f"\n📊 품질별 매칭:")
    print(f"   고품질 (>0.7): {high_quality}/{total} ({high_quality/total*100:.1f}%)")
    print(f"   중품질 (0.5-0.7): {medium_quality}/{total} ({medium_quality/total*100:.1f}%)")
    print(f"   저품질 (<0.5): {low_quality}/{total} ({low_quality/total*100:.1f}%)")
    
    # 빈 매칭 확인
    empty_source = sum(1 for x in result_df['원문'] if not str(x).strip())
    empty_target = sum(1 for x in result_df['번역문'] if not str(x).strip())
    
    if empty_source > 0:
        print(f"⚠️ 빈 원문: {empty_source}개")
    if empty_target > 0:
        print(f"⚠️ 빈 번역문: {empty_target}개")
    
    # 정렬 방법별 통계
    if 'align_method' in result_df.columns:
        align_stats = result_df['align_method'].value_counts()
        print(f"\n🔀 정렬 방법별 통계:")
        for method, count in align_stats.items():
            avg_sim = result_df[result_df['align_method'] == method]['similarity'].mean()
            print(f"   {method}: {count}회 (평균 유사도 {avg_sim:.3f})")
    
    return paragraph_stats