"""PA 메인 프로세서 - 단순화"""

import pandas as pd
from typing import List, Dict
from sentence_splitter import split_target_sentences_advanced, split_source_with_spacy

def get_embedder_function(embedder_name: str):
    """임베더 함수 로드"""
    
    if embedder_name == 'bge':
        try:
            import sys
            sys.path.append('../sa')
            from sa_embedders.bge import compute_embeddings_with_cache
            return compute_embeddings_with_cache
        except ImportError:
            return fallback_embedder
            
    elif embedder_name == 'st':
        try:
            import sys
            sys.path.append('../sa')
            from sa_embedders.sentence_transformer import compute_embeddings_with_cache
            return compute_embeddings_with_cache
        except ImportError:
            return fallback_embedder
    
    return fallback_embedder

def fallback_embedder(texts: List[str]):
    """대체 임베더 - TF-IDF"""
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
    except Exception:
        return np.random.randn(len(texts), 512)

def simple_align_paragraphs(
    tgt_sentences: List[str], 
    src_chunks: List[str], 
    embed_func,
    similarity_threshold: float = 0.3
) -> List[Dict]:
    """단순 정렬"""
    
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    if not tgt_sentences or not src_chunks:
        return []
    
    # 임베딩 생성
    tgt_embeddings = embed_func(tgt_sentences)
    src_embeddings = embed_func(src_chunks)
    
    # 유사도 매트릭스 계산
    sim_matrix = cosine_similarity(tgt_embeddings, src_embeddings)
    
    # 단순 정렬
    alignments = []
    used_src_indices = set()
    
    for tgt_idx, tgt_sent in enumerate(tgt_sentences):
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
            'align_method': 'simple_align'
        })
    
    # 사용되지 않은 원문들 추가
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
    
    return alignments

def process_paragraph_file(
    input_file: str, 
    output_file: str, 
    embedder_name: str = 'bge',
    max_length: int = 150,
    similarity_threshold: float = 0.3
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
        embed_func = get_embedder_function(embedder_name)
        print(f"🧠 임베더 로드 완료: {embedder_name}")
    except Exception as e:
        print(f"❌ 임베더 로드 실패: {e}")
        embed_func = fallback_embedder
    
    all_results = []
    
    for idx, row in df.iterrows():
        src_paragraph = str(row.get('원문', '')).strip()
        tgt_paragraph = str(row.get('번역문', '')).strip()
        
        if not src_paragraph or not tgt_paragraph:
            print(f"⚠️ 빈 내용 건너뜀: 행 {idx + 1}")
            continue
        
        try:
            print(f"📝 처리 중: 문단 {idx + 1}/{len(df)}")
            
            # ✅ 문장 분할 (올바른 호출)
            tgt_sentences = split_target_sentences_advanced(tgt_paragraph, max_length)
            src_chunks = split_source_with_spacy(src_paragraph, tgt_sentences)  # List[str] 전달
            
            print(f"   번역문: {len(tgt_sentences)}개 문장")
            print(f"   원문: {len(src_chunks)}개 청크")
            
            # 정렬 수행
            alignments = simple_align_paragraphs(
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
            traceback.print_exc()  # 디버깅용 상세 에러
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
        
        return result_df
        
    except Exception as e:
        print(f"❌ 결과 저장 실패: {e}")
        return None