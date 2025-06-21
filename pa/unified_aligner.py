"""통합된 PA 정렬기 - 모든 요소 연결"""

import sys
import os
sys.path.append('../sa')
import pandas as pd
from typing import List, Dict
import re
import numpy as np

# ✅ 1. 통합된 임베더 로딩
def get_unified_embedder(embedder_name: str):
    """통합된 임베더 로더"""
    
    print(f"🧠 임베더 로딩: {embedder_name}")
    
    # SA 임베더 우선 시도
    try:
        if embedder_name == 'bge':
            from sa_embedders.bge import compute_embeddings_with_cache
            print("✅ SA BGE 임베더 로드 성공")
            return compute_embeddings_with_cache
            
        elif embedder_name == 'st':
            from sa_embedders.sentence_transformer import compute_embeddings_with_cache
            print("✅ SA SentenceTransformer 임베더 로드 성공")
            return compute_embeddings_with_cache
            
        elif embedder_name == 'openai':
            from sa_embedders.openai import compute_embeddings_with_cache
            print("✅ SA OpenAI 임베더 로드 성공")
            return compute_embeddings_with_cache
            
    except ImportError as e:
        print(f"⚠️ SA 임베더 로드 실패: {e}")
    
    # 독립 임베더 시도
    try:
        if embedder_name == 'bge':
            from FlagEmbedding import FlagModel
            model = FlagModel('BAAI/bge-m3', use_fp16=True)
            
            def bge_embedder(texts: List[str]) -> np.ndarray:
                return model.encode(texts)
            
            print("✅ 독립 BGE 임베더 로드 성공")
            return bge_embedder
            
        elif embedder_name == 'st':
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            
            def st_embedder(texts: List[str]) -> np.ndarray:
                return model.encode(texts)
            
            print("✅ 독립 SentenceTransformer 임베더 로드 성공")
            return st_embedder
            
    except ImportError as e:
        print(f"⚠️ 독립 임베더 로드 실패: {e}")
    
    # 최후 수단: TF-IDF
    print("🔄 TF-IDF 대체 임베더 사용")
    return create_tfidf_embedder()

def create_tfidf_embedder():
    """TF-IDF 기반 대체 임베더"""
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    def tfidf_embedder(texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([]).reshape(0, 512)
        
        try:
            vectorizer = TfidfVectorizer(
                max_features=512, 
                ngram_range=(1, 2),
                analyzer='char'  # 한중일 문자 처리
            )
            embeddings = vectorizer.fit_transform(texts).toarray()
            
            # L2 정규화
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)
            
            return embeddings
        except Exception as e:
            print(f"⚠️ TF-IDF 실패: {e}")
            return np.random.randn(len(texts), 512)
    
    return tfidf_embedder

# ✅ 2. 통합된 전처리 (sentence_splitter.py에서 가져옴)
def preprocess_text_unified(text: str) -> str:
    """통합된 텍스트 전처리"""
    
    if not text or not isinstance(text, str):
        return ""
    
    # 개행 정리
    text = re.sub(r'\r\n|\r|\n', ' ', text)
    
    # 공백 정리
    text = re.sub(r'[\u00A0\u1680\u2000-\u200B\u202F\u205F\u3000\uFEFF\t]+', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    
    # 특수 문자 정리
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    return text.strip()

# ✅ 3. 번역문 우선 분할 (aligner_correct.py에서 개선)
def split_target_by_punctuation_space(text: str, max_length: int = 150) -> List[str]:
    """번역문 구두점+공백 분할"""
    
    # 전처리
    text = preprocess_text_unified(text)
    
    # 구두점+공백 패턴
    punctuation_space_pattern = r'([.!?。！？])\s+'
    parts = re.split(punctuation_space_pattern, text.strip())
    
    sentences = []
    current = ""
    
    i = 0
    while i < len(parts):
        part = parts[i].strip()
        
        if not part:
            i += 1
            continue
            
        if part in '.!?。！？':
            current += part
            if current.strip():
                sentences.append(current.strip())
                current = ""
        else:
            current += part
        
        i += 1
    
    if current.strip():
        sentences.append(current.strip())
    
    # 150자 초과 시 맥락 분할
    final_sentences = []
    for sentence in sentences:
        if len(sentence) <= max_length:
            final_sentences.append(sentence)
        else:
            context_splits = split_by_context_unified(sentence, max_length)
            final_sentences.extend(context_splits)
    
    return final_sentences

def split_by_context_unified(text: str, max_length: int) -> List[str]:
    """맥락 기반 분할 (개선)"""
    
    if len(text) <= max_length:
        return [text]
    
    context_patterns = [
        r'([,，]\s*(?:그런데|그러나|하지만|따라서|그리하여|그러므로|또한|또|그리고))',
        r'([,，]\s*[\'"])',
        r'(라고\s+[했말])',
        r'(다고\s+[했말])',
        r'([,，]\s*(?:이제|그때|그후|먼저|다음에|이어서))',
        r'([,，]\s*<[^>]+>)',
        r'([,，]\s*)',
        r'(에서\s+)',
        r'(에게\s+)',
        r'(으로\s+)',
    ]
    
    # 최적 분할점 찾기
    min_pos = int(max_length * 0.3)
    max_pos = int(min(max_length * 0.7, len(text) - 20))
    
    best_pos = None
    best_score = -1
    
    for priority, pattern in enumerate(context_patterns):
        matches = list(re.finditer(pattern, text))
        
        for match in matches:
            pos = match.end()
            
            if min_pos <= pos <= max_pos:
                center = max_length * 0.5
                distance_score = 1.0 - abs(pos - center) / center
                priority_score = 1.0 - (priority * 0.1)
                total_score = distance_score * 0.7 + priority_score * 0.3
                
                if total_score > best_score:
                    best_score = total_score
                    best_pos = pos
    
    if best_pos:
        left = text[:best_pos].strip()
        right = text[best_pos:].strip()
        
        result = []
        if left:
            if len(left) > max_length:
                result.extend(split_by_context_unified(left, max_length))
            else:
                result.append(left)
        
        if right:
            if len(right) > max_length:
                result.extend(split_by_context_unified(right, max_length))
            else:
                result.append(right)
        
        return result
    else:
        mid = len(text) // 2
        return [text[:mid].strip(), text[mid:].strip()]

# ✅ 4. 원문 매칭 분할
def split_source_to_match_target_unified(src_text: str, target_count: int) -> List[str]:
    """원문을 번역문 개수에 맞춰 분할"""
    
    src_text = preprocess_text_unified(src_text)
    
    if target_count <= 1:
        return [src_text]
    
    # 한문 구문 경계로 분할
    boundary_patterns = [
        r'([也矣焉哉])\s*',
        r'([而然則故且])\s*',
        r'([曰云][:：]\s*)',
        r'([，,]\s*)',
        r'([。.]\s*)',
    ]
    
    chunks = []
    remaining = src_text.strip()
    
    while remaining:
        found = False
        
        for pattern in boundary_patterns:
            match = re.search(pattern, remaining)
            if match:
                end_pos = match.end()
                chunk = remaining[:end_pos].strip()
                
                if chunk:
                    chunks.append(chunk)
                
                remaining = remaining[end_pos:].strip()
                found = True
                break
        
        if not found:
            if remaining.strip():
                chunks.append(remaining.strip())
            break
    
    chunks = chunks if chunks else [src_text]
    
    # 개수 조정
    if len(chunks) == target_count:
        return chunks
    elif len(chunks) < target_count:
        return expand_source_chunks_unified(chunks, target_count)
    else:
        return merge_source_chunks_unified(chunks, target_count)

def expand_source_chunks_unified(chunks: List[str], target_count: int) -> List[str]:
    """원문 청크 확장"""
    
    expanded = []
    need_expand = target_count - len(chunks)
    
    chunks_with_length = [(i, chunk, len(chunk)) for i, chunk in enumerate(chunks)]
    chunks_with_length.sort(key=lambda x: x[2], reverse=True)
    
    expand_indices = set(x[0] for x in chunks_with_length[:need_expand])
    
    for i, chunk in enumerate(chunks):
        if i in expand_indices and len(chunk) > 10:
            mid = len(chunk) // 2
            
            # 적절한 분할점 찾기
            for offset in range(min(5, len(chunk)//4)):
                pos = mid + offset
                if pos < len(chunk) and chunk[pos] in ' \t，,。.':
                    mid = pos + 1
                    break
                pos = mid - offset
                if pos > 0 and chunk[pos-1] in ' \t，,。.':
                    mid = pos
                    break
            
            left = chunk[:mid].strip()
            right = chunk[mid:].strip()
            
            if left and right:
                expanded.extend([left, right])
            else:
                expanded.append(chunk)
        else:
            expanded.append(chunk)
    
    return expanded

def merge_source_chunks_unified(chunks: List[str], target_count: int) -> List[str]:
    """원문 청크 병합"""
    
    if len(chunks) <= target_count:
        return chunks
    
    merged = []
    chunks_per_group = len(chunks) / target_count
    
    current_group = []
    group_size = 0
    
    for chunk in chunks:
        current_group.append(chunk)
        group_size += 1
        
        if group_size >= chunks_per_group or chunk == chunks[-1]:
            merged_text = ' '.join(current_group).strip()
            merged.append(merged_text)
            current_group = []
            group_size = 0
    
    return merged

# ✅ 5. 통합된 정렬 함수
def unified_alignment(
    src_paragraph: str,
    tgt_paragraph: str,
    embedder_name: str = 'bge',
    max_length: int = 150,
    similarity_threshold: float = 0.3
) -> List[Dict]:
    """통합된 번역문 우선 정렬"""
    
    print(f"🎯 통합 PA 정렬 시작")
    print(f"   원문 길이: {len(src_paragraph)}")
    print(f"   번역문 길이: {len(tgt_paragraph)}")
    
    # 1. 임베더 로드
    embed_func = get_unified_embedder(embedder_name)
    
    # 2. 번역문 분할 (우선)
    tgt_units = split_target_by_punctuation_space(tgt_paragraph, max_length)
    print(f"📝 번역문 분할: {len(tgt_units)}개")
    
    # 3. 원문 분할 (번역문에 맞춤)
    src_units = split_source_to_match_target_unified(src_paragraph, len(tgt_units))
    print(f"🔍 원문 분할: {len(src_units)}개")
    
    # 4. 1:1 정렬
    alignments = create_one_to_one_alignment_unified(
        src_units, tgt_units, embed_func, similarity_threshold
    )
    
    print(f"✅ 정렬 완료: {len(alignments)}개")
    
    return alignments

def create_one_to_one_alignment_unified(
    src_units: List[str],
    tgt_units: List[str],
    embed_func,
    similarity_threshold: float
) -> List[Dict]:
    """1:1 정렬 생성 (통합)"""
    
    alignments = []
    max_len = max(len(src_units), len(tgt_units))
    
    # 길이 맞추기
    while len(src_units) < max_len:
        src_units.append("")
    while len(tgt_units) < max_len:
        tgt_units.append("")
    
    # 유사도 계산
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        
        valid_src = [s for s in src_units if s.strip()]
        valid_tgt = [t for t in tgt_units if t.strip()]
        
        if valid_src and valid_tgt:
            src_embeddings = embed_func(valid_src)
            tgt_embeddings = embed_func(valid_tgt)
            
            similarities = []
            valid_src_idx = 0
            valid_tgt_idx = 0
            
            for i in range(max_len):
                if (i < len(src_units) and src_units[i].strip() and 
                    i < len(tgt_units) and tgt_units[i].strip()):
                    
                    if (valid_src_idx < len(src_embeddings) and 
                        valid_tgt_idx < len(tgt_embeddings)):
                        
                        sim = cosine_similarity(
                            [tgt_embeddings[valid_tgt_idx]], 
                            [src_embeddings[valid_src_idx]]
                        )[0][0]
                        similarities.append(sim)
                        
                        valid_src_idx += 1
                        valid_tgt_idx += 1
                    else:
                        similarities.append(0.0)
                else:
                    similarities.append(0.0)
        else:
            similarities = [0.0] * max_len
            
    except Exception as e:
        print(f"⚠️ 유사도 계산 실패: {e}")
        similarities = [0.0] * max_len
    
    # 정렬 결과 생성
    for i in range(max_len):
        alignments.append({
            '원문': src_units[i] if i < len(src_units) else "",
            '번역문': tgt_units[i] if i < len(tgt_units) else "",
            'similarity': float(similarities[i]) if i < len(similarities) else 0.0,
            'split_method': 'unified_target_first',
            'align_method': 'unified_1to1'
        })
    
    return alignments

# ✅ 6. 파일 처리 함수
def process_file_unified(
    input_file: str,
    output_file: str,
    embedder_name: str = 'bge',
    max_length: int = 150,
    similarity_threshold: float = 0.3
) -> pd.DataFrame:
    """통합된 파일 처리"""
    
    print(f"📂 통합 PA 파일 처리: {input_file}")
    
    # 파일 로드
    try:
        df = pd.read_excel(input_file)
        print(f"📄 {len(df)}개 문단 로드")
    except Exception as e:
        print(f"❌ 파일 로드 실패: {e}")
        return None
    
    # 필수 컬럼 확인
    if '원문' not in df.columns or '번역문' not in df.columns:
        print(f"❌ 필수 컬럼 없음: {list(df.columns)}")
        return None
    
    all_results = []
    
    for idx, row in df.iterrows():
        src_paragraph = str(row.get('원문', '')).strip()
        tgt_paragraph = str(row.get('번역문', '')).strip()
        
        if not src_paragraph or not tgt_paragraph:
            print(f"⚠️ 빈 내용 건너뜀: 행 {idx + 1}")
            continue
        
        try:
            print(f"📝 처리 중: 문단 {idx + 1}/{len(df)}")
            
            alignments = unified_alignment(
                src_paragraph,
                tgt_paragraph,
                embedder_name=embedder_name,
                max_length=max_length,
                similarity_threshold=similarity_threshold
            )
            
            # 문단식별자 추가
            for result in alignments:
                result['문단식별자'] = idx + 1
            
            all_results.extend(alignments)
            
        except Exception as e:
            print(f"❌ 문단 {idx + 1} 처리 실패: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_results:
        print("❌ 처리 결과 없음")
        return None
    
    # 결과 저장
    try:
        result_df = pd.DataFrame(all_results)
        
        columns = ['문단식별자', '원문', '번역문', 'similarity', 'split_method', 'align_method']
        result_df = result_df[columns]
        
        result_df.to_excel(output_file, index=False)
        
        print(f"💾 결과 저장: {output_file}")
        print(f"📊 총 {len(all_results)}개 정렬")
        
        # 결과 분석
        analyze_results_unified(result_df)
        
        return result_df
        
    except Exception as e:
        print(f"❌ 저장 실패: {e}")
        return None

def analyze_results_unified(df: pd.DataFrame):
    """결과 분석"""
    
    print(f"\n📊 결과 분석:")
    print(f"   평균 유사도: {df['similarity'].mean():.3f}")
    print(f"   최고 유사도: {df['similarity'].max():.3f}")
    print(f"   최저 유사도: {df['similarity'].min():.3f}")
    
    high_quality = sum(1 for x in df['similarity'] if x > 0.7)
    medium_quality = sum(1 for x in df['similarity'] if 0.5 <= x <= 0.7)
    low_quality = sum(1 for x in df['similarity'] if x < 0.5)
    total = len(df)
    
    print(f"   고품질 (>0.7): {high_quality}/{total} ({high_quality/total*100:.1f}%)")
    print(f"   중품질 (0.5-0.7): {medium_quality}/{total} ({medium_quality/total*100:.1f}%)")
    print(f"   저품질 (<0.5): {low_quality}/{total} ({low_quality/total*100:.1f}%)")

# ✅ 7. 테스트 함수
def test_unified_system():
    """통합 시스템 테스트"""
    
    test_case = {
        "src": "蒹葭蒼蒼白露為霜所謂伊人在水一方遡洄從之道阻且長遡游從之宛在水中央",
        "tgt": "蒹葭는 푸르르고 白露는 서리가 되었다. 이른바 그 사람은 물 한편에 있다. 거슬러 올라가며 따라가니 길이 험하고 멀다. 물살 따라 내려가며 따라가니 물 한가운데 있는 듯하다."
    }
    
    print("🧪 통합 시스템 테스트")
    
    result = unified_alignment(test_case['src'], test_case['tgt'])
    
    print("\n🎯 테스트 결과:")
    for i, r in enumerate(result, 1):
        print(f"{i}. 원문: {r['원문']}")
        print(f"   번역: {r['번역문']}")
        print(f"   유사도: {r['similarity']:.3f}")

if __name__ == "__main__":
    test_unified_system()