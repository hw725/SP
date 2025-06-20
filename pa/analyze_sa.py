"""SA 모듈 구조 탐지 스크립트 (수정)"""

import os
import sys
import importlib.util  # ✅ 올바른 import

def analyze_sa_structure():
    """SA 모듈 구조 분석"""
    
    sa_path = os.path.abspath(os.path.join('..', 'sa'))
    
    if not os.path.exists(sa_path):
        print(f"❌ SA 경로가 없습니다: {sa_path}")
        return
    
    print(f"📂 SA 경로: {sa_path}")
    
    # SA 디렉토리 내용 확인
    print("\n📋 SA 파일들:")
    for item in os.listdir(sa_path):
        item_path = os.path.join(sa_path, item)
        if os.path.isfile(item_path) and item.endswith('.py'):
            print(f"   📄 {item}")
        elif os.path.isdir(item_path):
            print(f"   📁 {item}/")
    
    # aligner.py 확인
    aligner_path = os.path.join(sa_path, 'aligner.py')
    if os.path.exists(aligner_path):
        print(f"\n🔍 {aligner_path} 함수들:")
        
        # SA 경로를 sys.path에 추가
        if sa_path not in sys.path:
            sys.path.insert(0, sa_path)
        
        try:
            # ✅ 올바른 동적 import
            spec = importlib.util.spec_from_file_location("sa_aligner_module", aligner_path)
            sa_aligner = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(sa_aligner)
            
            # 함수들 나열
            functions_found = []
            for name in dir(sa_aligner):
                if not name.startswith('_'):
                    obj = getattr(sa_aligner, name)
                    if callable(obj):
                        functions_found.append(name)
                        print(f"   ⚙️  {name}()")
            
            print(f"\n📊 총 {len(functions_found)}개 함수 발견")
            
            # 정렬 관련 함수들 찾기
            align_functions = [f for f in functions_found if 'align' in f.lower()]
            if align_functions:
                print(f"\n🎯 정렬 관련 함수들:")
                for func in align_functions:
                    print(f"   🔗 {func}")
            
            return functions_found, sa_aligner
                        
        except Exception as e:
            print(f"   ❌ import 실패: {e}")
            import traceback
            traceback.print_exc()
            return [], None
    
    else:
        print(f"\n❌ aligner.py가 없습니다: {aligner_path}")
        return [], None

def test_sa_functions(sa_module, functions):
    """SA 함수들 테스트"""
    
    if not sa_module or not functions:
        return
    
    print(f"\n🧪 SA 함수 테스트:")
    
    # 테스트 데이터
    test_tgt = ["이것은 테스트 문장입니다.", "두 번째 문장입니다."]
    test_src = ["这是测试句子。", "第二个句子。"]
    
    # 정렬 관련 함수들 테스트
    align_functions = [f for f in functions if 'align' in f.lower()]
    
    for func_name in align_functions:
        try:
            func = getattr(sa_module, func_name)
            print(f"\n🔍 {func_name} 시그니처 확인:")
            
            # 함수 시그니처 확인
            import inspect
            sig = inspect.signature(func)
            print(f"   매개변수: {list(sig.parameters.keys())}")
            
            # 간단한 호출 테스트 (에러만 확인)
            try:
                # 가능한 시그니처들 시도
                if 'src_units' in sig.parameters:
                    print(f"   ✅ src_units 매개변수 있음 - SA 표준 시그니처")
                elif len(sig.parameters) >= 2:
                    print(f"   ⚠️ 비표준 시그니처 - 매개변수 {len(sig.parameters)}개")
                
            except Exception as e:
                print(f"   ❌ 호출 테스트 실패: {e}")
                
        except Exception as e:
            print(f"   ❌ {func_name} 검사 실패: {e}")

if __name__ == "__main__":
    functions, sa_module = analyze_sa_structure()
    test_sa_functions(sa_module, functions)

"""PA 메인 프로세서 - 번역문 기준 원문 정렬"""

import sys
sys.path.append('../sa')
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

from sentence_splitter import split_target_sentences_advanced, split_source_with_spacy
from sa_embedders import get_embedder_module

def process_paragraph_alignment(
    src_paragraph: str, 
    tgt_paragraph: str, 
    embedder_name: str = 'bge',
    max_length: int = 150,
    similarity_threshold: float = 0.3
):
    """번역문 기준 원문 정렬 (올바른 방향)"""
    
    print(f"🔄 PA 처리 시작 (번역문→원문 정렬)")
    print(f"   임베더: {embedder_name}")
    
    # 1. 분할
    tgt_sentences = split_target_sentences_advanced(tgt_paragraph, max_length)
    src_chunks = split_source_with_spacy(src_paragraph, len(tgt_sentences))
    
    print(f"   번역문: {len(tgt_sentences)}개 문장 (기준점)")
    print(f"   원문: {len(src_chunks)}개 청크 (정렬 대상)")
    
    # 2. 임베더 로드
    embedder_module = get_embedder_module(embedder_name)
    embed_func = embedder_module.compute_embeddings_with_cache
    
    # 3. 번역문 각 문장에 대해 최적 원문 청크 찾기
    print("🎯 번역문 기준 원문 정렬...")
    
    tgt_embeddings = embed_func(tgt_sentences)
    src_embeddings = embed_func(src_chunks)
    
    alignments = []
    
    # 각 번역문에 대해 최적 원문 찾기 (중복 허용)
    for tgt_idx, tgt_sent in enumerate(tgt_sentences):
        tgt_emb = tgt_embeddings[tgt_idx]
        
        best_score = -1.0
        best_src_idx = -1
        
        for src_idx, src_chunk in enumerate(src_chunks):
            src_emb = src_embeddings[src_idx]
            similarity = cosine_similarity([tgt_emb], [src_emb])[0][0]
            
            if similarity > best_score:
                best_score = similarity
                best_src_idx = src_idx
        
        # 임계값 체크
        if best_score >= similarity_threshold:
            src_text = src_chunks[best_src_idx]
        else:
            src_text = ""
            best_src_idx = -1
        
        alignments.append({
            'paragraph_id': 1,
            'tgt_sentence_id': tgt_idx + 1,
            'src_chunk_id': best_src_idx + 1 if best_src_idx != -1 else -1,
            'tgt_sentence': tgt_sent,
            'src_chunk': src_text,
            'similarity': best_score,
            'split_method': 'spacy_lg',
            'align_method': 'tgt_to_src_direct'
        })
    
    print(f"✅ PA 처리 완료: {len(alignments)}개 정렬")
    return alignments

def process_paragraph_file(input_file: str, output_file: str, **kwargs):
    """파일 단위 처리"""
    
    # Excel 파일 로드
    df = pd.read_excel(input_file)
    
    all_results = []
    
    for idx, row in df.iterrows():
        src_paragraph = str(row.get('원문', ''))
        tgt_paragraph = str(row.get('번역문', ''))
        
        if src_paragraph and tgt_paragraph:
            results = process_paragraph_alignment(src_paragraph, tgt_paragraph, **kwargs)
            
            # paragraph_id 업데이트
            for result in results:
                result['paragraph_id'] = idx + 1
            
            all_results.extend(results)
    
    # 결과 저장
    result_df = pd.DataFrame(all_results)
    result_df.to_excel(output_file, index=False)
    
    print(f"💾 결과 저장: {output_file}")
    return result_df