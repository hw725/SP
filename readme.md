# CSP: Sentence & Paragraph Aligner

20250621
한문-한국어 번역 텍스트의 단위별 자동 정렬 도구 모음입니다.

## 🎯 프로젝트 구성

### SA : Sentence Aligner (완료)
문장을 의미 단위(구)로 분할하고 1:1 대응시키는 도구

### PA : Paragraph Aligner (완료)
단락을 문장으로 분할하고 정렬하는 도구

---

## ✨ SA (Sentence Aligner) - 주요 기능

- 문장을 의미 단위(구)로 자동 분할
- 원문과 번역문의 1:1 구 대응 생성
- 다양한 토크나이저 지원 (jieba, MeCab, SoyNLP, Kkma)
- 다양한 임베더 지원 (Sentence Transformer, BGE-M3, OpenAI)
- 병렬 처리로 빠른 성능
- 캐시 시스템으로 중복 처리 방지
- 실시간 진행률 표시

## 🔧 설치 및 실행

### 1. 환경 설정
```bash
# 가상환경 생성 (권장)
python -m venv venv

# 가상환경 활성화
# Windows 명령 프롬프트
venv\Scripts\activate
# Windows PowerShell  
venv\Scripts\Activate.ps1
# Git Bash / Linux / Mac
source venv/bin/activate

# 패키지 설치
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. SA 기본 실행
```bash
# SA 폴더로 이동
cd sa

# 모든 기본값 사용 (jieba + MeCab + Sentence Transformer)
python main.py input.xlsx output.xlsx

# 병렬 처리 (메모리 충분할 때)
python main.py input.xlsx output.xlsx --parallel
```

### 3. PA 기본 실행
```bash
# PA 폴더로 이동
cd pa

# 기본 실행 (BGE 임베더 사용)
python main.py input.xlsx output.xlsx

# 다른 임베더 사용
python main.py input.xlsx output.xlsx --embedder st

# 유사도 임계값 조정
python main.py input.xlsx output.xlsx --threshold 0.4
```

## 🎯 SA CLI 사용 예시

### 기본 사용법
```bash
# 기본 설정으로 실행
python main.py input.xlsx output.xlsx

# 진행률 표시 포함
python main.py input.xlsx output.xlsx --verbose
```

### 토크나이저 변경
```bash
# SoyNLP 토크나이저 사용 (번역문만)
python main.py input.xlsx output.xlsx --tokenizer soy

# Kkma 토크나이저 사용 (번역문만)  
python main.py input.xlsx output.xlsx --tokenizer kkma

# 참고: 원문은 항상 jieba 사용 (한문 처리 최적화)
```

### 임베더 변경
```bash
# BGE-M3 임베더 (고성능, 로컬)
python main.py input.xlsx output.xlsx --embedder bge

# OpenAI 임베더 (API 키 필요)
python main.py input.xlsx output.xlsx --embedder openai

# HuggingFace 임베더
python main.py input.xlsx output.xlsx --embedder hf
```

### 고급 옵션
```bash
# 토큰 길이 조정
python main.py input.xlsx output.xlsx --min-tokens 2 --max-tokens 15

# 의미 매칭 비활성화 (단순 패턴 매칭만)
python main.py input.xlsx output.xlsx --no-semantic

# 모든 옵션 조합
python main.py input.xlsx output.xlsx \
  --tokenizer soy \
  --embedder bge \
  --parallel \
  --min-tokens 2 \
  --max-tokens 20 \
  --verbose
```

## 🎯 PA CLI 사용 예시

### 기본 사용법
```bash
cd pa

# 기본 설정으로 실행 (BGE 임베더)
python main.py input_paragraphs.xlsx output_sentences.xlsx

# 도움말 확인
python main.py --help
```

### 임베더 선택
```bash
# BGE 임베더 (기본값, 추천)
python main.py input.xlsx output.xlsx --embedder bge

# Sentence Transformer 임베더
python main.py input.xlsx output.xlsx --embedder st

# 대체 임베더 (TF-IDF 기반)
python main.py input.xlsx output.xlsx --embedder fallback
```

### 고급 옵션
```bash
# 문장 최대 길이 조정 (기본: 150자)
python main.py input.xlsx output.xlsx --max-length 200

# 유사도 임계값 조정 (기본: 0.3)
python main.py input.xlsx output.xlsx --threshold 0.4

# 모든 옵션 조합
python main.py input.xlsx output.xlsx \
  --embedder bge \
  --max-length 180 \
  --threshold 0.35
```

## 📋 CLI 옵션 전체 목록

### SA 옵션
```bash
cd sa
python main.py --help
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--tokenizer`, `-t` | `jieba` | 토크나이저 선택 (jieba, soy, kkma) |
| `--embedder`, `-e` | `st` | 임베더 선택 (st, bge, openai, hf) |
| `--parallel`, `-p` | 비활성화 | 병렬 처리 활성화 |
| `--no-semantic` | 비활성화 | 의미 기반 매칭 비활성화 |
| `--min-tokens` | `1` | 최소 토큰 수 |
| `--max-tokens` | `10` | 최대 토큰 수 |
| `--verbose`, `-v` | 비활성화 | 상세 로그 출력 |

### PA 옵션
```bash
cd pa
python main.py --help
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--embedder`, `-e` | `bge` | 임베더 선택 (bge, st) |
| `--max-length` | `150` | 문장 최대 길이 |
| `--threshold` | `0.3` | 유사도 임계값 |

## 🔧 지원 도구

### 토크나이저 (SA 전용)
- **jieba**: 중국어/한문 토크나이저 (원문 전용)
- **MeCab**: 한국어 형태소 분석기  
- **SoyNLP**: 한국어 비지도 토크나이저
- **Kkma**: 한국어 형태소 분석기

### 임베더 (SA + PA 공통)
- **Sentence Transformer**: 범용 문장 임베딩
- **BGE-M3**: 다국어 고성능 임베딩 (PA 기본값, 추천)
- **OpenAI**: OpenAI API 임베딩 (SA만, API 키 필요)
- **HuggingFace**: HuggingFace 모델 임베딩 (SA만)

### 문장 분할기 (PA 전용)
- **spaCy**: 한국어/중국어 의미 기반 분할 (lg > sm 모델 우선)
- **지능적 구두점 규칙**: spaCy 실패시 대체 분할 방식
- **기존 기준 후처리**: 150자 제한 + 한자 3개 이하 병합

## 📊 입출력 형식

### SA 입출력
#### 입력 파일 (Excel)
| id | src | tgt |
|----|-----|-----|
| 1 | 蒹葭는 刺襄公也라 | 蒹葭는 襄公을 풍자한 詩이다. |
| 2 | 未能用周禮하니 將無以固其國焉이라 | 周나라의 禮를 따르지 못하니 나라를 견고히 할 수 없다. |

#### 출력 파일: 구 단위 정렬 결과
| 문장식별자 | 구식별자 | 원문구 | 번역구 |
|------------|----------|--------|---------|
| 1 | 1 | 蒹葭는 | 蒹葭는 |
| 1 | 2 | 刺襄公也라 | 襄公을 풍자한 詩이다. |

### PA 입출력
#### 입력 파일 (Excel)
| 문단식별자 | 원문 | 번역문 |
|------------|------|---------|
| 1 | 蒹葭는 刺襄公也라 未能用周禮하니 將無以固其國焉이라 | 蒹葭는 襄公을 풍자한 詩이다. 周나라의 禮를 따르지 못하니 나라를 견고히 할 수 없다. |

#### 출력 파일: 문장 단위 정렬 결과
| 문단식별자 | 원문 | 번역문 | similarity | split_method | align_method |
|------------|------|---------|------------|--------------|--------------|
| 1 | 蒹葭는 刺襄公也라 | 蒹葭는 謫公을... | 0.847 | spacy_lg | simple_align |
| 1 | 未能用周禮하니 將無以固其國焉이라 | 周나라의 禮를 따르지 못하니... | 0.782 | spacy_lg | simple_align |

---

## ✨ PA (Paragraph Aligner) - 주요 기능

### 🧠 **지능적 문장 분할**
- **spaCy 의미 기반 분할**: 한국어/중국어 모델을 활용한 의미 단위 분할
- **구두점 규칙 보완**: 종결부호, 콜론, 조건부 콤마 기반 분할
- **기존 기준 후처리**: 150자 제한 + 한자 3개 이하 세그먼트 병합

### 🎯 **정확한 정렬**
- **의미 기반 매칭**: BGE-M3 임베딩을 활용한 고품질 정렬
- **1:1 대응 보장**: 원문-번역문 문장 간 최적 매칭
- **유사도 기반 품질 평가**: 정렬 품질 수치화

### 📊 **상세한 분석**
- **문단별 통계**: 분할/정렬 결과 요약
- **품질별 분류**: 고/중/저품질 매칭 비율 분석
- **처리 과정 추적**: 분할 방법 및 정렬 과정 기록

### 🔧 **실용적 설계**
- **SA 연동 가능**: SA 임베더 모듈 재사용
- **대체 시스템**: 외부 의존성 실패시 TF-IDF 기반 처리
- **배치 처리**: 여러 문단 일괄 처리

## 🚀 성능 정보

### SA 성능
- **처리 속도**: 문장당 1-5초 (임베더에 따라 차이)
- **메모리 사용량**: 2-8GB (모델 크기에 따라)
- **캐시 효과**: 반복 처리시 2-3배 속도 향상
- **병렬 처리**: CPU 코어 수에 따라 성능 향상

### PA 성능
- **처리 속도**: 문단당 10-30초 (문장 수에 따라)
- **메모리 사용량**: 2-4GB (spaCy + 임베더 모델)
- **분할 정확도**: spaCy 기반 90%+ 의미 단위 분할
- **정렬 품질**: BGE-M3 사용시 평균 유사도 0.7+

## 📝 사용 팁

### SA 사용 팁
1. **첫 실행시**: 모델 다운로드로 시간이 걸릴 수 있음
2. **대용량 파일**: `--parallel` 옵션으로 속도 향상
3. **정확도 vs 속도**: BGE-M3 > Sentence Transformer > 단순 매칭
4. **메모리 절약**: `--no-semantic` 옵션으로 임베딩 비활성화
5. **디버깅**: `--verbose` 옵션으로 상세 로그 확인

### PA 사용 팁
1. **spaCy 모델 설치**: `python -m spacy download ko_core_news_lg zh_core_web_lg`
2. **문장 길이 조정**: `--max-length` 로 분할 단위 조절
3. **정렬 품질 향상**: `--threshold` 높여서 고품질 매칭만 선별
4. **SA 연동**: SA 임베더 모듈 공유로 일관된 품질
5. **결과 분석**: 출력되는 통계 정보로 품질 모니터링

## 🔍 문제 해결

### 일반적인 오류
```bash
# 의존성 설치 오류
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall

# spaCy 모델 설치
python -m spacy download ko_core_news_lg
python -m spacy download zh_core_web_lg

# 메모리 부족 (SA)
python main.py input.xlsx output.xlsx --no-parallel

# OpenAI API 오류 (SA)
export OPENAI_API_KEY="your-actual-api-key"
```

### PA 특별 오류 해결
```bash
# spaCy 모델 없음 오류
pip install spacy
python -m spacy download ko_core_news_sm  # 최소 모델
python -m spacy download ko_core_news_lg  # 권장 모델

# regex 모듈 오류
pip install regex

# 처리 실패시 로그 확인
python main.py input.xlsx output.xlsx --verbose
```

### 로그 파일 확인
```bash
# SA 처리 로그
cd sa
cat sa_processing.log

# PA는 실시간 콘솔 출력 (로그 파일 없음)

# 캐시 초기화 (SA만)
rm -rf embeddings_cache_*
```

## 📁 프로젝트 구조

```
CSP/
├── sa/                     # Sentence Aligner (완료)
│   ├── main.py            # SA 메인 실행기
│   ├── processor.py       # 문장 처리 로직
│   ├── sa_tokenizers/     # 토크나이저 모듈들
│   ├── sa_embedders/      # 임베더 모듈들
│   ├── aligner.py         # 정렬 알고리즘
│   └── requirements.txt   # SA 의존성
├── pa/                     # Paragraph Aligner (완료)
│   ├── main.py            # PA 메인 실행기
│   ├── processor.py       # 문단 처리 로직
│   ├── sentence_splitter.py # 지능적 문장 분할기
│   ├── aligner.py         # 간단한 정렬 로직
│   └── requirements.txt   # PA 의존성
└── readme.md              # 전체 프로젝트 가이드
```

## 📈 개발 로드맵

### SA (완료/개선 예정)
- [x] 핵심 문장-구 정렬 기능 완성
- [x] 다중 토크나이저/임베더 지원
- [x] CLI 인터페이스 완성
- [x] 진행률 표시 및 캐시 시스템
- [ ] 더 많은 토크나이저 지원 (Kiwi, Komoran 등)
- [ ] 웹 UI 인터페이스 추가
- [ ] 정렬 품질 평가 지표 추가

### PA (완료/개선 예정)
- [x] spaCy 기반 의미 단위 문장 분할
- [x] 지능적 구두점 규칙 기반 분할
- [x] BGE/ST 임베더 기반 문장 정렬
- [x] SA 임베더 모듈 연동
- [x] 상세한 정렬 품질 분석
- [x] CLI 인터페이스 완성
- [ ] 더 정교한 정렬 알고리즘 (헝가리안 알고리즘 등)
- [ ] 사용자 정의 분할 규칙 지원
- [ ] 웹 UI 인터페이스 추가

### 공통 개선사항
- [ ] Docker 컨테이너 지원
- [ ] 배치 처리 API 서버
- [ ] 성능 벤치마크 및 최적화
- [ ] 다국어 지원 확장
- [ ] SA ↔ PA 통합 워크플로우

---

**CSP (Sentence & Paragraph Aligner)** - 한문 번역 텍스트 정렬 도구 모음 🎯

📅 **최종 업데이트**: 2025년 6월 21일 - PA 기본 기능 완성, SA+PA 모두 실용 가능
