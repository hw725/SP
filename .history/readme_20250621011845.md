# CSP: Sentence & Paragraph Aligner

20250620
한문-한국어 번역 텍스트의 단위별 자동 정렬 도구 모음입니다.

## 🎯 프로젝트 구성

### SA : Sentence Aligner (완료)
문장을 의미 단위(구)로 분할하고 1:1 대응시키는 도구

### PA : Paragraph Aligner (개발 중)
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

## 📋 SA CLI 옵션 전체 목록

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

## 🔧 지원 도구

### 토크나이저
- **jieba**: 중국어/한문 토크나이저 (원문 전용)
- **MeCab**: 한국어 형태소 분석기  
- **SoyNLP**: 한국어 비지도 토크나이저
- **Kkma**: 한국어 형태소 분석기

### 임베더  
- **Sentence Transformer**: 범용 문장 임베딩 (기본값)
- **BGE-M3**: 다국어 고성능 임베딩 (추천)
- **OpenAI**: OpenAI API 임베딩 (API 키 필요)
- **HuggingFace**: HuggingFace 모델 임베딩

## 📊 SA 입출력 형식

### 입력 파일 (Excel)
| id | src | tgt |
|----|-----|-----|
| 1 | 蒹葭는 刺襄公也라 | 蒹葭는 襄公을 풍자한 詩이다. |
| 2 | 未能用周禮하니 將無以固其國焉이라 | 周나라의 禮를 따르지 못하니 나라를 견고히 할 수 없다. |

### 출력 파일 1: 분할 결과 (기본)
| id | src | tgt | src_units | tgt_units | alignments |
|----|-----|-----|-----------|-----------|------------|
| 1 | 蒹葭는 刺襄公也라 | 蒹葭는 謫公을... | ['蒹葭는', '刺襄公也라'] | ['蒹葭는', '襄公을...'] | [...] |

### 출력 파일 2: 구 단위 정렬 결과
| 문장식별자 | 구식별자 | 원문구 | 번역구 |
|------------|----------|--------|---------|
| 1 | 1 | 蒹葭는 | 蒹葭는 |
| 1 | 2 | 刺襄公也라 | 襄公을 풍자한 詩이다. |

---

## 🚧 PA (Paragraph Aligner) - 개발 중

### 개요
- **목표**: 단락을 문장으로 분할하고 원문-번역문 문장 간 정렬
- **상태**: 프로토타입 단계 (SA 작업 완료 후 개발 재개)
- **접근법**: SA의 성공한 로직을 문장 단위로 확장 적용

### 예상 기능
```bash
# PA 기본 실행 (예정)
cd pa
python main.py input_paragraph.xlsx output_sentences.xlsx

# 문장 분할 + 정렬
python main.py input.xlsx output.xlsx --sentence-splitter kss --embedder bge
```

### 예상 처리 과정
1. **단락 입력**: 긴 원문/번역문 단락
2. **문장 분할**: 원문/번역문을 각각 문장으로 분할
3. **문장 정렬**: SA와 유사한 의미 기반 정렬
4. **결과 출력**: 문장 단위 1:1 대응 결과

### 개발 계획
- [x] SA 핵심 로직 완성
- [ ] 문장 분할기 통합 (KSS, spaCy 등)
- [ ] 문장 레벨 의미 매칭 알고리즘
- [ ] PA 전용 CLI 인터페이스
- [ ] 성능 최적화 및 테스트

---

## ⚙️ 환경 설정

### OpenAI 임베더 사용시
```bash
# API 키 설정
export OPENAI_API_KEY="your-api-key-here"

# 또는 .env 파일 생성
echo "OPENAI_API_KEY=your-api-key" > .env
```

### 성능 최적화
- **메모리 부족시**: `--parallel` 옵션 제거
- **속도 향상**: `--embedder bge` 사용 (로컬, 고성능)
- **정확도 향상**: `--min-tokens 2 --max-tokens 20` 조정

## 🚀 성능 정보 (SA 기준)

- **처리 속도**: 문장당 1-5초 (임베더에 따라 차이)
- **메모리 사용량**: 2-8GB (모델 크기에 따라)
- **캐시 효과**: 반복 처리시 2-3배 속도 향상
- **병렬 처리**: CPU 코어 수에 따라 성능 향상

## 📝 사용 팁

1. **첫 실행시**: 모델 다운로드로 시간이 걸릴 수 있음
2. **대용량 파일**: `--parallel` 옵션으로 속도 향상
3. **정확도 vs 속도**: BGE-M3 > Sentence Transformer > 단순 매칭
4. **메모리 절약**: `--no-semantic` 옵션으로 임베딩 비활성화
5. **디버깅**: `--verbose` 옵션으로 상세 로그 확인

## 🔍 문제 해결

### 일반적인 오류
```bash
# 의존성 설치 오류
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall

# 메모리 부족
python main.py input.xlsx output.xlsx --no-parallel

# OpenAI API 오류  
export OPENAI_API_KEY="your-actual-api-key"
```

### 로그 파일 확인
```bash
# SA 처리 로그 확인
cd sa
cat sa_processing.log

# 캐시 초기화 (필요시)
rm -rf embeddings_cache_*
```

## 📁 프로젝트 구조

```
SP/
├── sa/                     # Sentence Aligner (완료)
│   ├── main.py            # SA 메인 실행기
│   ├── processor.py       # 문장 처리 로직
│   ├── sa_tokenizers/     # 토크나이저 모듈들
│   ├── sa_embedders/      # 임베더 모듈들
│   ├── aligner.py         # 정렬 알고리즘
│   └── requirements.txt   # SA 의존성
├── pa/                     # Paragraph Aligner (개발 중)
│   └── (개발 예정)
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

### PA (개발 예정)
- [ ] 문장 분할기 통합
- [ ] 단락-문장 정렬 알고리즘 개발
- [ ] PA 전용 CLI 구현
- [ ] SA와 PA 통합 워크플로우

### 공통 개선사항
- [ ] Docker 컨테이너 지원
- [ ] 배치 처리 API 서버
- [ ] 성능 벤치마크 및 최적화
- [ ] 다국어 지원 확장

---

**SP (Sentence & Paragraph Aligner)** - 한문 번역 텍스트 정렬 도구 모음 🎯

📅 **최종 업데이트**: 2025년 6월 20일 - SA 소규모 테스트 완료
