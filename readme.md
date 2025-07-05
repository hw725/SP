# CSP: Sentence & Paragraph Aligner

한문-한국어 번역 텍스트의 자동 정렬 CLI 도구 (SA/PA)

---

## 주요 특징
- **모듈 구조 개선**: 공통 모듈(`io_utils.py`, `aligner.py`, `processor.py`, `punctuation.py`, `embedders`, `tokenizers`)이 `core` 디렉토리로 이동하여 코드 재사용성 및 관리 용이성 향상.
- SA: 문장/구 단위 정렬 (원문: jieba, 번역문: mecab)
- PA: 문단→문장 분할 및 정렬 (spaCy는 번역문 분할에만 사용, 원문은 의미적 병합만 사용)
- **원문(한문/중문)은 항상 공백/구두점 기준 토큰화 후, 임베딩 유사도 기반으로 번역문 개수에 맞게 의미적으로 병합하며, 어절 내부에서 분할되지 않도록 로직 개선.**
- spaCy 기반 원문 분할, huggingface/transformers/sentence-transformers 등 모든 불필요한 코드/의존성/분기 완전 제거
- 지원 임베더: BGE-M3 (기본값), OpenAI(모델/키 직접 선택)
- **OpenAI API 키는 시스템 환경 변수 `OPENAI_API_KEY`를 통해 설정 가능.**
- 실시간 진행률(터미널 tqdm/GUI progress bar), 캐시, 상세 로그 지원
- CLI/GUI 환경에서 대용량/고품질 정렬에 최적화
- 경량화: 성능이 우수한 임베더(BGE, OpenAI)와 토크나이저(jieba, mecab)만 남기고 경량화
- **입출력 형식 유연성**: 입력 파일이 Excel(xlsx)이더라도 출력 파일을 CSV나 TXT 형식으로 지정 가능.

---

## CLI 사용법 (권장)
```bash
# SA 예시 (문장/구 단위 정렬)
poetry run python sa/main.py input.xlsx output.xlsx --tokenizer mecab --embedder bge --min-tokens 2 --max-tokens 10

# OpenAI 임베더 사용 (API 키는 환경 변수에서 자동 로드)
poetry run python sa/main.py input.xlsx output.xlsx --embedder openai --openai-model text-embedding-3-large

# PA 예시 (문단→문장 정렬)
poetry run python pa/main.py input.xlsx output.xlsx --embedder bge --max-length 180 --threshold 0.35

# 출력 파일을 CSV로 지정 (PA 예시)
poetry run python pa/main.py input.xlsx output.csv --embedder bge

# 출력 파일을 TXT로 지정 (SA 예시)
poetry run python sa/main.py input.xlsx output.txt --embedder openai
```

---

## 환경설정 및 설치 (Poetry 기반)
1. **가상환경 생성 및 활성화**
   - venv, conda 등 어떤 가상환경이든 사용 가능 (권장)
   - 예시(venv):
     ```bash
     python -m venv venv
     # Windows
     venv\Scripts\activate
     # Linux/WSL
     source venv/bin/activate
     ```
   - 예시(conda):
     ```bash
     conda create -n csp python=3.10
     conda activate csp
     ```
2. **Poetry 설치**
   - Poetry가 없다면 먼저 설치:
     ```bash
     pip install poetry
     # 또는 공식 설치법 참고: https://python-poetry.org/docs/#installation
     ```
3. **의존성 설치 (Poetry 사용)**
   - 프로젝트 루트에서:
     ```bash
     poetry install
     ```
   - poetry가 자동으로 pyproject.toml/poetry.lock 기반 모든 패키지 설치
   - poetry 환경에서 실행하려면:
     ```bash
     poetry run python pa/main.py ...
     poetry run python sa/main.py ...
     ```
   - poetry shell로 진입해도 됨:
     ```bash
     poetry shell
     python pa/main.py ...
     ```
4. **mecab 사용자 사전/한자어 지원 적용 방법**
   - stuser.dic은 표준국어대사전에서 한자어만 추출하여 만든 mecab 사용자 사전으로, 이번 업데이트에 함께 제공
    1. 사용자 사전 csv(stuser.csv)를 mecab-dict-index로 컴파일하여 stuser.dic 생성
        - mecab-dict-index 실행 파일이 있는 경로로 이동 후 아래 명령어 실행
        - 예시(Windows/Linux/WSL):
        ```bash
        mecab-dict-index -d <mecab-ko-dic 경로> -u stuser.dic -f UTF-8 -t UTF-8 <stuser.csv 경로>
        ```
        - stuser.csv는 UTF-8 인코딩, mecab-ko-dic 표준 csv 포맷 필요
        - 생성된 stuser.dic을 mecab-ko-dic 폴더 또는 원하는 경로에 복사
    2. Python 코드에서 -u 옵션으로 경로 지정
        ```python
        tagger = MeCab.Tagger('-d <mecab-ko-dic 경로> -u <stuser.dic 경로>')
        # 예시:
        # tagger = MeCab.Tagger('-d c:/.../mecab-ko-dic -u c:/.../stuser.dic')
        ```
   - 여러 사용자 사전을 함께 쓰고 싶으면 csv를 미리 병합하여 하나의 dic으로 컴파일
   - stuser.dic을 mecab-ko-dic 폴더에 복사하면 -u stuser.dic처럼 파일명만 지정해도 됨
   - 현재는 기본 mecab-ko-dic 또는 직접 생성한 사용자 사전만 사용 가능
5. **spaCy 모델 설치 (최초 1회, 번역문 분할에만 필요)**
   ```bash
   poetry run python -m spacy download ko_core_news_lg
   poetry run python -m spacy download zh_core_web_lg
   ```
6. **(GPU 사용 시) PyTorch CUDA wheel 별도 설치**
   - poetry 환경에서 아래 명령 실행:
     ```bash
     poetry run pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
     ```
7. **(CPU 사용 시) PyTorch CPU wheel 별도 설치**
   - poetry 환경에서 아래 명령 실행:
     ```bash
     poetry run pip install torch==2.7.1+cpu torchvision==0.22.1+cpu torchaudio==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu
     ```
   - 위 명령은 CUDA가 없는 환경(일반 PC/서버/노트북 등)에서 사용하세요.

---

## 개발/실행 예시
```bash
# SA/PA 폴더에서 직접 실행
python sa/main.py ...
python pa/main.py ...
```
- 입력/출력은 Excel(xlsx), CSV, TXT 파일 모두 지원
- OpenAI 임베더 사용 시 모델명/키를 CLI 옵션으로 직접 입력하거나 환경 변수 사용
- 진행률/에러/로그는 CLI(터미널 tqdm) 및 GUI(진행률 바) 모두 실시간 표시

---

## 실행파일/GUI 안내
- sa_gui.py, pa_gui.py 등 GUI는 직접 실행 가능
- 실행파일(.exe)은 추후 릴리즈 예정
- GUI에서는 진행률 바(progress bar)로 실시간 진행 상황 확인 가능
- 최신 실행파일 및 한자어 mecab 사용자사전 지원은 Releases에서 추후 확인

---

## 문제 해결
- 오류 발생 시 Issues로 문의
- spaCy 모델은 최초 실행 시 자동 다운로드 (수동 설치 권장, 번역문 분할에만 필요)
- mecab 관련 ImportError 발생 시 mecab-python3 설치 여부 확인
- huggingface/transformers/sentence-transformers 등은 더 이상 필요 없음

---

## 샘플 데이터 안내
- pa/input.xlsx, sa/input01.xlsx, sa/input02.xlsx 등은 샘플 입력 파일
- 실제 배포/설치 시에는 샘플 데이터만 포함, 작업 결과물(output.xlsx 등)은 미포함
- 샘플 입력 파일을 활용해 정렬 기능을 바로 테스트할 수 있음
- 실제 데이터로 작업할 때는 샘플 파일을 복사하거나, 동일한 형식의 Excel 파일을 사용

---

## PA(문단→문장 정렬) 샘플 입력/출력 예시
**입력 파일(Excel, xlsx):**

| 문단(원문) | 문단(번역문) |
|:-----------|:------------|
| 子曰 學而時習之 不亦說乎. 有朋自遠方來 不亦樂乎. 人不知而不慍 不亦君子乎. | 공자께서 말씀하셨다. 배우고 때때로 익히면 또한 기쁘지 아니한가. 벗이 먼 곳에서 찾아오면 또한 즐겁지 아니한가. 남이 알아주지 않아도 성내지 않으면 또한 군자가 아니겠는가. |

**출력 파일(Excel, xlsx):**

| 문단ID | 문장ID | 원문(분할) | 번역문(분할) |
|:-------|:-------|:-----------|:-------------|
| 1 | 1 | 子曰 學而時習之 不亦說乎 | 공자께서 말씀하셨다. 배우고 때때로 익히면 또한 기쁘지 아니한가 |
| 1 | 2 | 有朋自遠方來 不亦樂乎 | 벗이 먼 곳에서 찾아오면 또한 즐겁지 아니한가 |
| 1 | 3 | 人不知而不慍 不亦君子乎 | 남이 알아주지 않아도 성내지 않으면 또한 군자가 아니겠는가 |

- **원문(한문/중문)은 spaCy로 분할하지 않고, 항상 공백/구두점 기준 토큰화 후 의미적으로 병합**
- 번역문(한국어/중국어)은 spaCy로 의미 단위 분할

---

## SA(문장/구 정렬) 샘플 입력/출력 예시
**입력 파일(Excel, xlsx):**

| 원문(샘플) | 번역문(샘플) |
|:-----------|:------------|
| 子曰 學而時習之 不亦說乎 | 공자께서 말씀하셨다. 배우고 때때로 익히면 또한 기쁘지 아니한가 |
| 有朋自遠方來 不亦樂乎 | 벗이 먼 곳에서 찾아오면 또한 즐겁지 아니한가 |
| 人不知而不慍 不亦君子乎 | 남이 알아주지 않아도 성내지 않으면 또한 군자가 아니겠는가 |

**출력 파일(Excel, xlsx, 예: output01_phrase.xlsx):**

| 문장식별자 | 구식별자 | 원문구 | 번역구 |
|:----------|:--------|:-------|:-------|
| 1 | 1 | 子曰 | 공자께서 말씀하셨다 |
| 1 | 2 | 學而時習之 | 배우고 때때로 익히면 |
| 1 | 3 | 不亦說乎 | 또한 기쁘지 아니한가 |
| 2 | 1 | 有朋自遠方來 | 벗이 먼 곳에서 찾아오면 |
| 2 | 2 | 不亦樂乎 | 또한 즐겁지 아니한가 |
| 3 | 1 | 人不知而不慍 | 남이 알아주지 않아도 성내지 않으면 |
| 3 | 2 | 不亦君子乎 | 또한 군자가 아니겠는가 |

---

CLI 환경에서 대용량 한중문 정렬, 고품질 임베딩 연동, 환경별 설치/실행법을 모두 지원
