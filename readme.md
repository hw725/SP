# CSP: Sentence & Paragraph Aligner

한문-한국어 번역 텍스트의 자동 정렬 도구입니다.

---

## 🖥️ GUI 실행파일 제공

- **pa_gui.exe** : 문단(Paragraph) 정렬용 GUI
- **sa_gui.exe** : 문장(Sentence) 정렬용 GUI

> 최신 실행파일은 [Releases](https://github.com/hw725/CSP/releases)에서 다운로드하세요.

### 사용법
1. exe 파일을 더블클릭하여 실행
2. 입력/출력 파일 및 옵션 선택 후 "실행" 클릭
3. 결과 파일이 지정한 위치에 저장됨
4. 파이썬 설치 불필요

---

## 주요 기능

- **SA**: 문장/구 단위 정렬 (원문 토크나이저는 jieba 고정, 번역문 토크나이저 선택 가능)
- **PA**: 문단 → 문장 분할 및 정렬 (spaCy/stanza 기반)
- 다양한 임베더 지원: SentenceTransformer, BGE-M3, OpenAI(모델/키 직접 선택)
- 실시간 진행률, 캐시, 상세 로그 지원

---

## CLI 사용법 (고급 사용자용)

```bash
# SA 예시
python main.py input.xlsx output.xlsx --tokenizer mecab --embedder openai --openai-model text-embedding-3-large --openai-api-key sk-xxxx

# PA 예시
python main.py input.xlsx output.xlsx --embedder bge --max-length 180 --threshold 0.35
```

---

## 설치 및 개발자용 실행

```bash
# 가상환경 및 의존성 설치
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# SA/PA 폴더에서 직접 실행도 가능
python sa/main.py ...
python pa/main.py ...
```

---

## 참고

- 실행파일 용량이 크지만, 모든 라이브러리 포함으로 별도 설치 불필요
- OpenAI 임베더 사용 시 모델명/키를 GUI에서 직접 입력 가능
- 입력/출력은 Excel(xlsx) 파일만 지원

---

## 문제 해결

- 오류 발생 시 [Issues](https://github.com/your-repo/issues)로 문의
- spaCy/stanza 모델은 최초 실행 시 자동 다운로드

---

**최신 GUI 실행파일로 누구나 쉽게 CSP를 사용할 수 있습니다.**
