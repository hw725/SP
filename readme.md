## 20250604 의미 기반 병렬 구분할(spacy)
- 모시정의8 테스트 완료
- 마스킹, 언마스킹 적용 버전
- 의존성, 환경, 실행 관련 명령어(root 디렉토리)
  
  ```
  pip install -r requirements.txt
  python -m spacy download ko_core_news_lg
  python main.py input.xlsx output.xlsx
  ```
- 문장분할 마친 상태에서 구분할을 전제로 함
- 현토, 표점 모두 가능
- 가급적 한자 병기 전에 실행
- 군더더기 디버깅 구문은 추후 삭제하고 진행 막대, 완료 메시지 정도만 남길 예정


### 구조
- tokenizer : 원문(공백), 번역문(spacy) 의미 기반 분할
- punctuation : 마스킹, 언마스킹 함수 및 관련 부호 상세 설정
- embedder : 벡터 임베딩
- aligner : 의미 대응 배열
- io_manager : 엑셀 입출력
- main : 모듈 실행
