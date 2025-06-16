## 20250616 의미 기반 병렬 구분할

### SA : Sentence Aligner
- 소규모 테스트 완료
- 문장을 구로 분할 및 1:1 대응
- 토크나이저 : mecab, jieba 등
- 벡터 임베더 : bge-m3 등
- tokenizers.py, embedders.py에 여러 라이브러리를 클래스로 추가해서 교체 가능
- 실행 순서
```
  venv/scripts/activate
    # 가상 환경 활성화(권장) window 기준
  pip install -r requirements.txt
    # pip install --upgrade pip 필요할 수 있음
  python main.py input_s.xlsx output_s.xlsx --parallel --workers 4
    # 메모리가 부족하면 workers 수를 낮출 것
```

### PA : Paragraph Aligner
- 단락을 문장으로 분할
- 현재 프로토타입, SA 작업으로 지연됨
- SA 로직을 vice versa로 적용 예정
