## 20250614 의미 기반 병렬 구분할

### split_s
- 소규모 테스트 완료
- 문장을 구로 분할 및 1:1 대응
- 한국어 토크나이저 : soynlp, kkma
- 벡터 임베더 : bge-m3, openai, huggingface
```
  python main.py input_s.xlsx output_s.xlsx --parallel --workers 4
```
### split_p
- 단락을 문장으로 분할
- 현재 프로토타입