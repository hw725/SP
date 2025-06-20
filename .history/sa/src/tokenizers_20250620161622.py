def improved_mecab_tokenize(self, text: str) -> List[str]:
    """개선된 MeCab 토크나이징"""
    if not text.strip():
        return []
    
    try:
        # 기존 토크나이징
        tokens = self.tokenizer.morphs(text)
        
        # 의미 단위 병합 규칙
        merged_tokens = []
        temp_compound = ""
        
        for i, token in enumerate(tokens):
            # 한자어 연속 처리
            if self._is_hanja(token):
                temp_compound += token
                # 다음 토큰이 조사나 어미인 경우
                if i + 1 < len(tokens) and self._is_particle(tokens[i + 1]):
                    temp_compound += tokens[i + 1]
                    merged_tokens.append(temp_compound)
                    temp_compound = ""
                    i += 1  # 다음 토큰 스킵
                elif i + 1 >= len(tokens) or not self._is_hanja(tokens[i + 1]):
                    merged_tokens.append(temp_compound)
                    temp_compound = ""
            else:
                if temp_compound:
                    merged_tokens.append(temp_compound)
                    temp_compound = ""
                merged_tokens.append(token)
        
        return merged_tokens
        
    except Exception as e:
        logger.warning(f"MeCab 토크나이징 실패: {e}")
        return text.split()

def _is_hanja(self, token: str) -> bool:
    """한자 여부 판단"""
    return bool(re.search(r'[\u4e00-\u9fff]', token))

def _is_particle(self, token: str) -> bool:
    """조사/어미 여부 판단"""
    particles = ['은', '는', '이', '가', '을', '를', '에', '에서', '으로', '로', '와', '과']
    return token in particles or len(token) <= 2