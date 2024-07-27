"""
ML Flow의 모든 기준점은 중간 결과물이다.
기준점으로 각 파트를 모듈화하고, 이를 통해 테스트를 진행한다.
"""


class MockTokenizer:
    def __init__(self):
        pass

    def encode(self, sentence):
        return {
            'input_ids': [101, 2003, 1996, 2034, 14758, 1997, 13372, 102],
            'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]
        }


class MockLMEncoder:
    def encode(self, input_ids, attention_mask):
        return [0.1, 0.2, 0.3, 0.4]


class MockAEEncoder:
    def encode(self, lm_encoded_output):
        return [0.5, 0.6]


class MockDiffusionEncoder:
    def encode(self, ae_encoded_output):
        return [0.7, 0.8]


class MockAEDecoder:
    def decode(self, encoded_input):
        return [0.9, 0.8, 0.7]


class MockLMDecoder:
    def decode(self, ae_decoded_output):
        return "This is a reconstructed sentence."


class MockWorkflow:
    def __init__(self):
        self.tokenizer = MockTokenizer()
        self.lm_encoder = MockLMEncoder()
        self.ae_encoder = MockAEEncoder()
        self.diffusion_encoder = MockDiffusionEncoder()
        self.ae_decoder = MockAEDecoder()
        self.lm_decoder = MockLMDecoder()

    def process(self, sentence):
        # 입력 텍스트를 토크나이즈
        tokenized = self.tokenizer.encode(sentence)
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']
        # LM 인코딩
        lm_encoded = self.lm_encoder.encode(input_ids, attention_mask)
        # AE 인코딩
        ae_encoded = self.ae_encoder.encode(lm_encoded)
        # Diffusion 인코딩
        diffusion_encoded = self.diffusion_encoder.encode(ae_encoded)
        # AE 디코딩
        ae_decoded = self.ae_decoder.decode(diffusion_encoded)
        # LM 디코딩
        final_sentence = self.lm_decoder.decode(ae_decoded)
        return final_sentence
