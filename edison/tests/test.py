from edison.tests.mock import (
    MockTokenizer,
    MockLMEncoder,
    MockAEEncoder,
    MockDiffusionEncoder,
    MockAEDecoder,
    MockLMDecoder,
    MockWorkflow,
)


def test_tokenizer():
    mock_tokenizer = MockTokenizer()
    test_sentence = "This is a test sentence."
    result = mock_tokenizer.encode(test_sentence)
    assert result['input_ids'] == [101, 2003, 1996, 2034, 14758, 1997, 13372, 102], "Input IDs do not match"
    assert result['attention_mask'] == [1, 1, 1, 1, 1, 1, 1, 1], "Attention mask does not match"
    print("Tokenization test passed")


def test_lm_encoder():
    mock_lm_encoder = MockLMEncoder()
    lm_output = mock_lm_encoder.encode(input_ids=[1, 2, 3], attention_mask=[1, 1, 1])
    assert lm_output == [0.1, 0.2, 0.3, 0.4], "LM Encoding output mismatch"
    print("LM Encoding test passed")


def test_ae_encoder():
    mock_ae_encoder = MockAEEncoder()
    ae_output = mock_ae_encoder.encode(lm_encoded_output=[0.1, 0.2, 0.3, 0.4])
    assert ae_output == [0.5, 0.6], "AE Encoding output mismatch"
    print("AE Encoding test passed")


def test_diffusion_encoder():
    mock_diffusion_encoder = MockDiffusionEncoder()
    diffusion_output = mock_diffusion_encoder.encode(ae_encoded_output=[0.5, 0.6])
    assert diffusion_output == [0.7, 0.8], "Diffusion Encoding output mismatch"
    print("Diffusion Encoding test passed")


def test_ae_decoder():
    mock_ae_decoder = MockAEDecoder()
    decoded_output = mock_ae_decoder.decode(encoded_input=[0.5, 0.6])
    assert decoded_output == [0.9, 0.8, 0.7], "AE Decoding output mismatch"
    print("AE Decoding test passed")


def test_lm_decoder():
    mock_lm_decoder = MockLMDecoder()
    sentence_output = mock_lm_decoder.decode(ae_decoded_output=[0.9, 0.8, 0.7])
    assert sentence_output == "This is a reconstructed sentence.", "LM Decoding output mismatch"
    print("LM Decoding test passed")


def test_full_workflow():
    mock_workflow = MockWorkflow()
    input_sentence = "This is a test sentence."
    final_output = mock_workflow.process(input_sentence)

    assert final_output == "This is a reconstructed sentence.", "Final output mismatch"
    print("Full workflow test passed")
