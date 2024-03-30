"""
FLOW:
    1. mask된 input data 생성(train dataloader, collate_fn 이용)

    2. DeBERTa(disentangled attention, Enhanced Mask Decoder): Generator
        - token embedding + segment embedding
        - transformer block
            context -> V_c, K_c, Q_c
            relative PE -> Q_r, K_r
                torch.arange(0, q_size or k_size)
                -> bucket position
                -> rel position
                -> rel embedding
            QK^T = Q_r@K_c + Q_c@K_r + Q_c@K_c
            -> scale (= 1/math.sqrt(dim*scale_factor))
            -> softmax
            -> @V_c
            -> output
            -> linear (+ dropout)
            -> add
            -> layernorm
            -> output  # end attention
            -> linear + activation  # hidden dim -> intermediate dim
            -> linear (+ dropout)  # interm dim -> hidden dim
            -> add
            -> layernorm
            -> output
        - transformer block(-1:)
            output[-2] 사용: L-1개 transformers, 마지막 transformer는 enhanced mask decoder를 위한 것 (masked_language_model.py:59 참조)
            output[-2] + Absolute Position Encoding: query state
            output[-2]: key, value state (hidden state)
            query state, output[-2], rel embedding
            -> output
            -> repeat
            -> output
        - loss 계산
            mask 된 부분만 or 전체
    
    3. electra input data 생성
        - 이거 만드는 과정에서 generator 돌리고 generator loss도 계산
        - mask 된 부분 gen output으로 변경 or 전체 변경
        - input data에서 input_ids 변경 후 출력, 전체에 대한 loss 계산 위해 labels도 변경 해야 할 수 있음
    
    4. DeBERTa(disentangled attention, Enhanced Mask Decoder): Discriminator
        - GDES: Gradient-disentangled weight/embedding sharing
            generator와 embedding 공유
            + disc만의 parameter 더하기
        - 계산 방식은 generator와 같음
            prediction head만 바꿔서 사용 (+ task에 맞는 loss)
"""


# class Generator


# class Discriminator



# class DeBERTa






