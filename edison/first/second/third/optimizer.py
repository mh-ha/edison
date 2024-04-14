import torch


class Adam(torch.optim.Adam):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, **kwargs):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, **kwargs)

class AdamW(torch.optim.AdamW):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01, amsgrad=False, **kwargs):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, **kwargs)