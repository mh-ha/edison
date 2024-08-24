from abc import ABC, abstractmethod
from typing import Optional, Union, List, Tuple

from torch import nn, Tensor

from edison.layers.positional_embedding import SinusoidalPosEmb


class BaseDiffusion(nn.Module, ABC):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    @abstractmethod
    def forward(
        self,
        latent: Tensor,
        context: Optional[Tensor],
        times: Tensor,
        attention_mask: Optional[Tensor] = None,
        self_cond: Optional[Tensor] = None,
    ) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def encode(
        self,
        latent: Tensor,
        context: Optional[Tensor],
        alpha: Tensor,
        attention_mask: Optional[Tensor] = None,
        self_cond: Optional[Tensor] = None,
    ) -> Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def loss_fn(self):
        raise NotImplementedError

    @abstractmethod
    def training_step(
        self,
        latent: Tensor,
        context: Optional[Tensor],
        attention_mask: Optional[Tensor],
        times: Union[Tensor, float],
    ) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def sample(self, batch_size: int, lengths: List[int]) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def _build_time_mlp(
        self,
        internal_dim: int,
        time_emb_dim: int,
    ) -> nn.Module:
        sinu_pos_emb = SinusoidalPosEmb(internal_dim)
        return nn.Sequential(
            sinu_pos_emb,
            nn.Linear(internal_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

    def _build_time_projection(
        self,
        input_dim: int,
        output_dim: int,
    ) -> nn.Module:
        return nn.Sequential(
            nn.GELU(),
            nn.Linear(input_dim, output_dim),
        )

    def _build_projection(
        self,
        input_dim: int,
        output_dim: int,
    ) -> nn.Module:
        return nn.Linear(input_dim, output_dim)

    def _predict_start_from_v(
        self,
        latent: Tensor,
        alpha: Tensor,
        v: Tensor,
    ) -> Tensor:
        # TODO: 수식 이해하기
        return alpha.sqrt() * latent - (1 - alpha).sqrt() * v

    def _predict_noise_from_v(
        self,
        latent: Tensor,
        alpha: Tensor,
        v: Tensor,
    ) -> Tensor:
        return (1 - alpha).sqrt() * latent + alpha.sqrt() * v

    def _predict_noise_from_start(self, latent, alpha, x0):
        return (latent - alpha.sqrt() * x0) / (1 - alpha).sqrt().clamp(min=1e-8)

    # def _predict_v_from_start_and_eps(self, latent, alpha, x, noise):
    #     return alpha.sqrt() * noise - x * (1 - alpha).sqrt()


class BaseEncoder(nn.Module, ABC):
    def __init__(
        self,
        internal_dim: Tensor,
        depth: int,
    ) -> None:
        super().__init__()
        self.internal_dim = internal_dim
        self.depth = depth

    @abstractmethod
    def forward(
        self,
        latent: Tensor,
        context: Optional[Tensor],
        attention_mask: Optional[Tensor] = None,
        time_emb: Optional[Tensor] = None,
    ) -> Tensor:
        raise NotImplementedError


class BaseAutoEncoder(nn.Module, ABC):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    @abstractmethod
    def forward(
        self,
        encoder_outputs: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def encode(
        self,
        encoder_outputs: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def decode(
        self,
        encoder_outputs: Tensor
    ) -> Tensor:
        raise NotImplementedError
