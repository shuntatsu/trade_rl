import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MambaProxyExtractor(BaseFeaturesExtractor):
    """
    Feature Extractor mimicking Mamba (Sequence Modeling) structure.
    Currently uses GRU as a stable proxy for Windows/CPU compatibility.

    Structure:
    - Sequence Input (Lookback): -> [Reshape (N, 1)] -> GRU -> [Hidden]
    - Scalar Input (Micro, MTF, Account): -> MLP -> [Hidden]
    - Fusion: Concat -> Output
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 128,
        n_lookback: int = 100,
        d_model: int = 64,
        n_layers: int = 2,
    ):
        super().__init__(observation_space, features_dim)

        self.n_lookback = n_lookback
        input_dim = observation_space.shape[0]
        self.scalar_dim = input_dim - n_lookback

        # Sequence Processor (Mocking Mamba/SSM)
        # Input: (Batch, SeqLen, 1)
        self.sequence_model = nn.GRU(
            input_size=1,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.1 if n_layers > 1 else 0.0,
        )
        self.d_model = d_model

        # Scalar Processor
        self.scalar_mlp = nn.Sequential(
            nn.Linear(self.scalar_dim, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU()
        )

        # Fusion
        fusion_input_dim = d_model + 32
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, features_dim), nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Split Observation
        # shape: (Batch, TotalDim)

        # 1. Sequence Part (First n_lookback elements)
        seq_data = observations[:, : self.n_lookback]
        # Reshape to (Batch, SeqLen, InputSize=1)
        seq_data = seq_data.view(-1, self.n_lookback, 1)

        # Pass through Sequence Model
        # GRU returns (output, h_n). We take last hidden state or max pool?
        # Usually last output is fine for causal models.
        out, _ = self.sequence_model(seq_data)
        # out: (Batch, SeqLen, Hidden)
        seq_feat = out[:, -1, :]  # Last step features

        # 2. Scalar Part (Remaining)
        scalar_data = observations[:, self.n_lookback :]
        scalar_feat = self.scalar_mlp(scalar_data)

        # 3. Fusion
        combined = torch.cat([seq_feat, scalar_feat], dim=1)
        return self.fusion_head(combined)
