"""
AASIST-style Audio Anti-Spoofing Model.
Reference: Jung et al., "AASIST: Audio Anti-Spoofing using Integrated
           Spectro-Temporal Graph Attention Networks", Interspeech 2022.

Architecture:
1. Spectro-Temporal Feature Encoder:
   - Sinc/Bandpass Convolutions & Residual 2D blocks mapping raw audio to spectral & temporal nodes.
2. Heterogeneous Spectro-Temporal Graph Attention:
   - Spectral Graph Attention (S-GAT): Models sub-band frequency artifacts.
   - Temporal Graph Attention (T-GAT): Models inter-frame temporal phase incoherence.
   - Cross-Branch Multi-Head Attention Fusion.
3. Graph Readout & Projection Head:
   - Graph Attention Pooling -> Scalar Logit output (0=REAL, 1=FAKE).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SincConv(nn.Module):
    """
    Parametric Sinc-convolution layer to learn auditory filterbanks directly from speech.
    """
    def __init__(self, out_channels=70, kernel_size=129, sample_rate=16000, min_low_hz=50, min_band_hz=50):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        if kernel_size % 2 == 0:
            kernel_size += 1
            self.kernel_size = kernel_size

        # Initialize filter frequencies according to Mel scale
        low_hz = 30
        high_hz = sample_rate / 2 - (min_low_hz + min_band_hz)
        mel = torch.linspace(self.to_mel(low_hz), self.to_mel(high_hz), out_channels + 1)
        hz = self.to_hz(mel)

        self.low_hz_ = nn.Parameter(hz[:-1].unsqueeze(1))
        self.band_hz_ = nn.Parameter((hz[1:] - hz[:-1]).unsqueeze(1))

        # Hamming window buffer
        n = torch.linspace(0, (self.kernel_size / 2) - 1, int(self.kernel_size / 2))
        window = 0.54 - 0.46 * torch.cos(2 * math.pi * n / self.kernel_size)
        self.register_buffer("window", window)

        t_right = torch.linspace(1, (self.kernel_size / 2) - 1, int((self.kernel_size / 2) - 1)) / sample_rate
        self.register_buffer("t_right", t_right.unsqueeze(0))

    @staticmethod
    def to_mel(hz):
        return 2595 * math.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def forward(self, x):
        # x: [Batch, Samples]
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, self.t_right)
        f_times_t_high = torch.matmul(high, self.t_right)

        band_pass_left = ((torch.sin(2 * math.pi * f_times_t_high) - torch.sin(2 * math.pi * f_times_t_low)) / (self.t_right * math.pi)).flip(dims=[-1])
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = (torch.sin(2 * math.pi * f_times_t_high) - torch.sin(2 * math.pi * f_times_t_low)) / (self.t_right * math.pi)

        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass / (2 * band[:, None])

        filters = (band_pass * torch.cat([self.window.flip(0)[:-1], self.window], dim=0)).unsqueeze(1)
        # 1D Convolution over input speech
        return F.conv1d(x.unsqueeze(1), filters, stride=10, padding=self.kernel_size // 2)


class ResidualBlock2D(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Sequential()
        if in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + res)
        return out


class SpectroTemporalGraphAttention(nn.Module):
    """
    Dual-branch Graph Attention modeling spectral artifact correlations and temporal phase transitions.
    """
    def __init__(self, feature_dim=64):
        super().__init__()
        self.spectral_gat = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=4, batch_first=True)
        self.temporal_gat = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        # x: [Batch, Channels, Freq, Time]
        B, C, F_dim, T_dim = x.shape

        # 1. Spectral Nodes: Pool over Time -> [Batch, Freq, Channels]
        spec_nodes = x.mean(dim=-1).transpose(1, 2)
        s_att, _ = self.spectral_gat(spec_nodes, spec_nodes, spec_nodes)
        spec_out = self.norm1(spec_nodes + s_att).mean(dim=1)  # [Batch, C]

        # 2. Temporal Nodes: Pool over Freq -> [Batch, Time, Channels]
        temp_nodes = x.mean(dim=2).transpose(1, 2)
        t_att, _ = self.temporal_gat(temp_nodes, temp_nodes, temp_nodes)
        temp_out = self.norm2(temp_nodes + t_att).mean(dim=1)  # [Batch, C]

        # 3. Graph Readout Fusion
        fused = torch.cat([spec_out, temp_out], dim=-1)
        out = self.ffn(fused)
        return out


class AASISTModel(nn.Module):
    """
    Integrated AASIST Anti-Spoofing Architecture for In-The-Wild Voice Deepfake Detection.
    """
    def __init__(self, sample_rate=16000):
        super().__init__()
        self.sinc_conv = SincConv(out_channels=70, kernel_size=129, sample_rate=sample_rate)
        self.bn_sinc = nn.BatchNorm1d(70)

        # 2D Spectro-Temporal Residual Representation
        self.res1 = ResidualBlock2D(1, 32)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.res2 = ResidualBlock2D(32, 64)
        self.pool2 = nn.MaxPool2d((2, 2))

        # Integrated Spectro-Temporal Graph Attention
        self.graph_attention = SpectroTemporalGraphAttention(feature_dim=64)

        # Output Classifier
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: [Batch, 64000]
        sinc_feats = F.relu(self.bn_sinc(self.sinc_conv(x)))  # [Batch, 70, 6400]
        sinc_2d = sinc_feats.unsqueeze(1)  # [Batch, 1, 70, 6400]

        # Downsample temporal resolution
        sinc_2d = F.adaptive_avg_pool2d(sinc_2d, (70, 400))  # [Batch, 1, 70, 400]

        feat = self.pool1(self.res1(sinc_2d))  # [Batch, 32, 35, 200]
        feat = self.pool2(self.res2(feat))     # [Batch, 64, 17, 100]

        graph_embedding = self.graph_attention(feat)  # [Batch, 64]
        logits = self.head(graph_embedding).squeeze(-1)  # [Batch]
        return logits
