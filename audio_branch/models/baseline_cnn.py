"""
Baseline Audio Deepfake Detection Model.
Log-Mel Spectrogram Frontend + 2D Convolutional Neural Network.
"""

import torch
import torch.nn as nn
import librosa


class BaselineAudioCNN(nn.Module):
    def __init__(self, n_mels=128, n_fft=1024, hop_length=512, sample_rate=16000):
        super().__init__()
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate

        # Precompute Mel Filterbank buffer
        mel_fb = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=20, fmax=sample_rate // 2)
        self.register_buffer("mel_fb", torch.from_numpy(mel_fb).float())
        self.register_buffer("window", torch.hann_window(n_fft))

        # CNN Feature Extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def extract_features(self, x):
        """Extracts Log-Mel Spectrogram from raw waveform tensor [Batch, Samples]."""
        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            return_complex=True
        )
        mag = torch.abs(stft)  # [Batch, n_fft//2 + 1, Time]
        mel = torch.matmul(self.mel_fb, mag)  # [Batch, n_mels, Time]
        log_mel = torch.log(mel + 1e-6).unsqueeze(1)  # [Batch, 1, n_mels, Time]
        return log_mel

    def forward(self, x):
        features = self.extract_features(x)
        embeddings = self.features(features)
        logits = self.classifier(embeddings).squeeze(-1)
        return logits
