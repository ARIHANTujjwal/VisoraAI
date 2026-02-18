import torch
import torch.nn as nn


class CRNN(nn.Module):
    """
    CRNN: CNN -> sequence -> BiLSTM -> classifier
    Output shape: (T, B, C) where C = num_classes (includes CTC blank)
    """

    def __init__(self, num_classes: int, img_h: int = 32):  # constructor
        super().__init__()
        assert img_h in (32, 48), "Use 32 or 48 for simplicity."
        self.img_h = img_h
        # CNN feature extractor (simple and fast)
        # imput: (batch_size, channels, height, width)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(True), # 64 filters, 3x3 kernel, stride 1, padding 1
            nn.MaxPool2d(2, 2),  # H/2, W/2

            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # H/4, W/4

            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # H/8, W/4

            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(True),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d((2, 1), (2, 1)),  # H/16, W/4

            nn.Conv2d(512, 512, 2, 1, 0), nn.ReLU(True),  # reduce H to 1 if H=32
        )

        # After CNN, we expect feature map height to be 1
        self.rnn = nn.LSTM(
            input_size=512,  # CNN output channel [512 numbers describing visual features]
            hidden_size=256, # compresses information into 256 numbers, it is like mempry size
            num_layers=2,    # Layer 2 reads the output of layer 1, this allows deeper understanding of patterns
            bidirectional=True, # read left to right
            batch_first=False   # 
        )
        self.fc = nn.Linear(256 * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W)
        feat = self.cnn(x)  # (B, 512, 1, W')
        b, c, h, w = feat.shape
        assert h == 1, f"Expected CNN output height=1, got {h}"
        feat = feat.squeeze(2)         # (B, 512, W')
        feat = feat.permute(2, 0, 1)   # (W', B, 512) => time-major for CTC
        seq, _ = self.rnn(feat)        # (T, B, 512)
        logits = self.fc(seq)          # (T, B, num_classes)
        return logits
